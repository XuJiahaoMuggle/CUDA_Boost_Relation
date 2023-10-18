#include "mix_mat.hpp"
#include <utils/utils.h>
#include <cuda_boost/contiguous.cuh>
#include <cuda_boost/convert_data_type.cuh>
#include <image_proc/cvt_color.cuh>
#include <cuda_fp16.h>
#include <cstdlib>
#include <algorithm>
#ifdef __cplusplus
extern "C"
{
#endif
#include <libavcodec/avcodec.h>
#ifdef __cplusplus
}
#endif 

namespace tinycv
{
    void MixMat::set_up_data_(std::shared_ptr<BaseMemory> data)
    {
        // if no data is given.
        if (data == nullptr)
            data_ = std::make_shared<BaseMemory>(d_head_.device_id_);
        else
        {
            data_ = data;
            d_head_.device_id_ = data->device_id();
        }

        d_head_.status_ = Status::INIT;
        if (data_->host())
        {
            d_head_.status_ = Status::HOST;
            d_head_.bytes_ = data->host_data_size();
        }

        if (data_->device())
        {
            d_head_.status_ = Status::DEVICE;
            d_head_.bytes_ = data->device_data_size();
        }

    }

    MixMat::MixMat(int n, int c, int h, int w, DataType dtype, std::shared_ptr<BaseMemory> data, int device_id)
    {
        d_head_.set_dtype(dtype);
        d_head_.set_device_id(cuda_tools::get_device_id(device_id));

        set_up_data_(data);
        assert(d_head_.device_id_ != -1);  // device not found.
        d_head_.set_dims({n, c, h, w});
        d_head_.adjust_status();
    }
    
    MixMat::MixMat(const std::vector<int> &dims_array, DataType dtype, std::shared_ptr<BaseMemory> data, int device_id)
    {
        d_head_.set_dtype(dtype);
        d_head_.set_device_id(cuda_tools::get_device_id(device_id));
        set_up_data_(data);
        assert(d_head_.device_id_ != -1);  // device not found.
        d_head_.set_dims(dims_array);
        d_head_.adjust_status();
    }

    MixMat::MixMat(const int n_dims, const int *dims_array, DataType dtype, std::shared_ptr<BaseMemory> data, int device_id)
    {
        d_head_.set_dtype(dtype);
        d_head_.set_device_id(cuda_tools::get_device_id(device_id));
        set_up_data_(data);
        assert(d_head_.device_id_ != -1);  // device not found.

        d_head_.set_dims(std::vector<int> (dims_array, dims_array + n_dims));
        d_head_.adjust_status();
    }

    MixMat &MixMat::release()
    {
        data_.reset();  // ref count - 1.
        d_head_.reset();
        return *this;
    }

    void MixMat::assert_view_(int n_dims, const int *dim_array)
    {
        const std::vector<int> &dim0 = d_head_.dims_;
        int prod_dim0 = 1;
        for (int i = 0; i < dim0.size(); ++i)
            prod_dim0 *= dim0[i];
        int prod_dim_array = 1;
        for (int i = 0; i < n_dims; ++i)
            prod_dim_array *= dim_array[i];
        assert(prod_dim_array == prod_dim0);
    }

    MixMat &MixMat::view(int n_dims, const int *dim_array)
    {
        assert(is_contiguous());
        assert_view_(n_dims, dim_array);
        std::vector<int> buf(n_dims);
        int deduce_idx = -1;  // the index of deduce index.
        for (int i = 0; i < n_dims; ++i)
        {
            int dim = dim_array[i];
            if (dim == -1)  // only one deduce is allowed.
            {
                assert(deduce_idx == -1);
                deduce_idx = i;
            }
            buf[i] = dim;
        }
        // get the deduce index's value.
        if (deduce_idx != -1)
        {
            size_t dim = numel();
            for (int i = 0; i < n_dims; ++i)
            {
                if (i == deduce_idx)
                    continue;
                dim /= buf[i];
            }
            buf[deduce_idx] = dim;
        }
        d_head_.set_dims(buf);
        d_head_.adjust_status();
        return *this;
    }

    MixMat &MixMat::reshape(int n_dims, const int *dim_array)
    {
        if (!is_contiguous())
            contiguous();
        return view(n_dims, dim_array);
    }

    size_t MixMat::offset(int n_indices, const int *indices) const
    {
        const std::vector<size_t> &strides_ = d_head_.strides_;
        const std::vector<int> &dims_ = d_head_.dims_;
        assert(n_indices <= strides_.size());
        size_t off = 0;
        for (int i = 0; i < n_indices; ++i)
        {
            assert(indices[i] < dims_[i]);
            off += indices[i] * strides_[i];
        }
        return off;
    }

    MixMat& MixMat::to_host(bool copy_from_device)
    {
        // the owner is already host, just return it.
        if (d_head_.status_ == Status::HOST)
            return *this;
        // give the owner to host.
        d_head_.status_ = Status::HOST;
        data_->host(d_head_.bytes_);  // if space on host is enough, host() will not malloc.
        if (copy_from_device && data_->device())
        {
            cuda_tools::AutoExchangeDevice ex(d_head_.device_id_);
            CHECK_CUDA_RUNTIME(cudaMemcpyAsync(data_->host(), data_->device(), d_head_.bytes_, cudaMemcpyDeviceToHost, d_head_.stream_));
            CHECK_CUDA_RUNTIME(cudaStreamSynchronize(d_head_.stream_));
        }
        return *this;
    }

    MixMat &MixMat::to_device(bool copy_from_host)
    {
        // the owner is already gpu, just return it.
        if (d_head_.status_ == Status::DEVICE)
            return *this;
        d_head_.status_ = Status::DEVICE;
        data_->device(d_head_.bytes_);  // if space on device is enough, device() will not malloc.
        if (copy_from_host && data_->host())
        {
            cuda_tools::AutoExchangeDevice ex(d_head_.device_id_);
            CHECK_CUDA_RUNTIME(cudaMemcpyAsync(data_->device(), data_->host(), d_head_.bytes_, cudaMemcpyHostToDevice, d_head_.stream_));
        }
        return *this;
    }

    void MixMat::ref_data(const std::vector<int> &dims_array, void *p_host, size_t host_size, void *p_device, size_t device_size)
    {
        data_->ref_data(p_host, host_size, p_device, device_size);
        set_up_data_(data_);
        d_head_.set_dims(dims_array);
        d_head_.adjust_status();
    }
    
    template<typename dtype>
    void print_mat_recurse(
        std::ostream &out, 
        dtype *host_ptr, 
        const std::vector<int> &dims,
        const std::vector<size_t> &strides, 
        int dim_idx, int start_idx
    ){
        if (dim_idx == strides.size() - 1)  // last dim
        {
            for (int i = 0; i < dims[dim_idx]; ++i)
            {
                out << static_cast<float>(host_ptr[start_idx + i * strides[dim_idx]]) << " ";
            }
        }
        else
        {
            for (int i = 0; i < dims[dim_idx]; ++i)  // go to next dim with offset
            {
                print_mat_recurse(out, host_ptr, dims, strides, dim_idx + 1, start_idx + i * strides[dim_idx]);
                out << std::endl;
            }
        }
    }

    template<>
    void print_mat_recurse(
        std::ostream &out, 
        tinycv::float16 *host_ptr, 
        const std::vector<int> &dims,
        const std::vector<size_t> &strides, 
        int dim_idx, int start_idx
    ){
        if (dim_idx == strides.size() - 1)  // last dim
        {
            for (int i = 0; i < dims[dim_idx]; ++i)
            {
                out << __half2float(*reinterpret_cast<half *>(host_ptr + start_idx + i * strides[dim_idx])) << " ";
            }
        }
        else
        {
            for (int i = 0; i < dims[dim_idx]; ++i)  // go to next dim with offset
            {
                print_mat_recurse(out, host_ptr, dims, strides, dim_idx + 1, start_idx + i * strides[dim_idx]);
                out << std::endl;
            }
        }
    }

    // I just can't believe that even a print function is so hard.
    std::ostream& operator<<(std::ostream &out, MixMat &mat)
    {
        if (mat.d_head_.status_ == Status::INIT)
        {
            out << "[]" << std::endl;
            return out;
        }
        switch (mat.dtype())
        {
        case DataType::FLOAT32:
            print_mat_recurse(out, mat.host<float>(), mat.dims(), mat.strides(), 0, 0);    
            break;
        case DataType::INT32:
            print_mat_recurse(out, mat.host<int>(), mat.dims(), mat.strides(), 0, 0);
            break;
        case DataType::INT8:
            print_mat_recurse(out, mat.host<char>(), mat.dims(), mat.strides(), 0, 0);
            break;
        case DataType::UINT8:
            print_mat_recurse(out, mat.host<unsigned char>(), mat.dims(), mat.strides(), 0, 0);
            break;
        case DataType::FLOAT16:
            print_mat_recurse(out, mat.host<tinycv::float16>(), mat.dims(), mat.strides(), 0, 0);
            break;
        default:
            break;
        }
        return out;
    }

    bool MixMat::is_contiguous()
    {
        const std::vector<size_t> & strides_ = d_head_.strides_;
        const std::vector<int> & dims_ = d_head_.dims_;
        for (int i = 0; i < dims_.size() - 1; ++i)
        {
            if (dims_[i + 1] * strides_[i + 1] != strides_[i])
                return false;
        }
        return true;
    }

    MixMat &MixMat::contiguous(bool sync)
    {
        // already contiguous or INIT status.
        if (is_contiguous() || d_head_.status_ == INIT)
            return *this;
        Status former = d_head_.status_;  // record former status.
        if (d_head_.status_ == HOST)  // use CUDA contiguous to boost.
            to_device();
        // create a new data.    
        std::shared_ptr<BaseMemory> new_data = std::make_shared<BaseMemory>(d_head_.device_id_);
        cuda_tools::AutoExchangeDevice ex(d_head_.device_id_);
        new_data->device(d_head_.bytes_);
        contiguous_async(
            new_data->device(), data_->device(), 
            d_head_.strides_.size(), d_head_.strides_.data(), 
            d_head_.dims_.size(), d_head_.dims_.data(),
            numel(), d_head_.dtype_, d_head_.stream_, sync
        );  // contiguous stream asynchronize.
        // synchronize or not.
        data_.reset();  // former ref count - 1
        data_ = new_data;  // ref new data.
        d_head_.adjust_strides_with_dims();  // adjust contiguous strides from dims. 
        // recover to host.
        if (former == HOST)  
            to_host();
        return *this;
    }
    
    MixMat &MixMat::transpose()
    {
        const std::vector<int> &dims_ = d_head_.dims_;
        if (dims_.size() <= 1)
            return *this;
        int size = dims_.size();
        std::vector<int> dim0(size, 0);
        for (int i = 0; i < size; ++i)
            dim0[i] = i;
        dim0[size - 2] = size - 1;
        dim0[size - 1] = size - 2;

        d_head_.permute(size, dim0.data());
        return *this;
    }

    MixMat &MixMat::transpose(int n_dims, const int *dim0)
    {
        d_head_.permute(n_dims, dim0);
        return *this;
    }

    void MixMat::synchronize() const
    {
        cuda_tools::AutoExchangeDevice ex(d_head_.device_id_);
        CHECK_CUDA_RUNTIME(cudaStreamSynchronize(d_head_.stream_));
    }

    MixMat MixMat::clone() const
    {
        MixMat res(d_head_.dims_, d_head_.dtype_, nullptr, d_head_.device_id_);
        if (d_head_.status_ == HOST)
            memcpy(res.host(), this->host(), d_head_.bytes_);
        else if (d_head_.status_ == DEVICE)
        {
            cuda_tools::AutoExchangeDevice ex(d_head_.device_id_);
            CHECK_CUDA_RUNTIME(
                cudaMemcpyAsync(res.device(), this->device(), 
                d_head_.bytes_, cudaMemcpyDeviceToDevice, d_head_.stream_));
        }
        return res;
    }

    MixMat &MixMat::to(DataType dtype, bool sync)
    {
        if (dtype == d_head_.dtype_)
            return *this;
        DataType old_dtype = d_head_.dtype_;
        size_t bytes = tinycv::element_size(dtype) * numel();  // re-compute needed bytes.
        std::shared_ptr<BaseMemory> new_data = std::make_shared<BaseMemory>(d_head_.device_id_);
        cuda_tools::AutoExchangeDevice ex(d_head_.device_id_);  // exchange device
        void *new_device_data = new_data->device(bytes);  // allocate bytes
        void *old_device_data = this->device();
        convert_data_type_async(new_device_data, dtype, old_device_data, old_dtype, d_head_.numel(), d_head_.stream_, sync);
        data_.reset();  // ref cnt - 1.
        data_ = new_data;  // re-ref.

        d_head_.dtype_ = dtype;
        d_head_.bytes_ = bytes;  // in case of exchange to init status.
        d_head_.adjust_status();  // re-compute data bytes.
        return *this;
    }

    MixMat &MixMat::unsqueeze(int dim)
    {
        int n_dims = d_head_.n_dims();
        assert(n_dims >= dim && dim >= 0);
        std::vector<int> &dims = d_head_.dims_;
        auto it = dims.begin() + dim;
        dims.insert(it, 1);
        std::vector<size_t> &strides = d_head_.strides_;
        auto s_it = strides.begin() + dim;
        // TODO: How to make sure the stride.
        strides.insert(s_it, 0);
        return *this;
    }

    MixMat &MixMat::unsqueeze(int n_indices, const int *index_array)
    {
        std::vector<int> indices(index_array, index_array + n_indices);
        std::sort(indices.begin(), indices.end(), std::greater<int>());
        assert(indices[0] <= d_head_.n_dims());

        std::vector<int> &dims = d_head_.dims_;
        std::vector<size_t> &strides = d_head_.strides_;
        
        for (int i = 0; i < n_indices; ++i)
        {
            dims.insert(dims.begin() + indices[i], 1);
            // TODO: How to make sure the stride.
            strides.insert(strides.begin() + indices[i], 0);
        }
        return *this;
    }

    MixMat &MixMat::squeeze()
    {
        int n_dims = d_head_.n_dims();
        std::vector<int> &dims = d_head_.dims_;
        std::vector<size_t> &strides = d_head_.strides_;
        auto d_it = dims.begin();
        auto s_it = strides.begin();
        while (d_it != dims.end())
        {
            if (*d_it == 1)
            {
                d_it = dims.erase(d_it);
                s_it = strides.erase(s_it);
            }
            else
            {
                ++d_it;
                ++s_it;
            }
        }
        return *this;
    }

    MixMat &MixMat::squeeze(int idx)
    {
        assert(idx <= d_head_.n_dims());
        if (d_head_.dims_[idx] == 1)
        { 
            d_head_.dims_.erase(d_head_.dims_.begin() + idx);
            d_head_.strides_.erase(d_head_.strides_.begin() + idx);
        }
        return *this;
    }

    MixMat &MixMat::squeeze(int n_indices, const int *index_array)
    {
        std::vector<int> indices(index_array, index_array + n_indices);
        std::sort(indices.begin(), indices.end(), std::greater<int>());
        assert(indices[0] < d_head_.n_dims());

        std::vector<int> &dims = d_head_.dims_;
        std::vector<size_t> &strides = d_head_.strides_;
        
        for (int i = 0; i < n_indices; ++i)
        {
            if (dims[indices[i]] == 1)
            {
                dims.erase(dims.begin() + indices[i]);
                strides.erase(strides.begin() + indices[i]);
            }
        }
        return *this;
    }

    int element_size(DataType dtype)
    {
        switch (dtype)
        {
        case DataType::UINT8:
        case DataType::INT8:
            return 1;
        case DataType::INT32:
        case DataType::FLOAT32:
            return 4;
        case DataType::FLOAT16:
            return 2;
        default:
            return -1;
        }
    }

    const char *data_type_str(DataType dtype)
    {
        switch (dtype)
        {
        case DataType::FLOAT32:
            return "DataType.FLOAT32";
        case DataType::FLOAT16:
            return "DataType.FLOAT16";
        case DataType::INT32:
            return "DataType.INT32";
        case DataType::UINT8:
            return "DataType.UINT8";
        case DataType::INT8:
            return "DataType.INT8";
        default:
            return "Unknown Datatype";
        }
    }

    MixMat make_mix_mat_from_frame(const Frame &frame, void *stream)
    {
        std::shared_ptr<BaseMemory> read_buffer = nullptr;
        return make_mix_mat_from_frame(frame, read_buffer, stream);
    }

    MixMat make_mix_mat_from_frame(const Frame &frame, std::shared_ptr<BaseMemory> &read_buffer, void *stream)
    {
        if (frame.empty() || frame.mem_type_ == MemoryType::MEM_UNKNOWN) 
            return {};
        int n_channels = 3;
        int width = frame.width();
        int height = frame.height();
        MixMat ret({height, width, n_channels}, DataType::UINT8);  // h w c format as default.
        AVPixelFormat pix_fmt = (AVPixelFormat)frame.pixel_format();
        if (read_buffer == nullptr)
            read_buffer.reset(new BaseMemory());
        read_buffer->device(frame.bytes());
        cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
        MemoryType mem_type = frame.mem_type_;
        uint8_t *yuv_data = nullptr;
        if (mem_type == MemoryType::MEM_UNREGISTED || mem_type == MemoryType::MEM_PAGE_LOCKED)
        {
            CHECK_CUDA_RUNTIME(cudaMemcpyAsync(read_buffer->device(), frame.data(), frame.bytes(), cudaMemcpyHostToDevice, stream_));
            yuv_data = static_cast<uint8_t *>(read_buffer->device());
        }
        else 
            yuv_data = static_cast<uint8_t *>(frame.data());

        convert_yuv_to_rgb_hwc(
            yuv_data,
            pix_fmt,
            ret.device<uint8_t>(),
            width,
            height,
            stream_
        );
        ret.set_stream(stream_);
        return ret;
    }

    MixMat make_mix_mat_from_cvmat(const cv::Mat &cv_mat, void *stream)
    {
        std::shared_ptr<BaseMemory> read_buffer = nullptr;
        return make_mix_mat_from_cvmat(cv_mat, read_buffer, stream);
    }

    MixMat make_mix_mat_from_cvmat(const cv::Mat &cv_mat, std::shared_ptr<BaseMemory> &read_buffer, void *stream)
    {
        if (cv_mat.empty())
            return {};

        int n_channels = cv_mat.channels();
        int height = cv_mat.rows;
        int width = cv_mat.cols;
        if (read_buffer == nullptr)
            read_buffer.reset(new BaseMemory());
        size_t bytes = n_channels * height * width * sizeof(uint8_t);
        // copy source data to buffer.
        read_buffer->device(bytes);
        cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
        CHECK_CUDA_RUNTIME(cudaMemcpyAsync(read_buffer->device(), cv_mat.data, bytes, cudaMemcpyHostToDevice, stream_));
        // from BGR HWC TO RGB HWC
        MixMat ret({height, width, n_channels}, DataType::UINT8);
        ret.set_stream(stream_);
        uint8_t *ptr_bgr = static_cast<uint8_t *>(read_buffer->device());
        uint8_t *ptr_rgb = ret.device<uint8_t>();
        convert_bgr_to_rgb(
            ptr_bgr,
            ChannelType::CHANNEL_TYPE_HWC, 
            ptr_rgb,
            ChannelType::CHANNEL_TYPE_HWC,
            width,
            height,
            stream
        );
        return ret;
    }
};


