#ifndef MIX_TENSOR_H
#define MIX_TENSOR_H
#include <memory>
#include <vector>
#include <cassert>
#include <iostream>
#include <video_proc/frame_cache.hpp>
#include <opencv2/opencv.hpp>
#include "base_memory.hpp"

struct CUstream_st;
using cudaStream_t = CUstream_st *;

namespace tinycv
{   
    struct float16
    {
        unsigned short _;
    };
    
    enum DataType
    {
        UINT8,
        INT8,
        INT32,
        FLOAT32,
        FLOAT16
    };

    const char *data_type_str(DataType dtype);

    int element_size(DataType dtype);

    enum Status
    {
        INIT,  // need to reallocate/allocate
        HOST,  // host is using data
        DEVICE  // device is using data
    };

    struct DataHead
    {
        DataHead() = default;
        
        DataHead(const DataHead &d_head)
        : status_(d_head.status_), 
        dtype_(d_head.dtype_),
        device_id_(d_head.device_id_),
        dims_(d_head.dims_),
        strides_(d_head.strides_),
        bytes_(d_head.bytes_),
        stream_(d_head.stream_),
        stream_owner_(false)
        {}

        DataHead &operator=(const DataHead &d_head)
        {
            if (this == &d_head)
                return *this;
            reset();
            status_ = d_head.status_;
            dtype_ = d_head.dtype_;
            device_id_ = d_head.device_id_;
            dims_ = d_head.dims_;
            strides_ = d_head.strides_;
            bytes_ = d_head.bytes_;
            stream_ = d_head.stream_;
            stream_owner_ = false;
            return *this;
        }

        ~DataHead() {reset();}

        void dim_string()
        {
            dims_str_[0] = '\0';
            char *buf = dims_str_;
            size_t buf_cnt = sizeof(dims_str_);
            for (int i = 0; i < dims_.size(); ++i)
            {
                int size = 0;
                if (i != dims_.size() - 1)
                    size = snprintf(buf, buf_cnt, "%d x ", dims_[i]);
                else
                    size = snprintf(buf, buf_cnt, "%d", dims_[i]);
                buf += size;
                buf_cnt -= size;
            } 
        }

        void reset()
        {
            dims_.clear();
            strides_.clear();
            bytes_ = 0;
            status_ = Status::INIT;
            if (stream_owner_ && stream_)
            {
                cuda_tools::AutoExchangeDevice ex(device_id_);
                CHECK_CUDA_RUNTIME(cudaStreamDestroy(stream_));
            }
            stream_owner_ = false;
            stream_ = nullptr;  
        }

        void adjust_strides_with_dims()
        {
            strides_.resize(dims_.size());
            for (int i = strides_.size() - 1; i >= 0; --i)
            {
                if (i != strides_.size() - 1)
                    strides_[i] = strides_[i + 1] * dims_[i + 1];
                else
                    strides_[i] = 1;
            }
        }

        void adjust_status()
        {
            size_t needed_bytes = numel() * element_size();
            if (needed_bytes > bytes_)
                status_ = Status::INIT;
            bytes_ = needed_bytes;
        }

        size_t numel() const
        {
            size_t n = 1;
            for (int i = 0; i < dims_.size(); ++i)
                n *= dims_[i];
            return n;
        }

        int element_size() const
        {
            switch (dtype_)
            {
            case FLOAT32:
            case INT32:
                return 4;
            case FLOAT16:
                return 2;
            case UINT8:
            case INT8:
                return 1;
            default:
                INFO_FATAL("Unknown date type.");
                return -1;
            }
        }

        void set_device_id(int device_id)
        {
            device_id_ = device_id;
        }

        void set_dims(const std::vector<int> &dims)
        {
            dims_ = dims;
            dim_string();
            adjust_strides_with_dims();
        }

        void set_dtype(DataType dtype)
        {
            dtype_ = dtype;
        }

        void set_stream(cudaStream_t stream, bool stream_owner = false)
        {
            stream_ = stream;
            stream_owner_ = stream_owner;
        }

        void permute(int n_dims, const int *dim0)
        {
            assert(n_dims == dims_.size());
            if (dims_.size() == 1)
                return;
            std::vector<size_t> strides_cpy = strides_;
            for (int i = 0; i < n_dims; ++i)
                strides_[i] = strides_cpy[dim0[i]];

            std::vector<int> dims_cpy = dims_;
            for (int i = 0; i < n_dims; ++i)
                dims_[i] = dims_cpy[dim0[i]];
            dim_string();
        }
        
        int n_dims() const {return dims_.size();}

        Status status_ = Status::INIT;
        DataType dtype_ = DataType::FLOAT32;

        int device_id_ = -1;
        std::vector<int> dims_;
        char dims_str_[100] = {'\0'};
        std::vector<size_t> strides_;
        size_t bytes_ = 0;

        // stream will not be assigned to true only if set_stream() function is called.
        cudaStream_t stream_ = nullptr;
        bool stream_owner_ = false;
    };

    class MixMat
    {
    public:
        friend std::ostream& operator<<(std::ostream &out, MixMat &mat);
        MixMat() = default;
        /**
         * @brief To construct a new MixMat object, the dims of MixMat must be given.
         * The dims' format can be one of: [n, c, h, w] format like style, 
         * std::vector<int> style, C array style.
         * The data type of MixMat can be set as one of DataType::FLOAT32(as default)
         * DataType::FLOAT16, DataType::UINT8, DataType::INT32, @see DataType.
         * The data can also be set when construting by passing the data arg, it's a 
         * std::shared_ptr<BaseMemory> type @see BaseMemory.
         */
        MixMat(
            int n, int c, int h, int w, 
            DataType dtype = DataType::FLOAT32, 
            std::shared_ptr<BaseMemory> data = nullptr, 
            int device_id = -1
        );
        MixMat(
            const std::vector<int> &dims_array, 
            DataType dtype = DataType::FLOAT32, 
            std::shared_ptr<BaseMemory> data = nullptr, 
            int device_id = -1
        );
        MixMat(
            const int n_dims, const int *dims_array, 
            DataType dtype = DataType::FLOAT32, 
            std::shared_ptr<BaseMemory> data = nullptr, 
            int device_id = -1
        );
        
        /**
         * @brief This is not a deep copy constuct move, only the data head will be copyed.
         * so this is an O(1) move. Share the same data with rhs.
         * @param rhs (const MixMat &)
         */
        MixMat(const MixMat &rhs)
        : d_head_(rhs.d_head_), data_(rhs.data_)  // add ref count
        {}

        /**
         * @brief This is not a deep copy assigned move, only the data head will be copyed.
         * so this is an O(1) move. Share the same data with rhs.
         * @param rhs (const MixMat &)
         */
        MixMat &operator=(const MixMat &rhs)
        {
            if (this == &rhs)
                return *this;
            release();  // ref count - 1
            d_head_ = rhs.d_head_;
            data_ = rhs.data_;  // ref count + 1
            return *this;
        }
        
        /**
         * @brief Release the datahead resources, and data ref count -1; 
         * @see release()
         */
        ~MixMat() {release();}

        /**
         * @brief Release the datahead resources, and data ref count -1; 
         * @return (MixMat&): return the object itself. 
         */
        MixMat &release();

        /**
         * @brief View opeartion, this function will not change memory, but 
         * requires strides and dims are aligned. is_contiguous() can help check.
         * and the contiguous() function can readjust the momory.
         * @see is_contiguous(), contiguous()
         * The new dims' element number should be the same as the older, and only
         * one -1 index is allowed.
         * The new dims' format can be one of: C array, std::vector<int>, args...   
         */
        MixMat &view(int n_dims, const int *dim_array);
        MixMat &view(const std::vector<int> &dim_array)
        {
            return view(dim_array.size(), dim_array.data());
        }
        template <typename... args>
        MixMat &view(int dummy, args... dummy_array)
        {
            const int dims_array[] = {dummy, dummy_array...};
            return view(sizeof...(dummy_array) + 1, dims_array);    
        }

        /**
         * @brief Almost the same as view opearation, but check is_contiguous() inside.
         * which means if not contiguous, the contiguous funtion will be called. 
         */
        MixMat &reshape(int n_dims, const int *dim_array);
        MixMat &reshape(const std::vector<int> &dim_array)
        {
            return reshape(dim_array.size(), dim_array.data());
        }
        template <typename... args>
        MixMat &reshape(int dummy, args... dummy_array)
        {
            const int dims_array[] = {dummy, dummy_array...};
            return reshape(sizeof...(dummy_array) + 1, dims_array);    
        }

        /**
         * @brief Check whether the dims and strides are aligned.
         * aligned only when: strides[i] = strides[i + 1] * dims[i + 1] 
         * @return MixMat& 
         */
        bool is_contiguous();
        
        /**
         * @brief Adjust the memory so that the dims and strides can be aligned.
         * @note The memory is aligned by cuda kernel function. If the data is on host
         * the data will be copy to device first.
         * @param sync (bool): tu  
         * @return MixMat& 
         */
        MixMat &contiguous(bool sync = false);

        /**
         * @brief Return the offset based on the given indices.
         * The indices format can be one of: C array, std::vector<int>, args... 
         * @param n_indices 
         * @param indices 
         * @return size_t 
         */
        size_t offset(int n_indices, const int *indices) const;
        template <typename... args>
        size_t offset(int dummy, args... dummy_array) const
        {
            const int indices_array[] = {dummy, dummy_array...};
            return offset(sizeof...(dummy_array) + 1, indices_array);
        }
        size_t offset(const std::vector<int> &indices) const
        {
            return offset(indices.size(), indices.data());
        }

        /**
         * @brief Convert datatype.
         * 
         */
        MixMat &to(DataType dtype, bool sync = false);

        // TODO: Unstable: how to make sure the stride for insert index?
        MixMat &unsqueeze()
        {
            return unsqueeze(0);
        }
        MixMat &unsqueeze(int idx);
        MixMat &unsqueeze(int n_indices, const int *indices);
        template <typename... args>
        MixMat &unsqueeze(int dummy, args... dummy_array)
        {
            const int indices[] = {dummy, dummy_array...};
            return unsqueeze(sizeof...(dummy_array) + 1, indices);
        }

        // squeeze opeartion.
        MixMat &squeeze();
        MixMat &squeeze(int idx);
        MixMat &squeeze(int n_indices, const int *indices);
        template <typename... args>
        MixMat &squeeze(int dummy, args... dummy_array)
        {
            const int indices[] = {dummy, dummy_array...};
            return squeeze(sizeof...(dummy_array) + 1, indices);
        }
        
        // copy data from device to host.
        MixMat &to_host(bool copy_from_device = true);
        // copy data from host to device.
        MixMat &to_device(bool copy_from_host = true);

        // return the host data, call to_host at first.
        void *host() const {const_cast<MixMat *>(this)->to_host(); return data_->host();}
        // return the device data, call to_device at first.
        void *device() const {const_cast<MixMat *>(this)->to_device(); return data_->device();}

        // return the specified data type host/device pointer.
        template <typename dtype>
        dtype *host() {return (dtype *)host();}
        template <typename dtype, typename... args>
        dtype *host(int i, args... dummy_array) {return host<dtype>() + offset(i, dummy_array...);}
        
        template <typename dtype>
        dtype *storage() {return host<dtype>();}

        template <typename dtype>
        dtype *device() {return (dtype *)device();}
        template <typename dtype, typename... args>
        dtype *device(int i, args... dummy_array) {return device<dtype>() + offset(i, dummy_array...);}

        // return the specified indices' data.
        template <typename dtype, typename... args>
        dtype &at(int i, args... dummy_array) {return *(host<dtype>() + offset(i, dummy_array...));}

        // return the device id.
        int device_id() const {return d_head_.device_id_;}

        // return the number of element.
        size_t numel() const {return d_head_.numel();}

        // return the dims.
        const std::vector<int> &dims() const {return d_head_.dims_;}
        
        // return the strides.
        const std::vector<size_t> &strides() const {return d_head_.strides_;}
        
        // return the description of dims.
        const char *dims_str() {d_head_.dim_string(); return d_head_.dims_str_;}

        // return the number of bytes per element.
        int element_size() const {return d_head_.element_size();}
        
        // return the total number of bytes.
        size_t bytes() const {return d_head_.bytes_;}

        // ref the data from outside.
        void ref_data(const std::vector<int> &dims_array, void *p_host, size_t host_size, void *p_device, size_t device_size);

        /**
         * @brief Transpose opeartion, the strides and dims will be modified, this will cause
         * strides and dims misaligend.
         * The input format can be one of: C array, std::vector<int>, args... or exchange last two dims
         * as default. 
         * @return MixMat& 
         */
        MixMat &transpose();
        MixMat &transpose(int n_dims, const int *dim0);
        template <typename... args>
        MixMat &transpose(int dummy, args... dummy_dim)
        {
            const int dim0[] = {dummy, dummy_dim...};
            return transpose(sizeof...(dummy_dim) + 1, dim0);
        }
        MixMat &transpose(const std::vector<int> &dim0)
        {
            return transpose(dim0.size(), dim0.data());
        }

        // return stream_owner_
        bool is_stream_owner() const {return d_head_.stream_owner_;}

        // return the cuda strea
        cudaStream_t get_stream() const {return d_head_.stream_;}

        // set the cuda stream
        void set_stream(cudaStream_t stream, bool stream_owner = false)
        {
            d_head_.set_stream(stream, stream_owner);
        }

        bool empty() const {return data_ == nullptr;}
        
        int n_dims() const {return d_head_.dims_.size();}
        
        DataType dtype() const {return d_head_.dtype_;}
        
        // wait for cuda stream.
        void synchronize() const;

        // clone a new MixMat object, this function will copy data.
        MixMat clone() const;

        // return the data head
        DataHead &data_head() {return d_head_;}

        // return the ref data.
        std::shared_ptr<BaseMemory> data() const {return data_;}
    
    private:
        /**
         * @brief initialize data_.
         * if data is nullptr, data_ will be allocated and status_ will be set to INIT.
         * if data is not nullptr, data_ will be configured as data, the status_ is decided
         * by data.
         * @param data (std::shared_ptr<BaseMemory>)
         */
        void set_up_data_(std::shared_ptr<BaseMemory> data);
        
        void assert_view_(int n_dims, const int *dim_array);
        
        // TODO
        void assert_slice_(int n_dim, const int *dim_array);
        
    private:
        DataHead d_head_;
        std::shared_ptr<BaseMemory> data_ = nullptr;
    };

    std::ostream& operator<<(std::ostream &out, MixMat &mat);

    MixMat make_mix_mat_from_frame(const Frame &frame, void *stream = nullptr);
    MixMat make_mix_mat_from_frame(const Frame &frame, std::shared_ptr<BaseMemory> &read_buffer, void *stream = nullptr);

    MixMat make_mix_mat_from_cvmat(const cv::Mat &cv_mat, void *stream = nullptr);
    MixMat make_mix_mat_from_cvmat(const cv::Mat &cv_mat, std::shared_ptr<BaseMemory> &read_buffer, void *stream = nullptr);

};

#endif