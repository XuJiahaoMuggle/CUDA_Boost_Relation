#ifndef FRAME_CACHE_HPP
#define FRAME_CACHE_HPP
#include <cstdlib>
#include <cstring>
#include <memory>
#include <cuda_runtime.h>
#include <utils/utils.h>
#include <cstdio>
#include <cassert>
#include <cuda.h>

/**
 * @brief Memory type for video proc
 */
enum MemoryType
{
    MEM_UNKNOWN = -1,     /* unknown memory data type */
    MEM_UNREGISTED = 0,   /* acllocated by new or malloc */
    MEM_PAGE_LOCKED = 1,  /* page lock memory allocated by cudaMallocHost */
    MEM_DEVICE = 2,       /* device memory allocated by cudaMalloc */
};

/**
 * @brief Simple data struct for video proc with ref count.
 * There are four total memory types @see MemoryType, for MEM_UNREGISTED, MEM_PAGE_LOCKED, MEM_DEVICE
 * the memory release functions are configured as free, cudaFreeHost, cudaFree respectively.
 * For MEM_UNKNOWN user can configure the release function manually.
 * @warning Do not try to bind one memory to more than one Frame, this will free the momory more than once. 
 */
struct Frame
{
    /**
     * @brief Families of Frame construction.
     * To bind the frame data such as decoded frame data to Frame type object:
     * Frame(): Do nothing;
     * Frame(void *data, size_t bytes, dx dt): manually specified the free function;
     * Frame(void *data, size_t bytes, MemoryType mem_type): given the mem_type,this will lead to 
     * auto memory free when reference count is 0;
     * Frame(const Frame &frame): add the ref count only.
     */
    Frame() = default;

    template <typename dx>
    Frame(void *data, size_t bytes, dx dt, int width, int height, int pix_fmt, int64_t pts, int device_id = -1)  // for unknown memory data type
    : bytes_(bytes), height_(height), width_(width), pix_fmt_(pix_fmt), pts_(pts), device_id_(device_id)
    {
        data_ = std::shared_ptr<void> (data, dt); 
    }

    Frame(void *data, size_t bytes, MemoryType mem_type, int width, int height, int pix_fmt, int64_t pts, int device_id = -1)
    : bytes_(bytes), mem_type_(mem_type), height_(height), width_(width), pix_fmt_(pix_fmt), pts_(pts), device_id_(device_id)
    {
        assert(mem_type != MemoryType::MEM_UNKNOWN);
        if (mem_type_ == MemoryType::MEM_UNREGISTED)
            data_ = std::shared_ptr<void> (data, [](void *p_data) -> void {free(p_data);});
        else if (mem_type_ == MemoryType::MEM_PAGE_LOCKED)
            data_ = std::shared_ptr<void> (data, [this](void *p_data) -> void {
                cuda_tools::AutoExchangeDevice ex(this->device_id_);
                CHECK_CUDA_RUNTIME(cudaFreeHost(p_data));
            });
        else
            data_ = std::shared_ptr<void> (data, [this](void *p_data) -> void {
                cuda_tools::AutoExchangeDevice ex(this->device_id_);
                CHECK_CUDA_RUNTIME(cudaFree(p_data));
            });
    }

    Frame(const Frame &frame)
    {
        data_ = frame.data_;  // add ref count.
        bytes_ = frame.bytes_;
        height_ = frame.height_;
        width_ = frame.width_;
        pix_fmt_ = frame.pix_fmt_;
        mem_type_ = frame.mem_type_;
        pts_ = frame.pts_;
        device_id_ = frame.device_id_;
    }

    Frame &operator=(const Frame &frame)
    {
        if (this == &frame)
            return *this;
        data_.reset();  // minus ref count
        data_ = frame.data_;  // add ref count.
        bytes_ = frame.bytes_;
        height_ = frame.height_;
        width_ = frame.width_;
        pix_fmt_ = frame.pix_fmt_;
        mem_type_ = frame.mem_type_;
        pts_ = frame.pts_;
        device_id_ = frame.device_id_;
        return *this;
    }

    ~Frame()
    {
        data_.reset();
        bytes_ = 0;
    }

    bool empty() const { return bytes_ == 0; }

    void *data() const {return data_.get();}
    size_t bytes() const {return bytes_;}
    int height() const {return height_;}
    int width() const {return width_;}
    int pixel_format() const {return pix_fmt_;}
    MemoryType memory_type() const {return mem_type_;}
    int64_t pts() const {return pts_;}
    int height_ = 0;
    int width_ = 0;
    int pix_fmt_ = -1;
    int device_id_ = -1;
    size_t bytes_ = 0;
    int64_t pts_ = 0;
    MemoryType mem_type_ = MemoryType::MEM_UNKNOWN;
    std::shared_ptr<void> data_ = nullptr;
};

/**
 * @brief Frame memory controller.
 * @tparam dtype: Frame type as default.
 */
template <typename dtype = Frame>
class FrameAllocator
{
public:
    dtype *allocate(int size)
    {
        return (dtype *)malloc(sizeof(dtype) * size);
    }

    void deallocate(dtype *data)
    {
        free(data);
    }

    void construct(dtype *p, const dtype &rhs)
    {
        new (p) dtype(rhs);
    }
    
    void destruct(dtype *p)
    {
        p->~dtype();
    }
};

/**
 * @brief Frame Cache: a ringbuffer like structure to save frame.
 * you can use it like std::que: push, pop, size, empty, front are all supported.
 * the only difference is that the maximum size is limited, user can use full()
 * to check. 
 */
class FrameCache
{
public:
    FrameCache() = default;

    FrameCache(int size)
    : size_(size)
    {
        data_ = frame_allocator_.allocate(size + 1);
    }

    FrameCache(const FrameCache &rhs)
    {
        front_ = 0;
        rear_ = 0;
        size_ = rhs.size_;
        data_ = frame_allocator_.allocate(size_ + 1);
        for (int i = rhs.front_; i != rhs.rear_; i = (i + 1) % (size_ + 1))
            push(rhs.data_[i]);
    }

    FrameCache &operator=(const FrameCache &rhs)
    {
        if (this == &rhs)
            return *this;
        clear();
        front_ = 0;
        rear_ = 0;
        size_ = rhs.size_;
        data_ = frame_allocator_.allocate(size_ + 1);
        for (int i = rhs.front_; i != rhs.rear_; i = (i + 1) % (size_ + 1))
            push(rhs.data_[i]);
        return *this;
    }
    
    ~FrameCache()
    {
        clear();
    }

    bool empty() {return rear_ == front_;}

    bool full(){return (rear_ + 1) % (size_ + 1) == front_;}

    int size() {return rear_ >= front_ ? rear_ - front_ : rear_ + size_ + 1 - front_;}

    void clear()
    {
        while (!empty())
            pop();
        frame_allocator_.deallocate(data_);
        front_ = 0;
        rear_ = 0;
        size_ = 0;
        data_ = nullptr;
    }

    Frame &front() {return data_[front_];}

    void pop()
    {
        frame_allocator_.destruct(&data_[front_]);
        front_ = (front_ + 1) % (size_ + 1);                              
    }

    void push(const Frame &val)
    {
        if (full())
            return;
        frame_allocator_.construct(&data_[rear_], val);
        rear_ = (rear_ + 1) % (size_ + 1);
    }

private:
    int front_ = 0;
    int rear_ = 0;
    int size_ = 0;

    Frame *data_ = nullptr;
    FrameAllocator<Frame> frame_allocator_;
};

#endif