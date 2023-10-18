#ifndef BASE_MEMORY_HPP
#define BASE_MEMORY_HPP
#include <string.h>
#include <utils/utils.h>

namespace tinycv
{    
    class BaseMemory
    {
    public:
        /**
         * @brief Binding a new Base Memory to specified device id.
         * @param device_id (int): -1 as default, means the current device.
         */
        BaseMemory(int device_id = -1);

        /**
         * @brief Let a new Base Memory ref the given data, the data should be free 
         * by others. This means the new Base Memory object is not the owner of p_host
         * and p_device.
         * @param p_host (void *): the data on host. 
         * @param host_size (size_t): the data size of p_host.
         * @param p_device (void *): the data on device.
         * @param device_size (size_t): the data size of p_device.
         */
        BaseMemory(void *p_host, size_t host_size, void *p_device, size_t device_size);

        // @see release_all()
        virtual ~BaseMemory() { release_all(); }

        /**
         * @brief Ref the data from outside.
         * @param p_host (void *)
         * @param host_size (size_t)
         * @param p_device (void *)
         * @param device_size (size_t)
         */
        void ref_data(void *p_host, size_t host_size, void *p_device, size_t device_size);

        /**
         * @brief Return the host data
         * if the arg 'size' is given and size > host_data_size_, the data will be reallocated.
         * @param size (size_t): the data bytes to allocate.
         * @return void* 
         */
        void *host() const { return host_data_; }
        void *host(size_t size);

        /**
         * @brief Return the device data
         * if the arg 'size' is given and size > device_data_size_, the data will be reallocated.
         * @param size (size_t): the data bytes to allocate.
         * @return void* 
         */
        void *device() const { return device_data_; }
        void *device(size_t size);

        /**
         * @brief Reutrn the number of bytes on host.
         * @return size_t 
         */
        size_t host_data_size() const { return host_data_size_; }

        /**
         * @brief Reutrn the number of bytes on device.
         * @return size_t 
         */
        size_t device_data_size() const { return device_data_size_; }

        /**
         * @brief Release the data on host if own the data.
         */
        void release_host();

        /**
         * @brief Release the data on device if own the data.
         */
        void release_device();

        /**
         * @brief Release both and device data.
         */
        void release_all();

        /**
         * @brief Return the device id.
         * @return (int): > -1 on valid.
         */
        int device_id() const { return device_id_; }

        /**
         * @brief The flag of host referring outside data.
         * @return true on referring false on not.
         */
        bool is_host_owner() const { return host_owner_; }

        /**
         * @brief The flag of device referring outside data.
         * @return true on referring false on not.
         */
        bool is_device_owner() const { return device_owner_; }

    private:
        int device_id_ = -1;  // device id.

        void *host_data_ = nullptr;  // host data
        size_t host_data_size_ = 0;  // hosta data size (bytes)
        bool host_owner_ = false;   

        void *device_data_ = nullptr;  // device data
        size_t device_data_size_ = 0;  // device data size (bytes)
        bool device_owner_ = false;  
    };
};

#endif