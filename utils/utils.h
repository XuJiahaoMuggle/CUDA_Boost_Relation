#ifndef UTILS_H
#define UTILS_H

#include "check.h"
#include "ilogger.hpp"
#include <cuda.h>

namespace cuda_tools
{
    /**
     * @brief Return the valid device_id.
     * if invalid 
     *      return current device id
     * else 
     *      return device_id
     * @param device_id (int): -1 as default. 
     * @return (int): the valid device_id
     */
    int get_device_id(int device_id = -1);

    /**
     * @brief Check whether the device id is valid.
     * @param device_id (int)
     * @return true on valid, false on invalid.
     */
    bool check_device_id(int device_id);

    /**
     * @brief Auto exchange device. 
     * 
     */
    class AutoExchangeDevice
    {
    public:
        AutoExchangeDevice(int device_id);
        virtual ~AutoExchangeDevice();

    private:
        int old_device_id_ = -1; 
    };

    CUcontext create_cuda_ctx(int flags, int device_id);
};

#endif  // UTILS_H