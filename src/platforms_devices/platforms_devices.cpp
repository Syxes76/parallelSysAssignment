#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <CL/cl.hpp>

struct DeviceQuery
{

  void operator()(const cl::Device& device)
  {
    // clGetDeviceInfo
    std::string device_name, opencl_c_version, built_in_kernels;
    size_t max_work_group_size;
    cl_ulong local_mem_size, max_mem_alloc_size;
    cl_uint max_clock_frequency;
    device.getInfo(CL_DEVICE_NAME, &device_name);
    device.getInfo(CL_DEVICE_OPENCL_C_VERSION, &opencl_c_version);
    device.getInfo(CL_DEVICE_BUILT_IN_KERNELS, &built_in_kernels);
    device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_work_group_size);
    device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &local_mem_size);
    device.getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &max_clock_frequency);
    device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &max_mem_alloc_size);

    std::cout << "\n";
    std::cout << " Querying device " << device_name << "\n";
    std::cout << "  OpenCL version:      " << opencl_c_version << "\n";
    std::cout << "  built in kernels:    " << built_in_kernels << "\n";
    std::cout << "  max work group size: " << max_work_group_size << "\n";
    std::cout << "  local mem size:      " << local_mem_size << "\n";
    std::cout << "  max clock frequency: " << max_clock_frequency << "\n";
    std::cout << "  max mem alloc size:  " << max_mem_alloc_size << "\n";
    std::cout << "\n";
  }
};

struct PlatformQuery
{

  void operator()(const cl::Platform& platform)
  {

    // clGetPlatformInfo
    std::string platform_name, profile, version, vendor;
    platform.getInfo(CL_PLATFORM_NAME, &platform_name);
    platform.getInfo(CL_PLATFORM_PROFILE, &profile);
    platform.getInfo(CL_PLATFORM_VERSION, &version);
    platform.getInfo(CL_PLATFORM_VENDOR, &vendor);
    std::cout << "Querying platform " << platform_name << "\n";
    std::cout << " profile: " << profile << "\n";
    std::cout << " version: " << version << "\n";
    std::cout << " vendor:  " << vendor << "\n";

    // clGetDeviceIDs
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    std::for_each(devices.cbegin(), devices.cend(), DeviceQuery{});

    std::cout << "\n";
  }
};

int
main(int argc, char* argv[])
{
  // clGetPlatformIDs
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  std::for_each(platforms.cbegin(), platforms.cend(), PlatformQuery{});

  return 0;
}
