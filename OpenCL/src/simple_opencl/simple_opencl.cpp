#include <algorithm>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

std::string copy_kernel = R"(
__kernel void float_copy(
    __global float * dst,
    __global float * src
    )
{
    dst[get_global_id(0)] = src[get_global_id(0)];
}
)";

int
main(int argc, char* argv[])
{
  try {
    const size_t length = 2 << 20;

    std::vector<cl_float> source(length);
    std::vector<cl_float> destination(length);

    for (int i = 0; i < length; ++i)
      source[i] = static_cast<cl_float>(i);

    // clGetPlatformIDs
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // clGetDeviceIDs
    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    // clCreateContext
    cl::Context context{ devices };

    // clCreateCommandQueue
    cl::Device& device = devices[0];
    cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };

    // clCreateBuffer
    size_t byte_size = length * sizeof(cl_float);
    cl::Buffer source_buffer{ context, CL_MEM_READ_ONLY, byte_size };
    cl::Buffer destination_buffer{ context, CL_MEM_WRITE_ONLY, byte_size };

    // clEnqueueWriteBuffer
    queue.enqueueWriteBuffer(
      source_buffer, CL_TRUE, 0, byte_size, source.data());

    // clCreateProgramWithSource
    cl::Program program{ context, copy_kernel };

    try {
      // clBuildProgram
      program.build(devices);
    } catch (cl::Error&) {
      // clGetProgramInfo
      std::string build_log;
      std::ostringstream oss;
      program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &build_log);
      std::cerr << "--------------------------------\n";
      std::cerr << "Your program failed to compile: \n";
      std::cerr << build_log << std::endl;
      return 4;
    }

    // clCreateKernel
    cl::Kernel kernel{ program, "float_copy" };

    // clSetKernelArg
    kernel.setArg(0, destination_buffer);
    kernel.setArg(1, source_buffer);

    // clCreateUserEvent
    // clEnqueueNDRangeKernel
    cl::Event event;
    queue.enqueueNDRangeKernel(kernel,
                               cl::NullRange,
                               cl::NDRange(length),
                               cl::NullRange,
                               nullptr,
                               &event);
    // clWaitForEvents
    // clGetEventProfilingInfo
    cl_ulong t1, t2;
    event.wait();
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &t1);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &t2);

    cl_ulong nano_seconds = t2 - t1;

    // clEnqueueReadBuffer
    queue.enqueueReadBuffer(
      destination_buffer, CL_TRUE, 0, byte_size, destination.data());

    if (!std::equal(source.begin(), source.end(), destination.begin())) {
      std::cerr << "Failed to copy a bunch of bytes\n";
      return 5;
    }

    std::cout << byte_size << " bytes read and written in ";
    std::cout << nano_seconds << " nano seconds\n";
    std::cout << (2.0 * byte_size) / nano_seconds << " GB/s\n";

    return 0;

  } catch (cl::Error& e) {
    std::cerr << "OpenCL error: " << e.what() << "\n";
    return 1;
  } catch (std::exception& e) {
    std::cerr << e.what() << "\n";
    return 2;
  } catch (...) {
    std::cerr << "Unexpected error. Aborting!\n";
    return 3;
  }
}
