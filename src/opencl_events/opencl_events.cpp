#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <JC/util.hpp>

std::string kernels = R"(
__kernel void sinus(
    __global float * dst,
    __global float * src
    )
{
    float sinus = sin(src[get_global_id(0)]);
    dst[get_global_id(0)] = sinus * sinus;
}

__kernel void cosinus(
    __global float * dst,
    __global float * src
    )
{
    float cosinus = cos(src[get_global_id(0)]);
    dst[get_global_id(0)] = cosinus * cosinus;
}

__kernel void sum(
    __global float * dst,
    __global float * src1,
    __global float * src2
    )
{
    int index = get_global_id(0);
    dst[index] = src1[index] + src2[index];
}
)";

const float pi = std::acos(-1);

int
main(int argc, char* argv[])
{
  try {
    const size_t length = 2 << 20;

    std::vector<cl_float> source(length);
    std::vector<cl_float> destination(length);

    for (int i = 0; i < length; ++i)
      source[i] = static_cast<cl_float>(i * pi / length);

    // Get platforms and devices
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    cl::Context context{ devices };

    cl::Device& device = devices[0];

    // An out of order command queue
    cl::CommandQueue queue{ context,
                            device,
                            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE };

    // Create the necessary buffers
    size_t byte_size = length * sizeof(cl_float);
    cl::Buffer source_buffer{ context, CL_MEM_READ_ONLY, byte_size };
    cl::Buffer cos_buffer{ context, CL_MEM_READ_WRITE, byte_size };
    cl::Buffer sin_buffer{ context, CL_MEM_READ_WRITE, byte_size };
    cl::Buffer destination_buffer{ context, CL_MEM_WRITE_ONLY, byte_size };

    // clCreateProgramWithSource
    cl::Program program{ context, kernels };

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
    cl::Kernel cos_kernel{ program, "cosinus" };
    cl::Kernel sin_kernel{ program, "sinus" };
    cl::Kernel sum_kernel{ program, "sum" };

    // Set the kernel arguments for the three kernels
    cos_kernel.setArg(0, cos_buffer);
    cos_kernel.setArg(1, source_buffer);
    sin_kernel.setArg(0, sin_buffer);
    sin_kernel.setArg(1, source_buffer);
    sum_kernel.setArg(0, destination_buffer);
    sum_kernel.setArg(1, cos_buffer);
    sum_kernel.setArg(2, sin_buffer);

    // Executing everything in the appropriate order
    queue.enqueueWriteBuffer(
      source_buffer, CL_TRUE, 0, byte_size, source.data());

    std::vector<cl::Event> sum_events(2);  // vector of 2 events
    std::vector<cl::Event> read_events(1); // vector of 1 event
    queue.enqueueNDRangeKernel(cos_kernel,
                               cl::NullRange,
                               cl::NDRange(length),
                               cl::NullRange,
                               nullptr,
                               &sum_events[0]);
    queue.enqueueNDRangeKernel(sin_kernel,
                               cl::NullRange,
                               cl::NDRange(length),
                               cl::NullRange,
                               nullptr,
                               &sum_events[1]);
    queue.enqueueNDRangeKernel(sum_kernel,
                               cl::NullRange,
                               cl::NDRange(length),
                               cl::NullRange,
                               &sum_events,
                               &read_events[0]);
    queue.enqueueReadBuffer(destination_buffer,
                            CL_TRUE,
                            0,
                            byte_size,
                            destination.data(),
                            &read_events);

    auto close_enough = [](float value) {
      return std::abs(value - 1.0f) < 1e-6;
    };
    if (!std::all_of(destination.begin(), destination.end(), close_enough)) {
      for (int i = 0; i < 32; ++i)
        std::cerr << destination[i] << "\n";
      std::cerr << "Incorrect results. Aborting\n";
      return 5;
    }

    return 0;

  } catch (cl::Error& e) {
    std::cerr << "OpenCL error: " << e.what() << ": ";
    std::cerr << jc::readable_error(e.err()) << "\n";
    return 1;
  } catch (std::exception& e) {
    std::cerr << e.what() << "\n";
    return 2;
  } catch (...) {
    std::cerr << "Unexpected error. Aborting!\n";
    return 3;
  }
}
