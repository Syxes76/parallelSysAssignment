#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <JC/util.hpp>

std::string
generate_kernel(const std::string& function_invocation)
{
  std::string workspace_kernel = R"(
__kernel void workspace_query(__global uint * dst)
{
    int index = get_global_size(0) * get_global_id(1) + get_global_id(0);
    dst[index] = 
)";

  return workspace_kernel + function_invocation + "; }";
}

int
main(int argc, char* argv[])
{
  try {
    if (argc != 6) {
      std::cerr << "Usage: " << argv[0] << " <function-invocation> ";
      std::cerr << "<global-height> <global-width> ";
      std::cerr << "<local_height> <local_width>\n";
      return 1;
    }

    std::string function_invocation(argv[1]);
    int global_height = std::stoi(argv[2]);
    int global_width = std::stoi(argv[3]);
    int local_height = std::stoi(argv[4]);
    int local_width = std::stoi(argv[5]);

    int count = global_height * global_width;
    int bytes = count * sizeof(cl_uint);

    // Need one matrix only to show the result
    std::vector<cl_uint> result(count);

    // The usual stuff ...
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Context context(devices);
    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    cl::Buffer dst_buf(context, CL_MEM_WRITE_ONLY, bytes);

    std::string kernel_string = generate_kernel(function_invocation);

    cl::Program program =
      jc::build_program_from_string(kernel_string, context, devices[0]);
    cl::Kernel kernel(program, "workspace_query");
    kernel.setArg<cl::Memory>(0, dst_buf);

    cl::NDRange global(global_width, global_height);
    cl::NDRange local(local_width, local_height);

    cl_ulong t = jc::run_and_time_kernel(kernel, queue, global, local);

    // transfer destination data from the device to the host
    queue.enqueueReadBuffer(dst_buf, CL_TRUE, 0, bytes, result.data());

    // show the result
    for (int y = 0; y < global_height; ++y) {
      for (int x = 0; x < global_width; ++x) {
        std::cout << result[y * global_width + x] << " ";
      }
      std::cout << "\n";
    }

    return 0;
  } catch (cl::Error& e) {
    std::cerr << e.what() << ": " << jc::readable_error(e.err());
    return 1;
  } catch (std::exception& e) {
    std::cerr << e.what() << "\n";
    return 2;
  } catch (...) {
    std::cerr << "Unexpected error. Aborting!\n";
    return 3;
  }
}
