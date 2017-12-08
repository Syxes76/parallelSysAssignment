#include <algorithm>
#include <cassert>
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include <JC/util.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

std::vector<cl_int> opencl_sobel_filter(const std::vector<cl_int>& src, const int rows, const int cols)
{
    assert(src.size() == rows*cols);
    std::vector<cl_int> dst(rows*cols);

    cl::Device device = jc::get_device(CL_DEVICE_TYPE_CPU);
    cl::Context context{ device };
    cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };

    // TODO
    // (1) In a file called "kernels.ocl" in the appropriate directory,
    //     write the code for a kernel called "erosion".
    cl::Program program =
    jc::build_program_from_file("./kernels.ocl", context, device);
    cl::Kernel kernel{ program, "sobel_filter" };

    // (1) Create buffers
    size_t byte_size = src.size() * sizeof(cl_int);
    cl::Buffer dst_buffer{ context, CL_MEM_WRITE_ONLY, byte_size };
    cl::Buffer src_buffer{ context, CL_MEM_READ_ONLY, byte_size };

    // (2) Set arguments
    kernel.setArg(0, dst_buffer);
    kernel.setArg(1, src_buffer);
    kernel.setArg(2, rows);
    kernel.setArg(3, cols);

    size_t local_byte_size = (((256) * 3) + 6) * sizeof(cl_int);
    kernel.setArg(4, cl::Local(local_byte_size));

    // (3) Write data to device
    queue.enqueueWriteBuffer(src_buffer, CL_TRUE, 0, byte_size, src.data());

    // (4) Run and time kernel
    cl_ulong nanoseconds =
                jc::run_and_time_kernel(kernel,
                queue,
                cl::NDRange(jc::best_fit(src.size(), 256)),
                cl::NDRange(256));

    // (5) Read data from device
    queue.enqueueReadBuffer(dst_buffer, CL_TRUE, 0, byte_size, dst.data());
    std::cout << "Time elapsed: " << nanoseconds << " ns\n";

    return dst;
}

int main(int argc, char* argv[])
{
    try {
        Mat src, dst;

        // Load an image
        src = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE);
        dst = src.clone();
        if( !src.data )
        {
            return -1;
        }

        std::vector<cl_int> src_vector(src.rows*src.cols);
        for(int x = 0; x < src.rows; x++)
            for(int y = 0; y < src.cols; y++)
                src_vector[(x * src.cols) + y] = src.at<uchar>(x,y);

        std::vector<cl_int> dst_vector = opencl_sobel_filter(src_vector, src.rows, src.cols);
        for(int x = 0; x < src.rows; x++)
            for(int y = 0; y < src.cols; y++)
                dst.at<uchar>(x,y) = dst_vector[(x * src.cols) + y];
        
        imwrite ("./output.jpg", dst);

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
