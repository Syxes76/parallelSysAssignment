#include<iostream>
#include<string>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv )
{
    Mat src, dst;

    // Load images
    src = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    dst = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    if( !src.data || !dst.data)
    {
        return -1;
    }


    for(int x = 0; x < src.rows; x++)
        for(int y = 0; y < src.cols; y++)
            if (dst.at<uchar>(x,y) != src.at<uchar>(x,y))
            {
                cout << "Images are not the same !" << endl;
                return 0;
            }

    cout << "Images are the same !" << endl;
    return 0;
}