#include<iostream>
#include<string>
#include<cmath>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


// Computes the x component of the gradient vector
// at a given point in a image.
// returns gradient in the x direction
int xGradient(Mat image, int x, int y)
{
    return image.at<uchar>(x-1, y-1) +
            2*image.at<uchar>(x, y-1) +
            image.at<uchar>(x+1, y-1) -
            image.at<uchar>(x-1, y+1) -
            2*image.at<uchar>(x, y+1) -
            image.at<uchar>(x+1, y+1);
}

// Computes the y component of the gradient vector
// at a given point in a image
// returns gradient in the y direction

int yGradient(Mat image, int x, int y)
{
    return image.at<uchar>(x-1, y-1) +
            2*image.at<uchar>(x-1, y) +
            image.at<uchar>(x-1, y+1) -
            image.at<uchar>(x+1, y-1) -
            2*image.at<uchar>(x+1, y) -
            image.at<uchar>(x+1, y+1);
}

int main(int argc, char** argv )
{
    Mat src, dst;
    int gx, gy, sum;

    // Load an image
    src = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    dst = src.clone();
    if( !src.data )
    {
        return -1;
    }


    for(int x = 0; x < src.rows; x++)
        for(int y = 0; y < src.cols; y++)
            dst.at<uchar>(x,y) = 0.0;

    int x = 1;
    int y = 1;

    while (x < src.rows-1)
    {
        while (y < src.cols-1)
        {
            gx = xGradient(src, x, y);
            gy = yGradient(src, x, y);
            sum = abs(gx) + abs(gy);
            sum = max(sum,0);
            sum = min(sum,255);
            dst.at<uchar>(x,y) = sum;
            y++;
        }
        y = 1;
        x++;
    }

    imwrite ("./output_seq.jpg", dst);

    return 0;
}