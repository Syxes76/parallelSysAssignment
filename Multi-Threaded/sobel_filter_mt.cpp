#include<iostream>
#include<string>
#include<cmath>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

struct thread_param {
    Mat srcIm;
    Mat dstIm;
    int startX;
    int startY;
    int destX;
    int destY;
};

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

void *processPixels(void *args) 
{
    struct thread_param *param = (struct thread_param *)args;
    int gx, gy, sum;

    int threadStartX = (param->startX);
    int threadStartY = (param->startY);
    int threadDestX = (param->destX);
    int threadDestY = (param->destY);

    int x = threadStartX;
    int y = threadStartY;

    while (x <= threadDestX)
    {
        while ((y < ((param->srcIm).cols-1)) && !(x == threadDestX && y == threadDestY))
        {
            gx = xGradient((param->srcIm), x, y);
            gy = yGradient((param->srcIm), x, y);
            sum = abs(gx) + abs(gy);
            sum = max(sum,0);
            sum = min(sum,255);
            (param->dstIm).at<uchar>(x,y) = sum;
            y++;
        }
        y = 1;
        x++;
    }
    pthread_exit(NULL);
}

int main(int argc, char** argv )
{
    Mat src, dst;

    int numThreads = stoi(argv[1]);
    pthread_t threads[numThreads];

    long pixelsPerThread = ((src.rows)*(src.cols))/numThreads;

    // Load an image
    src = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE);
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

    for (int i = numThreads-1; i > 0; i--)
    {
        int tempX = (((x*src.cols)+y)+pixelsPerThread)/src.cols;
        int tempY = (((x*src.cols)+y)+pixelsPerThread)%src.cols;

        if (tempY == 0)
        {
            tempY++;
        }
        else if (tempY == src.cols-1)
        {
            tempX++;
            tempY = 1;
        }

        thread_param* newParam = new thread_param();
        newParam->srcIm = src;
        newParam->dstIm = dst;
        newParam->startX = x;
        newParam->startY = y;
        newParam->destX = tempX;
        newParam->destY = tempY;

        int error = pthread_create(&threads[i], NULL, processPixels, (void *)newParam);
        if (error) {
            cout << "Error:unable to create new thread," << error << endl;
            exit(-1);
        }

        x = tempX;
        y = tempY;
    }

    int gx, gy, sum;

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

    for (int i = numThreads-1; i > 0; i--)
    {
        pthread_join(threads[i], NULL);
    }

    imwrite ("./output.jpg", dst);

    return 0;
}