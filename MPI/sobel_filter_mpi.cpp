#include<iostream>
#include<vector>
#include<mpi.h>
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
    MPI_Init(&argc, &argv);

    int processID, processTotal;

    MPI_Comm_rank(MPI_COMM_WORLD, &processID);
    MPI_Comm_size(MPI_COMM_WORLD, &processTotal);

    Mat src, dst;

    // Load an image
    src = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    if (!src.data)
    {
        return -1;
    }

    int x, y, destX, destY;

    long pixelsPerProcess = ((src.rows)*(src.cols))/processTotal;

    if (processID == 0)
    {
        dst = src.clone();

        for(x = 0; x < src.rows; x++)
            for(y = 0; y < src.cols; y++)
                dst.at<uchar>(x,y) = 0.0;
    }

    x = 1;
    y = 1;

    destX = 1;
    destY = 1;

    for (int i = processTotal-1; i >= processID; i--)
    {
        x = destX;
        y = destY;

        destX = (((x*src.cols)+y)+pixelsPerProcess)/src.cols;
        destY = (((x*src.cols)+y)+pixelsPerProcess)%src.cols;

        if (destY == 0)
        {
            destY++;
        }
        else if (destY == src.cols-1)
        {
            destX++;
            destY = 1;
        }
    }

    int gx, gy, sum;

    unsigned char pixelArray[pixelsPerProcess] = {};
    long pixelArrayLength;

    if (processID == 0)
    {
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
        x = 1;
        y = 1;

        for (int i = processTotal-1; i > 0; i--)
        {
            MPI_Recv(&pixelArrayLength, 1, MPI_LONG_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(pixelArray, pixelArrayLength, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < pixelArrayLength; j++)
            {
                dst.at<uchar>(x,y) = pixelArray[j];
                y++;
                if (y == src.cols-1)
                {
                    x++;
                    y = 1;
                }
            }
        }
        imwrite ("./output.jpg", dst);
    }
    else
    {
        long count = 0;
        while (x <= destX)
        {
            while ((y < (src.cols-1)) && !(x == destX && y == destY))
            {
                gx = xGradient(src, x, y);
                gy = yGradient(src, x, y);
                sum = abs(gx) + abs(gy);
                sum = max(sum,0);
                sum = min(sum,255);
                pixelArray[count] = sum;
                count++;
                y++;
            }
            y = 1;
            x++;
        }
        MPI_Send(&count, 1, MPI_LONG_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(pixelArray, count, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}