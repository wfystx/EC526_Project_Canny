/*
		//                  EC526 Project: Canny Edge Dectection                                           //
		//                                                                                                 //
		//            Authors: Fuyao Wang, Mingdao Che, Pat rick Ntwari                                    //
		//                                                                                                 //
		// Introction: The codes is for EC526-Spring2020. And two prosopals for this project:              // 
		//                                                                                                 //
		// 1.It mainly contains four main steps to figure out the final results which is: GaussianBlur,    // 
		// Image Gradient Calculation,  Non-max-Suppression and threshold filter.                          //
		//                                                                                                 //
		// 2. Using the openMP as the parallelization method to comparing different running time between   //
		// different kernals.                                                                              //


		
		Date: 4/27/2020 
*/

/*
	 The Head file of the project.
*/
#include "pch.h" // pre-compiled header for visual studio.
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include "kernels.h" // Head file for differnt kernals of gradient calculation.
#include "opencv2/core.hpp"  //OpenCV lib. 
#include "opencv2/imgproc.hpp" //OpenCV lib.
#include "opencv2/highgui.hpp" //OpenCV lib. 
#include "omp.h" // OpenMP head file.
#include "mpi.h" // MPI head file

/*
	Global constant Delaration.
*/
#define M_PI 3.14159265358979323846
#define mode 4// 1 for sobel, 2 for prewitt, 3 for Robert, 4 for 5X Sobel, else scharr

using namespace cv;
using namespace std;

/*
  Function Delaration.
*/
bool checkExistence(string filename); // Check whether image exists.
void CannyProcess(); // Canny dector.
void GaussianBlur_sigma();
void Gradient_Images();
void NonMaxSuppress();
void DoubleThreshold_Hysteresis(int, int);
void convuloution(float** x, float** y, int sobelRad, int sobelWidth); // colvolution for 3X3.
void convuloution_2(float** x, float** y, int sobelRad, int sobelWidth);  // colvolution for 2X2.
void convuloution_5(float** x, float** y, int sobelRad, int sobelWidth); // colvolution for 5X5.
float** matrixgenerator(int col, int row);


/*
  Function Variables Delaration.
*/
Mat OImage; // Original Image Mat
Mat BImage; // blured Image Mat
Mat EMImage; // edgeMag Image Mat
Mat EAImage; // edgeAng Image Mat
Mat TEImage; // thinEdge Image Mat
Mat DTImage; // Threshold Edge Image Mat
int Convert_Radius, KernelWidth = 0;
float avgGradient = 0.0, var = 0.0;

int main(int argc, char** argv)
{
	Mat outs;
	//int coreNum = omp_get_num_procs(); // Show the process of parallel
	//std::cout << "The number of processes:"<< coreNum <<std::endl;
	string filename = "./image/phanSneeze.jpg";
	OImage = imread(filename, 0);
	
	printf("-----------------------Program---Start---------------------------------\n");
	printf("Existence: %d\n", checkExistence(filename));
	printf("Image Size: %d X %d\n", OImage.rows, OImage.cols);
	printf("-----------------------------------------------------------------------\n");
	printf("-----------------------Canny---Process---------------------------------\n");
	CannyProcess();
	printf("-----------------------------------------------------------------------\n");
	printf("----------------------Image----Storage---------------------------------\n");
	imwrite("OImage.jpg", OImage);
	imwrite("BImage.jpg", BImage);
	imwrite("EMImage.jpg", EMImage);
	imwrite("EAImage.jpg", EAImage);
	imwrite("TEImage.jpg", TEImage);
	imwrite("DTImage.jpg", DTImage);
	printf("-----------------------------------------------------------------------\n");
	//release memory
	BImage.setTo(Scalar(0));
	EMImage.setTo(Scalar(0));
	EAImage.setTo(Scalar(0));
	TEImage.setTo(Scalar(0));
	DTImage.setTo(Scalar(0));
	Convert_Radius = 0;
	KernelWidth = 0;
	printf("-----------------------Program----End------------------------ ---------\n");
	return 0;
}


void CannyProcess()
{	
	auto begin = chrono::high_resolution_clock::now();
	GaussianBlur_sigma(); //gaussian blurring
	Gradient_Images(); // gradient calculation 
	NonMaxSuppress(); // non maximun suppression
	auto end = chrono::high_resolution_clock::now();
	auto time = chrono::duration_cast<chrono::microseconds>(end - begin);
	cout << "Time elapsed (/microseconds): " << time.count() << endl;

	DoubleThreshold_Hysteresis(13, 25);

}

void GaussianBlur_sigma()
{
	// # of rows and cols
	int rs = OImage.rows;
	int cs = OImage.cols;

	int ost = 2;

	// gaussian kernel with standard deviation of 1 along with the sum of elements 
	const int8_t kernel[] = { 1,4,7,4,1,
						  4,16,26,16,4,
						  7,26,41,26,7,
						  4,16,26,16,4,
						  1,4,7,4,1 };
	const int kernelDiv = 273;

	const int8_t kernel14[] = { 2,4,5,4,2,
						  4,9,12,9,4,
						  5,12,15,12,5,
						  4,9,12,9,4,
						  2,4,5,4,2 };
	const int kernelDiv114 = 115;

	//clone original image						  
	BImage = OImage.clone();

	//perform convolve
	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < rs; i++) {
		for (int j = 0; j < cs; j++) {
			if ((i < ost) || (i >= rs - ost) || (j < ost) || (j >= cs - ost)) {
				BImage.at<uchar>(i, j) = OImage.at<uchar>(i, j);
				continue;
			}
			int count = 0;
			double conv = 0;
			for (int x = 0; x < (ost * 2) + 1; x++) {
				for (int y = 0; y < (ost * 2) + 1; y++) {
					conv += kernel[count] * (double)(OImage.at<uchar>(i + x - ost,
						j + y - ost));
					count++;
				}
			}
			BImage.at<uchar>(i, j) = conv / kernelDiv;
		}
	}
	#pragma omp barrier
}

void Gradient_Images()
{
	EMImage = Mat::zeros(BImage.rows, BImage.cols, CV_8UC1); // Initialize the Mag output
	EAImage = Mat::zeros(BImage.rows, BImage.cols, CV_8UC1); // Initialize the angle output

	if (mode == 1) {
		float** xkernel = new float* [3]; // For sobel kernal 
		float** ykernel = new float* [3];
		for (int x_i = 0; x_i < 3; x_i++)
		{
			xkernel[x_i] = new float [3];
			ykernel[x_i] = new float [3];
			memcpy(xkernel[x_i], sobelx[x_i], 3 * sizeof(float));
			memcpy(ykernel[x_i], sobely[x_i], 3 * sizeof(float));
		}
		
		convuloution(xkernel, ykernel, 1, 3);
	}
	else if (mode == 2)
	{
		float** xkernel = new float*[3]; // for prewitt
		float** ykernel = new float*[3];
		for (int x_i = 0; x_i < 3; x_i++)
		{
			xkernel[x_i] = new float[3];
			ykernel[x_i] = new float[3];
			memcpy(xkernel[x_i], prewittx[x_i], 3 * sizeof(float));
			memcpy(ykernel[x_i], prewitty[x_i], 3 * sizeof(float));
		}

		convuloution(xkernel, ykernel, 1, 3);
	}
	else if (mode == 3)
	{
		float** xkernel = new float*[2]; // for robert
		float** ykernel = new float*[2];
		for (int x_i = 0; x_i < 2; x_i++)
		{
			xkernel[x_i] = new float[2];
			ykernel[x_i] = new float[2];
			memcpy(xkernel[x_i], robertsx[x_i], 2 * sizeof(float));
			memcpy(ykernel[x_i], robertsy[x_i], 2 * sizeof(float));
		}

		convuloution_2(xkernel, ykernel, 1, 2);
	}
	else if (mode == 4)
	{
		float** xkernel = new float*[5]; // for 5*5 sobel
		float** ykernel = new float*[5];
		for (int x_i = 0; x_i < 5; x_i++)
		{
			xkernel[x_i] = new float[5];
			ykernel[x_i] = new float[5];
			memcpy(xkernel[x_i], sobel5x[x_i], 5 * sizeof(float));
			memcpy(ykernel[x_i], sobel5y[x_i], 5 * sizeof(float));
		}

		convuloution_5(xkernel, ykernel, 2, 5);
	}

	else {
		float** xkernel = new float*[3]; //  for scharr
		float** ykernel = new float*[3];
		for (int x_i = 0; x_i < 3; x_i++)
		{
			xkernel[x_i] = new float[3];
			ykernel[x_i] = new float[3];
			memcpy(xkernel[x_i], scharrx[x_i], 3 * sizeof(float));
			memcpy(ykernel[x_i], scharry[x_i], 3 * sizeof(float));
		}

		convuloution(xkernel, ykernel, 1, 3);
	}
}

// covolution functions for gradient caluate.
void convuloution(float** xkernel, float** ykernel, int sobelRad, int sobelWidth)
{	

	int sumGradient = 0;

    #pragma omp parallel for num_threads(4)
	for (int i = 0; i < BImage.rows; i++)
	{
		for (int j = 0; j < BImage.cols; j++)
		{
			// Declaring boundary pixel.
			if (i == sobelRad - 1 || i == BImage.rows - sobelRad || j == sobelRad - 1 || j == BImage.cols - sobelRad)
			{
				EMImage.at<uchar>(i, j) = 0;
				EAImage.at<uchar>(i, j) = 255;
				//printf("%d\t%d\n", i, j);
			}
			else
			{
				int sumX = 0;
				int sumY = 0;
				// Convoluton with kernals.
				for (int x = 0; x < sobelWidth; x++)
					for (int y = 0; y < sobelWidth; y++)
					{
						sumX += xkernel[x][y] * BImage.at<uchar>(i + x - sobelRad, j + y - sobelRad);
						sumY += ykernel[x][y] * BImage.at<uchar>(i + x - sobelRad, j + y - sobelRad);
					}

				int mag = sqrt(sumX*sumX + sumY * sumY);
				if (mag > 255)  mag = 255;
				EMImage.at<uchar>(i, j) = mag;

				sumGradient += mag;

				int ang = (atan2(sumY, sumX) / M_PI) * 180;
				// Angle
				//4 angle, 0 45 90 135
				if (((ang < 22.5) && (ang >= -22.5)) || (ang >= 157.5) || (ang < -157.5))
					ang = 0;
				if (((ang >= 22.5) && (ang < 67.5)) || ((ang < -112.5) && (ang >= -157.5)))
					ang = 45;
				if (((ang >= 67.5) && (ang < 112.5)) || ((ang < -67.5) && (ang >= -112.5)))
					ang = 90;
				if (((ang >= 112.5) && (ang < 157.5)) || ((ang < -22.5) && (ang >= -67.5)))
					ang = 135;
				EAImage.at<uchar>(i, j) = ang;

			}
		}
	}
	
	
	#pragma omp barrier

	avgGradient = float(sumGradient) / float(BImage.cols * BImage.rows);
	printf("average gradient: %.2f\n", avgGradient);

	float sumVar = 0;

	// Calculate the avg
	for (int i = 0; i < BImage.rows; i++)
	{
		for (int j = 0; j < BImage.cols; j++)
		{
			sumVar += (EMImage.at<uchar>(i, j) - avgGradient) * (EMImage.at<uchar>(i, j) - avgGradient);
		}
	}

	var = sqrt(sumVar / (BImage.cols * BImage.rows));
	printf("average gradient: %.2f\n", var);
}

void convuloution_2(float** xkernel, float** ykernel, int sobelRad, int sobelWidth)
{

	int sumGradient = 0;

	#pragma omp parallel for num_threads(4)
	
	for (int i = 0; i < BImage.rows; i++)
	{
		for (int j = 0; j < BImage.cols; j++)
		{	
			// Declaring boundary pixel.
			if (i == sobelRad - 1 || j == sobelRad - 1 )
			{
				EMImage.at<uchar>(i, j) = 0;
				EAImage.at<uchar>(i, j) = 255;
				//printf("%d\t%d\n", i, j);
			}
			else
			{
				int sumX = 0;
				int sumY = 0;
				// Convoluton with kernals.
				for (int x = 0; x < sobelWidth; x++)
					for (int y = 0; y < sobelWidth; y++)
					{
						//printf("%d\t%d\t%d\t%d%d\t%d\n", i, j, x, y, xkernel[x][y], ykernel[x][y]);
						sumX += xkernel[x][y] * BImage.at<uchar>(i + x - sobelRad, j + y - sobelRad);
						sumY += ykernel[x][y] * BImage.at<uchar>(i + x - sobelRad, j + y - sobelRad);
						//printf("%d\t%d\t%d\t%d%d\t%d\n", i, j, x, y, sumX, sumY);
					}

				int mag = sqrt(sumX*sumX + sumY * sumY);
				if (mag > 255)  mag = 255;
				EMImage.at<uchar>(i, j) = mag;

				sumGradient += mag;

				int ang = (atan2(sumY, sumX) / M_PI) * 180;
				// Angle
				//4 angle, 0 45 90 135
				if (((ang < 22.5) && (ang >= -22.5)) || (ang >= 157.5) || (ang < -157.5))
					ang = 0;
				if (((ang >= 22.5) && (ang < 67.5)) || ((ang < -112.5) && (ang >= -157.5)))
					ang = 45;
				if (((ang >= 67.5) && (ang < 112.5)) || ((ang < -67.5) && (ang >= -112.5)))
					ang = 90;
				if (((ang >= 112.5) && (ang < 157.5)) || ((ang < -22.5) && (ang >= -67.5)))
					ang = 135;
				EAImage.at<uchar>(i, j) = ang;

			}
		}
	}

	#pragma omp barrier	

	avgGradient = float(sumGradient) / float(BImage.cols * BImage.rows);
	printf("average gradient: %.2f\n", avgGradient);

	float sumVar = 0;
	// Calculate the avg.
	for (int i = 0; i < BImage.rows; i++)
	{
		for (int j = 0; j < BImage.cols; j++)
		{
			sumVar += (EMImage.at<uchar>(i, j) - avgGradient) * (EMImage.at<uchar>(i, j) - avgGradient);
		}
	}

	var = sqrt(sumVar / (BImage.cols * BImage.rows));
	printf("average gradient: %.2f\n", var);
}

void convuloution_5(float** xkernel, float** ykernel, int sobelRad, int sobelWidth)
{

	int sumGradient = 0;
	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < BImage.rows; i++)
	{
		for (int j = 0; j < BImage.cols; j++)
		{	
			// Declaring boundary pixel.
			if (i == sobelRad - 1 || i == sobelRad - 2 || i == BImage.rows - sobelRad || i == BImage.rows - sobelRad + 1 || j == sobelRad - 1 || j == sobelRad - 2 || j == BImage.cols - sobelRad || j == BImage.cols - sobelRad + 1)
			{
				EMImage.at<uchar>(i, j) = 0;
				EAImage.at<uchar>(i, j) = 255;
				//printf("%d\t%d\n", i, j);
			}
			else
			{
				int sumX = 0;
				int sumY = 0;
				// Convoluton with kernals.
				for (int x = 0; x < sobelWidth; x++)
					for (int y = 0; y < sobelWidth; y++)
					{
						//printf("%d\t%d\t%d\t%d%d\t%d\n", i, j, x, y, xkernel[x][y], ykernel[x][y]);
						sumX += xkernel[x][y] * BImage.at<uchar>(i + x - sobelRad, j + y - sobelRad);
						sumY += ykernel[x][y] * BImage.at<uchar>(i + x - sobelRad, j + y - sobelRad);
						//printf("%d\t%d\t%d\t%d%d\t%d\n", i, j, x, y, sumX, sumY);
					}

				int mag = sqrt(sumX*sumX + sumY * sumY);
				if (mag > 255)  mag = 255;
				EMImage.at<uchar>(i, j) = mag;

				sumGradient += mag;
				

				int ang = (atan2(sumY, sumX) / M_PI) * 180;
				//  Angle
				//4 angle, 0 45 90 135
				if (((ang < 22.5) && (ang >= -22.5)) || (ang >= 157.5) || (ang < -157.5))
					ang = 0;
				if (((ang >= 22.5) && (ang < 67.5)) || ((ang < -112.5) && (ang >= -157.5)))
					ang = 45;
				if (((ang >= 67.5) && (ang < 112.5)) || ((ang < -67.5) && (ang >= -112.5)))
					ang = 90;
				if (((ang >= 112.5) && (ang < 157.5)) || ((ang < -22.5) && (ang >= -67.5)))
					ang = 135;
				EAImage.at<uchar>(i, j) = ang;

			}
		}
	}

	#pragma omp barrier

	avgGradient = float(sumGradient) / float(BImage.cols * BImage.rows);
	printf("average gradient: %.2f\n", avgGradient);

	float sumVar = 0;
	// calculate the avg.
	for (int i = 0; i < BImage.rows; i++)
	{
		for (int j = 0; j < BImage.cols; j++)
		{
			sumVar += (EMImage.at<uchar>(i, j) - avgGradient) * (EMImage.at<uchar>(i, j) - avgGradient);
		}
	}

	var = sqrt(sumVar / (BImage.cols * BImage.rows));
	printf("average gradient: %.2f\n", var);
}

void NonMaxSuppress()
{
	TEImage = EMImage.clone();
	int m = TEImage.rows, n = TEImage.cols;

	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
				TEImage.at<uchar>(i, j) = 0;
			}
			else
			{
				switch (EAImage.at<uchar>(i, j)) {
					//0 degree direction, left and right
				case 0:
					if (EMImage.at<uchar>(i, j) < EMImage.at<uchar>(i, j + 1) || EMImage.at<uchar>(i, j) < EMImage.at<uchar>(i, j - 1))
						TEImage.at<uchar>(i, j) = 0;
					break;
					//45 degree direction,up right and down left
				case 45:
					if (EMImage.at<uchar>(i, j) < EMImage.at<uchar>(i + 1, j - 1) || EMImage.at<uchar>(i, j) < EMImage.at<uchar>(i - 1, j + 1))
						TEImage.at<uchar>(i, j) = 0;
					break;
					//90 degree direction, up and down
				case 90:
					if (EMImage.at<uchar>(i, j) < EMImage.at<uchar>(i + 1, j) || EMImage.at<uchar>(i, j) < EMImage.at<uchar>(i - 1, j))
						TEImage.at<uchar>(i, j) = 0;
					break;
					//135 degree direction, up left and down right
				case 135:
					if (EMImage.at<uchar>(i, j) < EMImage.at<uchar>(i - 1, j - 1) || EMImage.at<uchar>(i, j) < EMImage.at<uchar>(i + 1, j + 1))
						TEImage.at<uchar>(i, j) = 0;
					break;
				}
			}
		}
	}
	#pragma omp barrier
}





void DoubleThreshold_Hysteresis(int lowTh, int highthreshold)
{
	DTImage = TEImage.clone();

	for (int i = 0; i < DTImage.rows; i++)
	{
		for (int j = 0; j < DTImage.cols; j++)
		{
			if (TEImage.at<uchar>(i, j) > highthreshold)
				DTImage.at<uchar>(i, j) = 255;
			else if (TEImage.at<uchar>(i, j) < lowTh)
				DTImage.at<uchar>(i, j) = 0;
			else
			{
				bool isHigher = false;
				bool doConnect = false;
				for (int x = i - 1; x < i + 2; x++)
				{
					for (int y = j - 1; y < j + 2; y++)
					{
						if (x <= 0 || y <= 0 || x > DTImage.rows || y > DTImage.cols)
							continue;
						else
						{
							if (TEImage.at<uchar>(x, y) > highthreshold)
							{
								DTImage.at<uchar>(i, j) = 255;
								isHigher = true;
								break;
							}
							else if (TEImage.at<uchar>(x, y) <= highthreshold && TEImage.at<uchar>(x, y) >= lowTh)
								doConnect = true;
						}
					}
					if (isHigher)    break;
				}
				if (!isHigher && doConnect)
					for (int x = i - 2; x < i + 3; x++)
					{
						for (int y = j - 2; y < j + 3; y++)
						{
							if (x < 0 || y < 0 || x > DTImage.rows || y > DTImage.cols)
								continue;
							else
							{
								if (TEImage.at<uchar>(x, y) > highthreshold)
								{
									DTImage.at<uchar>(i, j) = 255;
									isHigher = true;
									break;
								}
							}
						}
						if (isHigher)    break;
					}
				if (!isHigher)   DTImage.at<uchar>(i, j) = 0;
			}
		}
	}
}

bool checkExistence(std::string filename)
{
	std::ifstream f;
	f.open(filename);

	return f.is_open();
}