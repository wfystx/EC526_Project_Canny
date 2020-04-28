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
#include "kernels.h" // Head file for differnt kernals of gradient calculation.
#include "opencv2/core.hpp"  //OpenCV lib. 
#include "opencv2/imgproc.hpp" //OpenCV lib.
#include "opencv2/highgui.hpp" //OpenCV lib. 
#include "omp.h" // OpenMP head file.


/*
	Global constant Delaration.
*/
#define M_PI 3.14159265358979323846
#define mode 1// 1 for sobel, 2 for prewitt, 3 for Robert, 4 for 5X Sobel, else scharr

using namespace cv;

/*
  Function Delaration.
*/
bool checkExistence(string filename); // Check whether image exists.
void createGaussianKernel(int); // Generate gaussian kernal for blurring.
void cannyDector(); // Canny dector.
void useGaussianBlur();
void getGradientImg();
void nonMaxSuppress();
void lessHysteresisThreshold(int, int);
void moreHysteresisThreshold();
void convuloution(float** x, float** y, int sobelRad, int sobelWidth); // colvolution for 3X3.
void convuloution_2(float** x, float** y, int sobelRad, int sobelWidth);  // colvolution for 5X5.
void convuloution_5(float** x, float** y, int sobelRad, int sobelWidth); // colvolution for 5X5.
float** matrixgenerator(int col, int row);
Mat combineImage();


/*
  Function Variables Delaration.
*/
Mat OImage; // Original Image Mat
Mat BImage; // blured Image Mat
Mat EMImage; // edgeMag Image Mat
Mat EAImage; // edgeAng Image Mat
Mat TEImage; // thinEdge Image Mat
Mat thresholdImage; // thinEdge Image Mat
Mat lowTho, highTho, sobelX, sobelY;
int *gaussianMask, maskRad, maskWidth = 0, maskSum = 0;
float sigma = 0.0, avgGradient = 0.0, var = 0.0;

int main(int argc, char** argv)
{
	Mat combinedImage;
	Mat outs;
	int coreNum = omp_get_num_procs(); // Show the process of parallel
	std::cout << "The number of processes:"<< coreNum <<std::endl;
	string filename = "./image/lena.jpg";
	OImage = imread(filename, 0);
	
	printf("Existence: %d\n", checkExistence(filename));
	printf("Image Size: %d X %d\n", OImage.rows, OImage.cols);
	//std::cout << "Existence " << checkExistence(filename) << endl;
	//std::cout << "image height " << OImage.rows << endl;

	bool isNewSigma = true;
	while (isNewSigma)
	{
		char wndName[] = "Canny Process";
		isNewSigma = false;
		createGaussianKernel(0);
		cannyDector();
		imwrite("./TEImage_sobel.jpg", TEImage);
		//combine all images for showing
		//combinedImage = combineImage();
		//if (combinedImage.rows > 600) {
		//	resize(combinedImage, combinedImage, Size(combinedImage.cols / 1.4, combinedImage.rows / 1.4));
		//}

				// to fix graphics issue)
		Mat res;
		//combinedImage.convertTo(res,CV_8UC3,255);
		BImage.convertTo(res, CV_8UC3, 255);
		imwrite("bluredImage.jpg", res);
		EMImage.convertTo(res, CV_8UC3, 255);
		imwrite("edgeMagImage.jpg", res);
		EAImage.convertTo(res, CV_8UC3, 255);
		imwrite("edgeAngImage.jpg", res);
		TEImage.convertTo(res, CV_8UC3, 255);
		imwrite("thinEdgeImage.jpg", res);
		thresholdImage.convertTo(res, CV_8UC3, 255);
		imwrite("thresholdImage.jpg", res);

		//imshow(wndName, combinedImage);
		//waitKey(10);

		//char tryNewSigma;
		//printf("Do you want to try other sigma?(Y/N): ");
		//scanf_s("%s", &tryNewSigma);
		//if (tryNewSigma == 'y' || tryNewSigma == 'Y') {
		//	isNewSigma = true;
		//	printf("\n-------------Please Try Another Sigma-------------\n");
		//	destroyWindow(wndName);
		//	combinedImage.release();
		//}
		//release memory
		free(gaussianMask);
		BImage.setTo(Scalar(0));
		EMImage.setTo(Scalar(0));
		sobelY.setTo(Scalar(0));
		sobelX.setTo(Scalar(0));
		EAImage.setTo(Scalar(0));
		TEImage.setTo(Scalar(0));
		thresholdImage.setTo(Scalar(0));
		sigma = 0.0;
		maskRad = 0;
		maskWidth = 0;
		maskSum = 0;
	}
	printf("-------Program End-------\n");
	return 0;
}
//Create Gaussian Kernel.
void createGaussianKernel(int widthType)
{
	//printf("Please input standard deviation(>0) and press Enter: ");
	//scanf_s("%f", &sigma);
	sigma = 1;

	if (sigma < 0.01) sigma = 0.01;
	//compute mask width according to sigma value
	if (widthType == 0) {
		//For canny
		maskWidth = int((sigma - 0.01) * 3) * 2 + 1;
	}
	else if (widthType == 1) {
		//for LoG
		maskWidth = 5;
	}

	if (maskWidth < 1)   maskWidth = 1;
	printf("Sigma is %.2f, Mask Width is %d.\n", sigma, maskWidth);
	//declare mask as dynamic memory
	gaussianMask = (int*)malloc(maskWidth * maskWidth * sizeof(int));

	//double gaussianMaskDou[maskWidth][maskWidth], maskMin = 0.0;
	//int gaussianMaskInt[maskWidth][maskWidth];
	double maskMin = 0.0;
	double** gaussianMaskDou = new double*[maskWidth];
	int** gaussianMaskInt = new int*[maskWidth];

	for (int index = 0; index < maskWidth; index++)
	{
		gaussianMaskDou[index] = new double[maskWidth];
		gaussianMaskInt[index] = new int[maskWidth];
	}

	maskRad = maskWidth / 2;
	int i, j;
	//construct the gaussian mask
	for (int x = -maskRad; x <= maskRad; x++)
	{
		for (int y = -maskRad; y <= maskRad; y++)
		{
			i = x + maskRad;
			j = y + maskRad;
			//gaussian 2d function
			gaussianMaskDou[i][j] = exp((x*x + y * y) / (-2 * sigma*sigma));
			//min value of mask is the first one
			if (i == 0 && j == 0)  maskMin = gaussianMaskDou[0][0];
			//convert mask value double to integer
			gaussianMaskInt[i][j] = cvRound(gaussianMaskDou[i][j] / maskMin);
			maskSum += gaussianMaskInt[i][j];
		}
	}

	//printf("Mask Sum is %d, rad is %d.\n", maskSum, maskRad);
	//represent mask using global pointer
	for (i = 0; i < maskWidth; i++)
		for (j = 0; j < maskWidth; j++)
			*(gaussianMask + i * maskWidth + j) = gaussianMaskInt[i][j];
}

void cannyDector()
{
	useGaussianBlur(); //gaussian blurring

	getGradientImg(); // gradient calculation 
	nonMaxSuppress(); // non maximun suppression

	if (mode == 1) {
		int highTh = 0;
		highTh = avgGradient + 1.2 * var;
		printf("low: %d high: %d \n", int(highTh / 2), highTh);
		//lessHysteresisThreshold(int(highTh / 2), highTh);
		//lessHysteresisThreshold(25, 50);
	}
	//else if (mode == 2) {
	//	moreHysteresisThreshold();
	//}
	//else if (mode == 0) {
	//	lessHysteresisThreshold(32, 64);
	//}

}
void useGaussianBlur()
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
	//clone original image						  
	BImage = OImage.clone();

	//perform convolve
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
}

void getGradientImg()
{
	EMImage = Mat::zeros(BImage.rows, BImage.cols, CV_8UC1); // Initialize the Mag output
	EAImage = Mat::zeros(BImage.rows, BImage.cols, CV_8UC1); // Initialize the angle output
	sobelX = Mat::zeros(BImage.rows, BImage.cols, CV_8UC1);
	sobelY = Mat::zeros(BImage.rows, BImage.cols, CV_8UC1);

	if (mode == 1) {
		float** xMask = new float* [3]; // For sobel kernal 
		float** yMask = new float* [3];
		for (int x_i = 0; x_i < 3; x_i++)
		{
			xMask[x_i] = new float [3];
			yMask[x_i] = new float [3];
			memcpy(xMask[x_i], sobelx[x_i], 3 * sizeof(float));
			memcpy(yMask[x_i], sobely[x_i], 3 * sizeof(float));
		}
		
		convuloution(xMask, yMask, 1, 3);
	}
	else if (mode == 2)
	{
		float** xMask = new float*[3]; // for prewitt
		float** yMask = new float*[3];
		for (int x_i = 0; x_i < 3; x_i++)
		{
			xMask[x_i] = new float[3];
			yMask[x_i] = new float[3];
			memcpy(xMask[x_i], prewittx[x_i], 3 * sizeof(float));
			memcpy(yMask[x_i], prewitty[x_i], 3 * sizeof(float));
		}

		convuloution(xMask, yMask, 1, 3);
	}
	else if (mode == 3)
	{
		float** xMask = new float*[2]; // for robert
		float** yMask = new float*[2];
		for (int x_i = 0; x_i < 2; x_i++)
		{
			xMask[x_i] = new float[2];
			yMask[x_i] = new float[2];
			memcpy(xMask[x_i], robertsx[x_i], 2 * sizeof(float));
			memcpy(yMask[x_i], robertsy[x_i], 2 * sizeof(float));
		}

		convuloution_2(xMask, yMask, 1, 2);
	}
	else if (mode == 4)
	{
		float** xMask = new float*[5]; // for 5*5 sobel
		float** yMask = new float*[5];
		for (int x_i = 0; x_i < 5; x_i++)
		{
			xMask[x_i] = new float[5];
			yMask[x_i] = new float[5];
			memcpy(xMask[x_i], sobel5x[x_i], 5 * sizeof(float));
			memcpy(yMask[x_i], sobel5y[x_i], 5 * sizeof(float));
		}

		convuloution_5(xMask, yMask, 2, 5);
	}

	else {
		float** xMask = new float*[3]; //  for scharr
		float** yMask = new float*[3];
		for (int x_i = 0; x_i < 3; x_i++)
		{
			xMask[x_i] = new float[3];
			yMask[x_i] = new float[3];
			memcpy(xMask[x_i], scharrx[x_i], 3 * sizeof(float));
			memcpy(yMask[x_i], scharry[x_i], 3 * sizeof(float));
		}

		convuloution(xMask, yMask, 1, 3);
	}
}

// covolution functions for gradient caluate.
void convuloution(float** xMask, float** yMask, int sobelRad, int sobelWidth)
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
				sobelX.at<uchar>(i, j) = 0;
				sobelY.at<uchar>(i, j) = 0;
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
						sumX += xMask[x][y] * BImage.at<uchar>(i + x - sobelRad, j + y - sobelRad);
						sumY += yMask[x][y] * BImage.at<uchar>(i + x - sobelRad, j + y - sobelRad);
					}

				int mag = sqrt(sumX*sumX + sumY * sumY);
				if (mag > 255)  mag = 255;
				EMImage.at<uchar>(i, j) = mag;

				sumGradient += mag;
				//Process sobel X
				if (sumX < 0) {
					if (sumX < -255 || sumX == -255) {
						sobelX.at<uchar>(i, j) = 255;
					}
					else {
						sobelX.at<uchar>(i, j) = sumX * (-1);
					}
				}
				else if (sumX > 255 || sumX == 255) {
					sobelX.at<uchar>(i, j) = 255;
				}
				else {
					sobelX.at<uchar>(i, j) = sumX;
				}
				//Process soble Y
				if (sumY < 0) {
					if (sumY < -255 || sumY == -255) {
						sobelY.at<uchar>(i, j) = 255;
					}
					else {
						sobelY.at<uchar>(i, j) = sumY * (-1);
					}
				}
				else if (sumY > 255 || sumY == 255) {
					sobelY.at<uchar>(i, j) = 255;
				}
				else {
					sobelY.at<uchar>(i, j) = sumY;
				}

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

void convuloution_2(float** xMask, float** yMask, int sobelRad, int sobelWidth)
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
				sobelX.at<uchar>(i, j) = 0;
				sobelY.at<uchar>(i, j) = 0;
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
						//printf("%d\t%d\t%d\t%d%d\t%d\n", i, j, x, y, xMask[x][y], yMask[x][y]);
						sumX += xMask[x][y] * BImage.at<uchar>(i + x - sobelRad, j + y - sobelRad);
						sumY += yMask[x][y] * BImage.at<uchar>(i + x - sobelRad, j + y - sobelRad);
						//printf("%d\t%d\t%d\t%d%d\t%d\n", i, j, x, y, sumX, sumY);
					}

				int mag = sqrt(sumX*sumX + sumY * sumY);
				if (mag > 255)  mag = 255;
				EMImage.at<uchar>(i, j) = mag;

				sumGradient += mag;
				//Process sobel X
				if (sumX < 0) {
					if (sumX < -255 || sumX == -255) {
						sobelX.at<uchar>(i, j) = 255;
					}
					else {
						sobelX.at<uchar>(i, j) = sumX * (-1);
					}
				}
				else if (sumX > 255 || sumX == 255) {
					sobelX.at<uchar>(i, j) = 255;
				}
				else {
					sobelX.at<uchar>(i, j) = sumX;
				}
				//Process soble Y
				if (sumY < 0) {
					if (sumY < -255 || sumY == -255) {
						sobelY.at<uchar>(i, j) = 255;
					}
					else {
						sobelY.at<uchar>(i, j) = sumY * (-1);
					}
				}
				else if (sumY > 255 || sumY == 255) {
					sobelY.at<uchar>(i, j) = 255;
				}
				else {
					sobelY.at<uchar>(i, j) = sumY;
				}

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

void convuloution_5(float** xMask, float** yMask, int sobelRad, int sobelWidth)
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
				sobelX.at<uchar>(i, j) = 0;
				sobelY.at<uchar>(i, j) = 0;
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
						//printf("%d\t%d\t%d\t%d%d\t%d\n", i, j, x, y, xMask[x][y], yMask[x][y]);
						sumX += xMask[x][y] * BImage.at<uchar>(i + x - sobelRad, j + y - sobelRad);
						sumY += yMask[x][y] * BImage.at<uchar>(i + x - sobelRad, j + y - sobelRad);
						//printf("%d\t%d\t%d\t%d%d\t%d\n", i, j, x, y, sumX, sumY);
					}

				int mag = sqrt(sumX*sumX + sumY * sumY);
				if (mag > 255)  mag = 255;
				EMImage.at<uchar>(i, j) = mag;

				sumGradient += mag;
				//Process sobel X
				if (sumX < 0) {
					if (sumX < -255 || sumX == -255) {
						sobelX.at<uchar>(i, j) = 255;
					}
					else {
						sobelX.at<uchar>(i, j) = sumX * (-1);
					}
				}
				else if (sumX > 255 || sumX == 255) {
					sobelX.at<uchar>(i, j) = 255;
				}
				else {
					sobelX.at<uchar>(i, j) = sumX;
				}
				//Process soble Y
				if (sumY < 0) {
					if (sumY < -255 || sumY == -255) {
						sobelY.at<uchar>(i, j) = 255;
					}
					else {
						sobelY.at<uchar>(i, j) = sumY * (-1);
					}
				}
				else if (sumY > 255 || sumY == 255) {
					sobelY.at<uchar>(i, j) = 255;
				}
				else {
					sobelY.at<uchar>(i, j) = sumY;
				}

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

//float** matrixgenerator(int col, int row)
//{
//	float** xMask = new float*[3];
//	float** yMask = new float*[3];
//	for (int x_i = 0; x_i < 3; x_i++)
//	{
//		xMask[x_i] = new float[3];
//		yMask[x_i] = new float[3];
//	}
//	memcpy(xMask, scharrx, 9 * sizeof(float));
//	memcpy(yMask, scharry, 9 * sizeof(float));
//}



void nonMaxSuppress()
{
	std::cout << "performing nonMaxSuppress" << std::endl;
	TEImage = EMImage.clone();
	int m = TEImage.rows, n = TEImage.cols;

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
}





void lessHysteresisThreshold(int lowTh, int highTh)
{
	thresholdImage = TEImage.clone();

	for (int i = 0; i < thresholdImage.rows; i++)
	{
		for (int j = 0; j < thresholdImage.cols; j++)
		{
			if (TEImage.at<uchar>(i, j) > highTh)
				thresholdImage.at<uchar>(i, j) = 255;
			else if (TEImage.at<uchar>(i, j) < lowTh)
				thresholdImage.at<uchar>(i, j) = 0;
			else
			{
				bool isHigher = false;
				bool doConnect = false;
				for (int x = i - 1; x < i + 2; x++)
				{
					for (int y = j - 1; y < j + 2; y++)
					{
						if (x <= 0 || y <= 0 || x > thresholdImage.rows || y > thresholdImage.cols)
							continue;
						else
						{
							if (TEImage.at<uchar>(x, y) > highTh)
							{
								thresholdImage.at<uchar>(i, j) = 255;
								isHigher = true;
								break;
							}
							else if (TEImage.at<uchar>(x, y) <= highTh && TEImage.at<uchar>(x, y) >= lowTh)
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
							if (x < 0 || y < 0 || x > thresholdImage.rows || y > thresholdImage.cols)
								continue;
							else
							{
								if (TEImage.at<uchar>(x, y) > highTh)
								{
									thresholdImage.at<uchar>(i, j) = 255;
									isHigher = true;
									break;
								}
							}
						}
						if (isHigher)    break;
					}
				if (!isHigher)   thresholdImage.at<uchar>(i, j) = 0;
			}
		}
	}
}

void moreHysteresisThreshold()
{
	lowTho = Mat::zeros(BImage.rows, BImage.cols, CV_8UC1);
	highTho = Mat::zeros(BImage.rows, BImage.cols, CV_8UC1);
	Mat avg = Mat::zeros(BImage.rows, BImage.cols, CV_32FC1);
	Mat var = Mat::zeros(BImage.rows, BImage.cols, CV_32FC1);

	for (int i = 0; i < BImage.rows; i++) {
		for (int j = 0; j < BImage.cols; j++) {

			float sumGra = 0;
			for (int x = i - 10; x < i + 11; x++) {
				for (int y = j - 10; y < j + 11; y++) {
					float gra;
					if (x < 0 || y < 0) {
						gra = 0;
					}
					else {
						gra = EMImage.at<uchar>(x, y);
					}

					sumGra += gra;
				}
			}
			avg.at<float>(i, j) = sumGra / float(21 * 21);
			//printf("%0.2f ", avg.at<float>(i,j));
		}
	}

	for (int i = 0; i < BImage.rows; i++) {
		for (int j = 0; j < BImage.cols; j++) {

			float sumVar = 0;
			for (int x = i - 10; x < i + 11; x++) {
				for (int y = j - 10; y < j + 11; y++) {
					float gra;
					if (x < 0 || y < 0) {
						gra = 0;
					}
					else {
						gra = EMImage.at<uchar>(x, y);
					}

					sumVar += (gra - avg.at<float>(i, j))*(gra - avg.at<float>(i, j));
				}
			}
			var.at<float>(i, j) = sqrt(sumVar / float(21 * 21));
			//printf("%0.2f ", var.at<float>(i,j));
		}
	}

	int lowTh, highTh;
	thresholdImage = TEImage.clone();

	for (int i = 0; i < thresholdImage.rows; i++)
	{
		for (int j = 0; j < thresholdImage.cols; j++)
		{
			highTh = int(avg.at<float>(i, j) + 1.1*var.at<float>(i, j));
			lowTh = highTh / 2;

			if (TEImage.at<uchar>(i, j) < int(avg.at<float>(i, j) / 5)) {
				thresholdImage.at<uchar>(i, j) = 0;
			}
			else { //added

				if (TEImage.at<uchar>(i, j) > highTh)
					thresholdImage.at<uchar>(i, j) = 255;
				else if (TEImage.at<uchar>(i, j) < lowTh)
					thresholdImage.at<uchar>(i, j) = 0;
				else
				{
					bool isHigher = false;
					bool doConnect = false;
					for (int x = i - 1; x < i + 2; x++)
					{
						for (int y = j - 1; y < j + 2; y++)
						{
							if (x <= 0 || y <= 0 || x > thresholdImage.rows || y > thresholdImage.cols)
								continue;
							else
							{
								if (TEImage.at<uchar>(x, y) > highTh)
								{
									thresholdImage.at<uchar>(i, j) = 255;
									isHigher = true;
									break;
								}
								else if (TEImage.at<uchar>(x, y) <= highTh && TEImage.at<uchar>(x, y) >= lowTh)
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
								if (x < 0 || y < 0 || x > thresholdImage.rows || y > thresholdImage.cols)
									continue;
								else
								{
									if (TEImage.at<uchar>(x, y) > highTh)
									{
										thresholdImage.at<uchar>(i, j) = 255;
										isHigher = true;
										break;
									}
								}
							}
							if (isHigher)    break;
						}
					if (!isHigher)   thresholdImage.at<uchar>(i, j) = 0;
				}
			}//added
		}
	}
}

Mat combineImage()
{
	Mat h1CombineImage, h2CombineImage, allImage;
	Mat extraImage = Mat(OImage.rows, OImage.cols, CV_8UC1, Scalar(255));
	char sigmaChar[10];
	sprintf_s(sigmaChar, "%.2f", sigma);

	putText(extraImage, "Ori, Gaus, Grad, Grad X", Point(10, 20), FONT_HERSHEY_PLAIN, 1, Scalar(0));
	putText(extraImage, "NMS, Threshold, White, Grad Y", Point(10, 38), FONT_HERSHEY_PLAIN, 1, Scalar(0));
	putText(extraImage, "Sigma: ", Point(10, 56), FONT_HERSHEY_PLAIN, 1, Scalar(0));
	putText(extraImage, sigmaChar, Point(65, 56), FONT_HERSHEY_PLAIN, 1, Scalar(0));

	hconcat(OImage, BImage, h1CombineImage);
	hconcat(h1CombineImage, EMImage, h1CombineImage);
	hconcat(h1CombineImage, sobelY, h1CombineImage);
	hconcat(TEImage, thresholdImage, h2CombineImage);
	hconcat(h2CombineImage, extraImage, h2CombineImage);
	hconcat(h2CombineImage, sobelX, h2CombineImage);
	vconcat(h1CombineImage, h2CombineImage, allImage);

	return allImage;
}

bool checkExistence(std::string filename)
{
	std::ifstream f;
	f.open(filename);

	return f.is_open();
}