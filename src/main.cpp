#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <opencv/cxmisc.h>
#include <opencv/cvaux.h>
#include <vector>
#include <string>
#include <algorithm>
#include <ctype.h>

using namespace std;
using namespace cv;

void loadImagePair(Mat &img1, Mat &img2, int i)
{
    stringstream ss1, ss2;

    ss1 << "images/imgLeft_" << i << ".png";
    ss2 << "images/imgRight_" << i << ".png";

    img1 = imread(ss1.str());
    img2 = imread(ss2.str());
}

void correspondenciaBM(Mat &imagenIzq, Mat &imagenDer)
{
	Mat imagenIzq1CH, imagenDer1CH;

	Mat MapaDispBM(imagenIzq.size().height, imagenIzq.size().width, CV_16S, Scalar(0));
	Mat MapaDispBM_Norm(imagenIzq.size().height, imagenIzq.size().width, CV_8U, Scalar(0));

	StereoBM stereoBM;

	stereoBM.state->preFilterType = 1;
	stereoBM.state->preFilterSize = 41;
	stereoBM.state->preFilterCap = 31;
	stereoBM.state->SADWindowSize = 31;
	stereoBM.state->minDisparity = -65;
	stereoBM.state->numberOfDisparities = 128;
	stereoBM.state->textureThreshold = 10;
	stereoBM.state->uniquenessRatio = 15;

	cvtColor(imagenIzq, imagenIzq1CH, CV_BGR2GRAY); //Convierte las imagenes a CV_8UC1 (1 canal)
	cvtColor(imagenDer, imagenDer1CH, CV_BGR2GRAY);

	stereoBM(imagenIzq1CH, imagenDer1CH, MapaDispBM);

	normalize(MapaDispBM, MapaDispBM_Norm, 0, 255, CV_MINMAX, CV_8U);

	imshow("Mapa de Disparidad StereoBM", MapaDispBM_Norm);
}

void correspondenciaSGBM(Mat &imagenIzq, Mat &imagenDer)
{
	Mat MapaDispSGBM, MapaDispSGBM_Norm;

	StereoSGBM stereoSGBM;

	stereoSGBM.SADWindowSize = 3;
	stereoSGBM.numberOfDisparities = 144;
	stereoSGBM.preFilterCap = 63;
	stereoSGBM.minDisparity = -39;
	stereoSGBM.uniquenessRatio = 10;
	stereoSGBM.speckleWindowSize = 100;
	stereoSGBM.speckleRange = 32;
	stereoSGBM.disp12MaxDiff = 1;
	stereoSGBM.fullDP = false;
	stereoSGBM.P1 = 216;
	stereoSGBM.P2 = 864;

	stereoSGBM(imagenIzq, imagenDer, MapaDispSGBM);

	normalize(MapaDispSGBM, MapaDispSGBM_Norm, 0, 255, CV_MINMAX, CV_8U);

	imshow("Mapa de Disparidad StereoSGBM", MapaDispSGBM_Norm);
}

int main(int argc, char *argv[])
{
	VideoCapture capLeft(1); // open the Left camera
	VideoCapture capRight(0); // open the Right camera

	int numSnapshotLeft = 0, numSnapshotRight = 0; //Contador de fotos...
	string snapshotFilenameLeft = "0", snapshotFilenameRight = "0"; //Nombre del archivo de la captura...

	char key = 0;

	ofstream texto;
	texto.open("list.txt");
	texto.close();

	fstream lista;

	if(!capLeft.isOpened() || !capRight.isOpened())  // check if we succeeded
	{
		cerr << "ERROR: No se pudieron abrir la/s camaras." << endl;
		return -1;
	}

	namedWindow("CAMARA IZQUIERDA",1);
	namedWindow("CAMARA DERECHA",1);

	cout << "Puedes presionar la letra 'C' para realizar una captura de las camaras." << endl;

	while(key != 27)
	{
		bool isValid = true;

		Mat frameLeft;
		Mat frameRight;
		Mat imagenIzq;
		Mat imagenDer;

		try
		{
			capLeft >> frameLeft; // get a new frame from left camera
			capRight >> frameRight; //get a new frame from right camera
		}
		catch( Exception& e )
		{
			cout << "Se ha producido un error. Se ignora el marco. " << e.err << endl;
			isValid = false;
		}

		if (isValid)
		{
			try
			{
				imshow("CAMARA IZQUIERDA", frameLeft);
				imshow("CAMARA DERECHA", frameRight);

				/************************************************************
				 *    This is the place for all the cool stuff that you      *
				 *    want to do with your stereo images                     *
				 ************************************************************/

				//TODO:...

				if(key == 99)
				{
					lista.open("list.txt", fstream::in|fstream::out|fstream::ate);

					cout << "Captura imagen izquierda" << endl;
					imwrite("images/imgLeft_" + snapshotFilenameLeft + ".png", frameLeft);
					lista << "images/imgLeft_" + snapshotFilenameLeft + ".png" << endl;

					numSnapshotLeft++;
					snapshotFilenameLeft = static_cast<ostringstream*>(&(ostringstream() << numSnapshotLeft))->str();

					cout << "Captura imagen derecha" << endl;
					imwrite("images/imgRight_" + snapshotFilenameRight + ".png", frameRight);
					lista << "images/imgRight_" + snapshotFilenameRight + ".png" << endl;

					numSnapshotRight++;
					snapshotFilenameRight = static_cast<ostringstream*>(&(ostringstream() << numSnapshotRight))->str();

					lista.close();
				}

				if(key == 97)
				{
					Mat imgI = imread("imgLeft_0.png");
					Mat imgD = imread("imgRight_0.png");
					Mat Anaglifo;

					for(int i=0; i<imgI.rows; i++)
					{
						for(int j=0; j<imgI.cols; j++)
						{
							// You can now access the pixel value with cv::Vec3b 0->Azul; 1->Verde; 2->Rojo
							imgI.at<Vec3b>(i,j)[0] = 0;
							imgI.at<Vec3b>(i,j)[1] = 0;
						}
					}

					for(int i=0; i<imgD.rows; i++)
					{
						for(int j=0; j<imgD.cols; j++)
						{
							// You can now access the pixel value with cv::Vec3b
							imgD.at<Vec3b>(i,j)[2] = 0;
						}
					}

					namedWindow( "IMAGEN MODIFICADA IZQ",1);
					imshow( "IMAGEN MODIFICADA IZQ", imgI );

					namedWindow( "IMAGEN MODIFICADA DER",1);
					imshow( "IMAGEN MODIFICADA DER", imgD );

					double alpha = 0.5;
					double beta = 0.5;

					addWeighted( imgI, alpha, imgD, beta, 0.0, Anaglifo);

					namedWindow( "ANAGLIFO",1);
					imshow( "ANAGLIFO", Anaglifo );
				}

				if(key == 109)
				{
					int spatialRad = 10;			// mean shift parameters
					int colorRad = 10;
					int maxPyrLevel = 2;

					Mat imgLeftNormal = imread("imgLeft_0.png");
					Mat ImgLeftFilter;
					Mat imgRightNormal = imread("imgRight_0.png");
					Mat ImgRightFilter;

					pyrMeanShiftFiltering(imgLeftNormal, ImgLeftFilter, spatialRad, colorRad, maxPyrLevel );
					pyrMeanShiftFiltering(imgRightNormal, ImgRightFilter, spatialRad, colorRad, maxPyrLevel );

					namedWindow( "Mean Shift Filtro Izquierda",1);
					imshow( "Mean Shift Filtro Izquierda", ImgLeftFilter );
					namedWindow( "Mean Shift Filtro Derecha",1);
					imshow( "Mean Shift Filtro Derecha", ImgRightFilter );

				}

				if(key == 107)
				{
					// The camera properties
					int w = 640;
					int h = 480;
					int fps = 20;

					// The chessboard properties
					//CvSize chessboardSize(9,6);

					CvSize chessboardSize = cvSize(9,6);
					float squareSize = 1.0f;

					// This should contain the physical location of each corner, but since we don't know them, we are assigning constant positions
					vector<vector<Point3f> > objPoints;
					// The chessboard corner points in the images
					vector<vector<Point2f> > imagePoints1, imagePoints2;
					vector<Point2f> corners1, corners2;

					// The constant positions of each obj points
					vector<Point3f> obj;
					for (int y = 0; y < chessboardSize.height; y++) {
						for (int x = 0; x < chessboardSize.width; x++) {
							obj.push_back(Point3f(y * squareSize, x * squareSize, 0));
						}
					}
					/*for (int i = 0; i < chessboardSize.width * chessboardSize.height; i++) {
					     obj.push_back(Point3f(i / chessboardSize.width, i % chessboardSize.height, 0.0f));
					     }*/

					// The images, which are proceeded
					Mat img1, img2;

					// The grayscale versions of the images
					Mat gray1, gray2;

					// Get the image count
					int imageCount;
					cout << "How much images to load: " << endl;
					cin >> imageCount;

					// The image number of the current image (nullbased)
					int i = 0;
					// Whether the chessboard corners in the images were found
					bool found1 = false, found2 = false;

					while (i < imageCount)
					{
						// Load the images
						cout << "Attempting to load image pair " << i << endl;
						loadImagePair(img1, img2, i);
						cout << "Loaded image pair" << endl;

						// Convert to grayscale images
						cvtColor(img1, gray1, CV_BGR2GRAY);
						cvtColor(img2, gray2, CV_BGR2GRAY);

						// Find chessboard corners
						found1 = findChessboardCorners(img1, chessboardSize, corners1, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
						found2 = findChessboardCorners(img2, chessboardSize, corners2, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

						cout << "found 1/2: " << found1 << "/" << found2 << endl;

						// Find corners to subpixel accuracy
						if (found1) {
							cornerSubPix(gray1, corners1, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
							drawChessboardCorners(gray1, chessboardSize, corners1, found1);
						}
						if (found2) {
							cornerSubPix(gray2, corners2, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
							drawChessboardCorners(gray2, chessboardSize, corners2, found2);
						}

						// Store corners
						if (found1 && found2) {
							imagePoints1.push_back(corners1);
							imagePoints2.push_back(corners2);
							objPoints.push_back(obj);
							cout << "Corners stored" << endl;
							i++;
						}
						// Error
						else {
							cout << "Corners not found! Stopping" << endl;
							return 0;
						}
					}

					cout << "Starting calibration" << endl;
					Mat CM1 = Mat(3, 3, CV_64F);
					Mat CM2 = Mat(3, 3, CV_64F);
					Mat D1 = Mat(1, 5, CV_64F);
					Mat D2 = Mat(1, 5, CV_64F);
					Mat R = Mat(3, 3, CV_64F);
					Mat T = Mat(3, 1, CV_64F);
					Mat E = Mat(3, 3, CV_64F);
					Mat F = Mat(3, 3, CV_64F);
					//stereoCalibrate(objPoints, imagePoints1, imagePoints2, CM1, D1, CM2, D2, img1.size(), R, T, E, F,
					//CV_CALIB_SAME_FOCAL_LENGTH | CV_CALIB_ZERO_TANGENT_DIST, cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5));
					stereoCalibrate(objPoints, imagePoints1, imagePoints2, CM1, D1, CM2, D2, img1.size(), R, T, E, F,
							cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5), 0);
					cout << "Done calibration" << endl;

					cout << "Starting rectification" << endl;
					Mat R1 = Mat(3, 3, CV_64F);
					Mat R2 = Mat(3, 3, CV_64F);
					Mat P1 = Mat(3, 4, CV_64F);
					Mat P2 = Mat(3, 4, CV_64F);
					Mat Q = Mat(4, 4, CV_64F);
					stereoRectify(CM1, D1, CM2, D2, img1.size(), R, T, R1, R2, P1, P2, Q);
					cout << "Done rectification" << endl;

					cout << "Starting to store results" << endl;
					FileStorage fs("stereocalib.yml", FileStorage::WRITE);
					fs << "CM1" << CM1;
					fs << "CM2" << CM2;
					fs << "D1" << D1;
					fs << "D2" << D2;
					fs << "R" << R;
					fs << "T" << T;
					fs << "E" << E;
					fs << "F" << F;
					fs << "R1" << R1;
					fs << "R2" << R2;
					fs << "P1" << P1;
					fs << "P2" << P2;
					fs << "Q" << Q;
					fs.release();
					cout << "Done storing results" << endl;

					cout << "Starting to apply undistort" << endl;
					Mat map1x = Mat(img1.size().height, img1.size().width, CV_32F);
					Mat map1y = Mat(img1.size().height, img1.size().width, CV_32F);
					Mat map2x = Mat(img2.size().height, img2.size().width, CV_32F);
					Mat map2y = Mat(img2.size().height, img2.size().width, CV_32F);
					initUndistortRectifyMap(CM1, D1, R1, P1, img1.size(), CV_32FC1, map1x, map1y);
					initUndistortRectifyMap(CM2, D2, R2, P2, img2.size(), CV_32FC1, map2x, map2y);
					cout << "Done applying undistort" << endl;

					// The rectified images
					Mat imgU1 = Mat(img1.size(), img1.type());
					Mat imgU2 = Mat(img2.size(), img2.type());

					// Show rectified images
					i = 0;
					while (i < imageCount) {

						// Load the images
						cout << "Attempting to load image pair " << i << endl;
						loadImagePair(img1, img2, i);
						cout << "Loaded image pair" << endl;
						i++;

						remap(img1, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
						remap(img2, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
						//remap(img1, imgU1, map1x, map1y, INTER_LINEAR, BORDER_DEFAULT);
						//remap(img2, imgU2, map2x, map2y, INTER_LINEAR, BORDER_DEFAULT);

						imshow("img1", img1);
						imshow("img2", img2);
						imshow("rec1", imgU1);
						imshow("rec2", imgU2);

						int key = waitKey(0);
						if (key == 'q') {
							break;
						}
					}
					cout << "Calibración y Rectificación Finalizada :D!" << endl;
				}

				if(key == 114)
				{
					cout << "Captura imagen izquierda" << endl;
					imwrite("imgLeft.png", frameLeft);

					cout << "Captura imagen derecha" << endl;
					imwrite("imgRight.png", frameRight);

					Mat CapturaIzq = imread("imgLeft.png");
					Mat CapturaDer = imread("imgRight.png");

					Mat CM1 = Mat(3, 3, CV_64F);
					Mat CM2 = Mat(3, 3, CV_64F);
					Mat D1 = Mat(1, 5, CV_64F);
					Mat D2 = Mat(1, 5, CV_64F);
					Mat R = Mat(3, 3, CV_64F);
					Mat T = Mat(3, 1, CV_64F);
					Mat E = Mat(3, 3, CV_64F);
					Mat F = Mat(3, 3, CV_64F);
					Mat R1 = Mat(3, 3, CV_64F);
					Mat R2 = Mat(3, 3, CV_64F);
					Mat P1 = Mat(3, 4, CV_64F);
					Mat P2 = Mat(3, 4, CV_64F);
					Mat Q = Mat(4, 4, CV_64F);

					FileStorage fs2("stereocalib.yml", FileStorage::READ);
					fs2["CM1"] >> CM1;
					fs2["CM2"] >> CM2;
					fs2["D1"] >> D1;
					fs2["D2"] >> D2;
					fs2["R"] >> R;
					fs2["T"] >> T;
					fs2["E"] >> E;
					fs2["F"] >> F;
					fs2["R1"] >> R1;
					fs2["R2"] >> R2;
					fs2["P1"] >> P1;
					fs2["P2"] >> P2;
					fs2["Q"] >> Q;
					fs2.release();

					cout << "Starting to apply undistort" << endl;
					Mat map1x = Mat(CapturaIzq.size().height, CapturaIzq.size().width, CV_32F);
					Mat map1y = Mat(CapturaIzq.size().height, CapturaIzq.size().width, CV_32F);
					Mat map2x = Mat(CapturaDer.size().height, CapturaDer.size().width, CV_32F);
					Mat map2y = Mat(CapturaDer.size().height, CapturaDer.size().width, CV_32F);
					initUndistortRectifyMap(CM1, D1, R1, P1, CapturaIzq.size(), CV_32FC1, map1x, map1y);
					initUndistortRectifyMap(CM2, D2, R2, P2, CapturaDer.size(), CV_32FC1, map2x, map2y);
					cout << "Done applying undistort" << endl;

					Mat imgUIzq = Mat(CapturaIzq.size(), CapturaIzq.type());
					Mat imgUDer = Mat(CapturaDer.size(), CapturaDer.type());

					remap(CapturaIzq, imgUIzq, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
					remap(CapturaDer, imgUDer, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());

					namedWindow("Imagen Izquierda Rectificada", 1);
					imshow("Imagen Izquierda Rectificada", imgUIzq);

					cout << "Imagen Izquierda Rectificada ¡Guardada!" << endl;
					imwrite("imgLeftRect.png", imgUIzq);

					namedWindow("Imagen Derecha Rectificada", 1);
					imshow("Imagen Derecha Rectificada", imgUDer);

					cout << "Imagen Derecha Rectificada ¡Guardada!" << endl;
					imwrite("imgRightRect.png", imgUDer);

//					Mat opencv_disparity(imgUIzq.size().height, imgUIzq.size().width, CV_16S, Scalar(0));
//					Mat opencv_disparity_image(imgUIzq.size().height, imgUIzq.size().width, CV_8U, Scalar(0));
//
//					StereoBM stereo;
//
//					stereo.state->preFilterType = 1;
//					stereo.state->preFilterSize = 41;
//					stereo.state->preFilterCap = 31;
//					stereo.state->SADWindowSize = 31;
//					stereo.state->minDisparity = -65;
//					stereo.state->numberOfDisparities = 128;
//					stereo.state->textureThreshold = 10;
//					stereo.state->uniquenessRatio = 15;

//					while(1)
//					{
//						cvtColor(imgUIzq, imgUIzq, CV_BGR2GRAY);
//						cvtColor(imgUDer, imgUDer, CV_BGR2GRAY);
//
//						Mat disparity_image;
//
//						stereo(imgUIzq, imgUDer, opencv_disparity);
//
//						//opencv_disparity.convertTo(opencv_disparity_image, CV_8U);
//
//						imshow("disparity_image StereoBM", opencv_disparity);
//						cvWaitKey(15);
//
//					}


//					Mat imagen11, imagen22;
//
//					Mat opencv_disparity2(imagen1.size().height, imagen1.size().width, CV_16S, Scalar(0));
//					Mat opencv_disparity_image2(imagen1.size().height, imagen1.size().width, CV_8U, Scalar(0));
//					Mat disparityBMNorm;
//
//					StereoBM stereo2;
//
//					stereo2.state->preFilterType = 1;
//					stereo2.state->preFilterSize = 41;
//					stereo2.state->preFilterCap = 31;
//					stereo2.state->SADWindowSize = 31;
//					stereo2.state->minDisparity = -105;
//					stereo2.state->numberOfDisparities = 128;
//					stereo2.state->textureThreshold = 10;
//					stereo2.state->uniquenessRatio = 15;
//
//					cvtColor(imagen1, imagen11, CV_BGR2GRAY);
//					cvtColor(imagen2, imagen22, CV_BGR2GRAY);
//
//					//Mat disparity_image2;
//
//					stereo(imagen11, imagen22, opencv_disparity2);
//
//					//opencv_disparity.convertTo(opencv_disparity_image, CV_8U);
//
//					normalize(opencv_disparity2, opencv_disparity_image2, 0, 255, CV_MINMAX, CV_8U);
//
//					imshow("disparity_image StereoBM 2", opencv_disparity_image2);



//					Mat resultCurrentFrameCPU, resultNorCurrentFrameCPU;
//
//					StereoSGBM sbm;
//
//					sbm.SADWindowSize = 3;
//					sbm.numberOfDisparities = 144;
//					sbm.preFilterCap = 63;
//					sbm.minDisparity = -39;
//					sbm.uniquenessRatio = 10;
//					sbm.speckleWindowSize = 100;
//					sbm.speckleRange = 32;
//					sbm.disp12MaxDiff = 1;
//					sbm.fullDP = false;
//					sbm.P1 = 216;
//					sbm.P2 = 864;
//
//					sbm(imagen1, imagen2, resultCurrentFrameCPU);
//
//					normalize(resultCurrentFrameCPU, resultNorCurrentFrameCPU, 0, 255, CV_MINMAX, CV_8U);
//
//					imshow("disparity_image StereoSGBM", resultNorCurrentFrameCPU);


//					Mat* pair;
//					Mat part;
//
//					pair = CreateMat( imgUIzq.height*2, imgUIzq.width, CV_8UC3 );
//
//                    cvGetCols( pair, &part, 0, imgUIzq.width );
//                    cvCvtColor( img1r, &part, CV_GRAY2BGR );
//                    cvGetCols( pair, &part, imgUIzq.width, imgUIzq.width*2 );
//                    cvCvtColor( img2r, &part, CV_GRAY2BGR );
//
//
//					for( j = 0; j < imgUIzq.width; j += 16 )
//						cvLine( pair, Point(j,0), Point(j,imgUIzq.height*2), CV_RGB(0,255,0));
//
//					cvShowImage( "rectified", pair );

				}

				if(key == 98)
				{
					Mat imagenIzq = imread("imagen1.png");
					Mat imagenDer = imread("imagen2.png");

					correspondenciaBM(imagenIzq, imagenDer);
				}

				if(key == 115)
				{
					Mat imagenIzq = imread("imagen1.png");
					Mat imagenDer = imread("imagen2.png");

					correspondenciaSGBM(imagenIzq, imagenDer);
				}

			}

			catch( Exception& e )
			{
				/************************************************************
				 *    Sometimes an "Unrecognized or unsuported array type"   *
				 *    exception is received so we handle it to avoid dying   *
				 ************************************************************/
				cout << "An exception occurred. Ignoring frame. " << e.err << endl;
			}
		}

		key = waitKey(20);

	}

	return 0;
}


