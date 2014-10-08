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

static void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y,x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

void histograma(Mat mapadisp)
{

	// (2)allocate Mat to draw a histogram image
	const int ch_width = 260;
	const int sch = mapadisp.channels();
	Mat hist_img(Size(ch_width * sch, 200), CV_8UC3, Scalar::all(255));

	vector<MatND> hist(3);
	const int hist_size = 256;
	const int hdims[] = {hist_size};
	const float hranges[] = {0,256};
	const float* ranges[] = {hranges};
	double max_val = .0;

	if(sch==1) {
		// (3a)if the source image has single-channel, calculate its histogram
		calcHist(&mapadisp, 1, 0, Mat(), hist[0], 1, hdims, ranges, true, false);
		minMaxLoc(hist[0], 0, &max_val);
	} else {
		// (3b)if the souce image has multi-channel, calculate histogram of each plane
		for(int i=0; i<sch; ++i) {
			calcHist(&mapadisp, 1, &i, Mat(), hist[i], 1, hdims, ranges, true, false);
			double tmp_val;
			minMaxLoc(hist[i], 0, &tmp_val);
			max_val = max_val < tmp_val ? tmp_val : max_val;
		}
	}

	// (4)scale and draw the histogram(s)
	Scalar color = Scalar::all(100);
	for(int i=0; i<sch; i++) {
		if(sch==3)
			color = Scalar((0xaa<<i*8)&0x0000ff,(0xaa<<i*8)&0x00ff00,(0xaa<<i*8)&0xff0000, 0);
		hist[i].convertTo(hist[i], hist[i].type(), max_val?200./max_val:0.,0);
		for(int j=0; j<hist_size; ++j) {
			int bin_w = saturate_cast<int>((double)ch_width/hist_size);
			rectangle(hist_img,
					Point(j*bin_w+(i*ch_width), hist_img.rows),
					Point((j+1)*bin_w+(i*ch_width), hist_img.rows-saturate_cast<int>(hist[i].at<float>(j))),
					color, -1);
		}
	}
	// (5)show the histogram image, and quit when any key pressed
	namedWindow("Histograma", CV_WINDOW_AUTOSIZE);
	imshow("Histograma", hist_img);
}


void correspondenciaBIRCHFIELD(Mat &imagenIzq, Mat &imagenDer)
{
	IplImage _srcLeft = imagenIzq;
	IplImage _srcRight = imagenDer;
	IplImage* srcLeft = &_srcLeft;
	IplImage* srcRight = &_srcRight;
	IplImage* leftImage = cvCreateImage(cvGetSize(srcLeft), IPL_DEPTH_8U, 1);
	IplImage* rightImage = cvCreateImage(cvGetSize(srcRight), IPL_DEPTH_8U, 1);
	IplImage* depthImage = cvCreateImage(cvGetSize(srcRight), IPL_DEPTH_8U, 1);

	cvCvtColor(srcLeft, leftImage, CV_BGR2GRAY);
	cvCvtColor(srcRight, rightImage, CV_BGR2GRAY);

	cvFindStereoCorrespondence( leftImage, rightImage, CV_DISPARITY_BIRCHFIELD, depthImage, 50, 15, 3, 6, 8, 15);

	Mat MapaDipBIRCHFIELD(depthImage);

	imshow("Mapa de Disparidad BIRCHFIELD", MapaDipBIRCHFIELD);
	imwrite("MapaDisparidad.png", MapaDipBIRCHFIELD);

	cout << "Disparidad(1,1)=" << MapaDipBIRCHFIELD.at<double>(1,1) << endl;

	//...

	Mat Q = Mat(4, 4, CV_64F);

	FileStorage fs2("stereocalib.yml", FileStorage::READ);
	fs2["Q"] >> Q;
	fs2.release();

	double Q00, Q01, Q02, Q03, Q10, Q11, Q12, Q13, Q20, Q21, Q22, Q23, Q30, Q31, Q32, Q33;
	Q00 = Q.at<double>(0,0);
	Q01 = Q.at<double>(0,1);
	Q02 = Q.at<double>(0,2);
	Q03 = Q.at<double>(0,3);
	Q10 = Q.at<double>(1,0);
	Q11 = Q.at<double>(1,1);
	Q12 = Q.at<double>(1,2);
	Q13 = Q.at<double>(1,3);
	Q20 = Q.at<double>(2,0);
	Q21 = Q.at<double>(2,1);
	Q22 = Q.at<double>(2,2);
	Q23 = Q.at<double>(2,3);
	Q30 = Q.at<double>(3,0);
	Q31 = Q.at<double>(3,1);
	Q32 = Q.at<double>(3,2);
	Q33 = Q.at<double>(3,3);

	cout << "Q(0,0) = " << Q00 << " Q(0,1) = " << Q01 << " Q(0,2) = " << Q02 << " Q(0,3) = "<< Q03 <<
			" Q(1,0) = " << Q10 << " Q(1,1) = " << Q11 << " Q(1,2) = " << Q12 << " Q(1,3) = "<< Q13 <<
			" Q(2,0) = " << Q20 << " Q(2,1) = " << Q21 << " Q(2,2) = "<< Q22 << " Q(2,3) = " << Q23 <<
			" Q(3,0) = " << Q30 << " Q(3,1) = " << Q31 << " Q(3,2) = "<< Q32 <<" Q(3,3) = "<< Q33 << endl;

	histograma(MapaDipBIRCHFIELD);

//
//	  Mat recons3D(MapaDipBIRCHFIELD.size(), CV_32FC3);
//	  //Reproject image to 3D
//	  cout << "Reprojecting image to 3D..." << endl;
//	  reprojectImageTo3D( MapaDipBIRCHFIELD, recons3D, Q, false, CV_32F );
//	  imshow("algo xD", recons3D);
//
//	  	FileStorage storage("MapaDipBIRCHFIELD.yml", FileStorage::WRITE);
//	  	for(int i=0; i<MapaDipBIRCHFIELD.rows; i++)
//	  	{
//	  		for(int j=0; j<MapaDipBIRCHFIELD.cols; j++)
//	  		{
//	  			storage << "MapaDipBIRCHFIELD" << recons3D.at<double>(i,j);
//	  		}
//	  	}
//	  	storage.release();
}

void correspondenciaBM(Mat &imagenIzq, Mat &imagenDer)
{
	Mat imagenIzq1CH, imagenDer1CH, g_disp;

	Mat MapaDispBM(imagenIzq.size().height, imagenIzq.size().width, CV_16S, Scalar(0));
	Mat MapaDispBM_Norm(imagenIzq.size().height, imagenIzq.size().width, CV_8U, Scalar(0));

	StereoBM stereoBM;

	stereoBM.state->preFilterType = 1;
	stereoBM.state->preFilterSize = 9;
	stereoBM.state->preFilterCap = 32;
	stereoBM.state->SADWindowSize = 9;
	stereoBM.state->minDisparity = 0;
	stereoBM.state->numberOfDisparities = 32;
	stereoBM.state->textureThreshold = 0;
	stereoBM.state->uniquenessRatio = 0;

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

	cout << "Coordenadas en Mapa de Disparidad 8U = " << MapaDispSGBM_Norm.at<Vec3f>(152,82) << endl;

	circle(MapaDispSGBM_Norm, Point(152,82), 5, Scalar(255,0,0), 1);
	//circle(MapaDispSGBM_Norm, Point(383,466), 5, Scalar(150,0,0), 1);
	//circle(MapaDispSGBM_Norm, Point(400,281), 5, Scalar(200,0,0), 1);
	//circle(MapaDispSGBM_Norm, Point(99,476), 5, Scalar(210,0,0), 1);

	Mat MapaDispSGBM_Norm_32F;
	MapaDispSGBM_Norm.convertTo(MapaDispSGBM_Norm_32F, CV_32F); //Convertir mapa de disparidad en 32 bits

	cout << "Coordenadas en Mapa de Disparidad 32F = " << MapaDispSGBM_Norm_32F.at<Vec3f>(152,82) << endl;

	imshow("Mapa de Disparidad StereoSGBM", MapaDispSGBM_Norm);

	//cout << "Disparidad(152,82)=" << MapaDispSGBM_Norm.at<double>(152,82) << endl;
	//cout << "Disparidad(400,281)=" << MapaDispSGBM_Norm.at<double>(400,281) << endl;
	//cout << "Disparidad(383,466)=" << MapaDispSGBM_Norm.at<double>(383,466) << endl;
	//cout << "Disparidad(99,476)=" << MapaDispSGBM_Norm.at<double>(99,476) << endl;

	Mat Q = Mat(4, 4, CV_64F);
	FileStorage fs2("stereocalib.yml", FileStorage::READ);
	fs2["Q"] >> Q;
	fs2.release();

	Mat recons3D(MapaDispSGBM_Norm_32F.size(), CV_32FC3);
	//Reproject image to 3D
	cout << "Reprojecting image to 3D..." << endl;
	reprojectImageTo3D(MapaDispSGBM_Norm_32F, recons3D, Q, false, CV_32F);
	//imshow("algo xD", recons3D);

	cout << recons3D.at<Vec3f>(152,82) << endl;

	//saveXYZ("Hola", recons3D);
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
					//int w = 640;
					//int h = 480;
					//int fps = 20;

					// The chessboard properties
					//CvSize chessboardSize(9,6);

					CvSize chessboardSize = cvSize(9,6);
					float squareSize = 2.5f;

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
				}

				if(key == 98)
				{
					Mat imagenIzq = imread("imgLeftRect.png");
					Mat imagenDer = imread("imgRightRect.png");

					correspondenciaBM(imagenIzq, imagenDer);
				}

				if(key == 115)
				{
					Mat imagenIzq = imread("imgLeftRect.png");
					Mat imagenDer = imread("imgRightRect.png");

					correspondenciaSGBM(imagenIzq, imagenDer);
				}

				if(key == 116)
				{
					Mat imagenIzq = imread("imagen1.png");
					Mat imagenDer = imread("imagen2.png");

					correspondenciaBIRCHFIELD(imagenIzq, imagenDer);
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


