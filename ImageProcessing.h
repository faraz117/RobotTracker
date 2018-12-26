#pragma once
#include"Blob.hpp"
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/videoio.hpp>
#include<iostream>
#include<conio.h>
#include<filesystem>

class ImageProcessing {

public:
	void process_image_colored(cv::Mat& frame);
	void process_image(cv::Mat& imgFrame1, cv::Mat& imgFrame2);
	std::vector<Blob> get_blob(cv::Mat& imgThresh);
	void createControlBar();
	void arrow_detection(cv::Mat src);
	void plot_detections(std::vector<Blob> blobs, cv::Mat& originalImage);
private:
	const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
	const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
	const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
	const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
	const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

	int iLowH = 58;
	int iHighH = 141;
	int iLowS = 80;
	int iHighS = 145;
	int iLowV = 216;
	int iHighV = 255;

};
