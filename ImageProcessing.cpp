#include "ImageProcessing.h"

void ImageProcessing::createControlBar() {
	//Create trackbars in "Control" window
	cv::namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"
	cvCreateTrackbar("LowH", "Control", &iLowH, 255); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Control", &iHighH, 255);

	cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &iHighS, 255);

	cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Control", &iHighV, 255);
}

void ImageProcessing::arrow_detection(cv::Mat src) {

	if (src.empty())
	{
		std::cout << "cannot open " << std::endl;
		return;
	}
	std::vector<Blob> blobs;
	std::vector<std::vector<cv::Point> > contours;

	cv::findContours(src, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat imgContours(src.size(), CV_8UC3, SCALAR_BLACK);

	cv::drawContours(imgContours, contours, -1, SCALAR_WHITE, -1);


	std::vector<std::vector<cv::Point> > convexHulls(contours.size());

	for (unsigned int i = 0; i < contours.size(); i++) {
		cv::convexHull(contours[i], convexHulls[i]);
	}

	for (auto &convexHull : convexHulls) {
		Blob possibleBlob(convexHull);

		if (possibleBlob.boundingRect.area() > 10 && possibleBlob.boundingRect.width > 10) {
			blobs.push_back(possibleBlob);
		}
	}

	cv::Mat imgConvexHulls(src.size(), CV_8UC3, SCALAR_BLACK);

	convexHulls.clear();

	for (auto &blob : blobs) {
		convexHulls.push_back(blob.contour);
	}

	cv::drawContours(imgConvexHulls, convexHulls, -1, SCALAR_WHITE, -1);
	imshow("Convex Hulls", imgConvexHulls);
	cv::Mat dst, cdst;
	cv::Canny(src, dst, 50, 200, 3);
	cv::cvtColor(dst, cdst, CV_GRAY2BGR);


	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(dst, lines, 1, CV_PI / 180, 50, 50, 10);
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];
		
		cv::line(cdst, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 3, CV_AA);
	}
	cv::imshow("detected lines", cdst);
}
void ImageProcessing::process_image_colored(cv::Mat& frame) {
	cv::Mat imgThresholdColored;
	cv::Mat imgFrame1Copy = frame.clone();
	cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	cv::Mat structuringElement9x9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));


	//cv::cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2HSV);
	cv::inRange(imgFrame1Copy, cv::Scalar(iLowH, iLowS, iLowV), cv::Scalar(iHighH, iHighS, iHighV), imgThresholdColored);
	imshow("Thresholded Image", imgThresholdColored);
	cv::GaussianBlur(imgThresholdColored, imgThresholdColored, cv::Size(5, 5), 0);
	cv::dilate(imgThresholdColored, imgThresholdColored, structuringElement5x5);
	cv::dilate(imgThresholdColored, imgThresholdColored, structuringElement5x5);
	cv::erode(imgThresholdColored, imgThresholdColored, structuringElement5x5);
	arrow_detection(imgThresholdColored);
	//plot_detections(get_blob(imgThresholdColored) , imgFrame1Copy);
	// Get the best robot1
	// Get the best Robot2
	// Get the best Object
	// Get the target
	
}
void ImageProcessing::process_image(cv::Mat& imgFrame1, cv::Mat& imgFrame2) {
	cv::Mat imgFrame1Copy = imgFrame1.clone();
	cv::Mat imgFrame2Copy = imgFrame2.clone();
	cv::Mat imgDifference;
	cv::Mat imgThresh;

	cv::cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
	cv::cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);
	cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
	cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);

	cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

	//cv::imshow("imgDifference", imgDifference);

	cv::threshold(imgDifference, imgThresh, 20, 255.0, CV_THRESH_BINARY);
	//plot_detections(imgThresh,imgFrame2Copy);
	//cv::imshow("imgThresh", imgThresh);
	imgFrame1 = imgFrame2.clone(); // move frame 1 up to where frame 2 is
}
std::vector<Blob> ImageProcessing::get_blob(cv::Mat& imgThresh) {
	std::vector<Blob> blobs;

	cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	cv::Mat structuringElement9x9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

	cv::dilate(imgThresh, imgThresh, structuringElement5x5);
	cv::dilate(imgThresh, imgThresh, structuringElement5x5);
	cv::erode(imgThresh, imgThresh, structuringElement5x5);

	cv::Mat imgThreshCopy = imgThresh.clone();

	std::vector<std::vector<cv::Point> > contours;

	cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat imgContours(imgThresh.size(), CV_8UC3, SCALAR_BLACK);

	cv::drawContours(imgContours, contours, -1, SCALAR_WHITE, -1);


	std::vector<std::vector<cv::Point> > convexHulls(contours.size());

	for (unsigned int i = 0; i < contours.size(); i++) {
		cv::convexHull(contours[i], convexHulls[i]);
	}

	for (auto &convexHull : convexHulls) {
		Blob possibleBlob(convexHull);

		if (possibleBlob.boundingRect.area() > 10 && possibleBlob.boundingRect.width > 10) {
			blobs.push_back(possibleBlob);
		}
	}

	cv::Mat imgConvexHulls(imgThresh.size(), CV_8UC3, SCALAR_BLACK);

	convexHulls.clear();

	for (auto &blob : blobs) {
		convexHulls.push_back(blob.contour);
	}

	cv::drawContours(imgConvexHulls, convexHulls, -1, SCALAR_WHITE, -1);
	imshow("Convex Hulls", imgConvexHulls);
	return blobs;
}


void ImageProcessing::plot_detections(std::vector<Blob> blobs , cv::Mat& originalImage) {
 
	auto index = std::distance(blobs.begin(), std::max_element(blobs.begin(), blobs.end(), [](const Blob a, const Blob b) {return a.dblDiagonalSize < b.dblDiagonalSize; }));
	cv::rectangle(originalImage, blobs[index].boundingRect, SCALAR_RED, 2);             // draw a red box around the blob
	cv::circle(originalImage, blobs[index].centerPosition, 3, SCALAR_GREEN, -1);        // draw a filled-in green circle at the center
	cv::imshow("Detections", originalImage);
}