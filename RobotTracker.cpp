// RobotTracker.cpp : Defines the entry point for the console application.
//
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/core/utility.hpp>
#include"ImageProcessing.h"

#define	CAMERA

void read_video(cv::Mat& frame, bool isCamera, cv::VideoCapture arenaVideo);

int main()
{
	char chCheckForEscKey = 0;
	// Video Processing Object
	ImageProcessing processor_instance;

	std::vector<cv::Rect2d> objects;
	// Read video file
	cv::Mat currentFrame;
	cv::Mat previousFrame;
	processor_instance.createControlBar();
#ifdef CAMERA
	cv::VideoCapture arenaVideo(0);
#else
	// Video File Object
	cv::VideoCapture arenaVideo;
	std::string file_path = std::filesystem::current_path().generic_string() + "/video.mp4";
	std::cout << "Current path is " << file_path << '\n';
	arenaVideo.open(file_path);
#endif // CAMERA
	read_video(currentFrame,true,arenaVideo);
	while (chCheckForEscKey != 27) {
		previousFrame = currentFrame.clone();
		read_video(currentFrame,true,arenaVideo);
		cv::imshow("source", currentFrame);
		processor_instance.process_image_colored(currentFrame);
		chCheckForEscKey = cv::waitKey(1);      // get key press in case user pressed esc

	}

	if (chCheckForEscKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
		cv::waitKey(0);                         // hold the windows open to allow the "end of video" message to show
	}

	//TODO: Output the image here
	//TODO: Run the state machine to get the two robots and objects
	//TODO: Save the points and keep changing
	// note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows
	return(0);
}

void read_video(cv::Mat& frame , bool isCamera , cv::VideoCapture arenaVideo) {
	cv::Mat currentFrame;
	// Camera Object
	if (isCamera) {
		if (arenaVideo.isOpened()) {
			arenaVideo >> currentFrame;
			cv::resize(currentFrame, currentFrame, cv::Size(), 0.75, 0.75);
			frame = currentFrame;
			return;
		}
	}
	else {
		if (!arenaVideo.isOpened()) {                                                 // if unable to open video file
			std::cout << "\nerror reading video file" << std::endl << std::endl;      // show error message
			_getch();                    // it may be necessary to change or remove this line if not using Windows
			return;                                                              // and exit program
		}
		if ((arenaVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < arenaVideo.get(CV_CAP_PROP_FRAME_COUNT)) {       // if there is at least one more frame
			arenaVideo.read(currentFrame);                            // read it
			cv::resize(currentFrame, currentFrame, cv::Size(), 0.50, 0.50);
			frame = currentFrame;
		}
	}
}
