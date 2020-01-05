#pragma once
/*
	CDetectFaces - Scan an OpenCV Mat frame with DetectFace implementation and draw rectangles around faces detected.
*/
#include "IDetectFace.h"
#include <mutex>
#include <thread>
#include <condition_variable>
#include <functional>
#include <dlib/opencv.h>

#define MAX_QUEUE_SIZE 1

class CDetectFaces
{
private:
	CDetectFaces();
	~CDetectFaces();

private:
// Properties

	// Current selected AI method
	std::string _detectMethod;
	std::mutex _detectMethodLock;

	// Flag to add rectangle to image around detected face
	bool _addRectToFace;
	std::mutex _addRectToFaceLock;

	// Queue of cv::Mat images to run detection on
	std::vector<cv::Mat> _imageQueue;
	std::mutex _imageQueueLock;

	// Images processed per second
	int _imagesPerSecond;
	std::mutex _imagesPerSecondLock;

	// Wait for this signal if image queue is empty
	std::condition_variable _imageQueueWait;
	std::mutex _imageQueueWaitLock;

	// Exit thread flag
	bool _exitingFlag;
	std::mutex _exitingFlagLock;

	// Detect face thread
	std::unique_ptr<std::thread> _detectThread;

	// Callback method to pass detected face images to
	std::function<void(cv::Mat)> _detectedCallback;

public:
// Methods
	static CDetectFaces &Instance()
	{
		static CDetectFaces instance;
		return instance;
	}

	// Set initial settings
	bool Start(std::string method, std::function<void(cv::Mat)> callback, std::wstring &error);

	// Stop face detection thread
	bool Stop(std::wstring &error);

	// Get set for _detectMethod
	void GetDetectMethod(std::string&value);
	void SetDetectMethod(std::string value);

	// Get set for _addRectToFace
	void GetAddRectToFace(bool &value);
	void SetAddRectToFace(bool value);

	// Get set for _exitingFlag
	void GetExitingFlag(bool &value);
	void SetExitingFlag(bool value);

	// Get set for _imagesPerSecond
	void GetImagesPerSecond(int &value);
	void SetImagesPerSecond(int value);

	// Add new image to queue
	bool AddImageToQueue(cv::Mat image);

	// Look for a face in image on separate thread
	static void DetectFacesThread(CDetectFaces *pThis);
};