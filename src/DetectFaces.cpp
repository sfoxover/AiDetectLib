#include "defines.h"
#include "helpers.h"
#include "DetectFaces.h"
#include "DetectFaceFactory.h"
#include <filesystem>
#ifndef WIN32
#include <unistd.h>
#else
#include "windows.h"
#endif
#include <iostream>

// #define CAFFE

CDetectFaces::CDetectFaces()
{
	SetDetectMethod("none");
	SetExitingFlag(false);
	_addRectToFace = true;
	_imagesPerSecond = 0;
}

CDetectFaces::~CDetectFaces()
{
	std::wstring error;
	bool bOK = Stop(error);
	assert(bOK);
}

// Set initial settings
bool CDetectFaces::Start(std::string method, std::function<void(cv::Mat)> callback, std::wstring &error)
{
	_detectedCallback = callback;

	// Supported methods are OpenCV, Dnn, Hog, Mod and none
	SetDetectMethod(method);

	// Stop image processing thread
	bool bOK = Stop(error);
	assert(bOK);

	// Create new image processing thread 
	if (method != "Off")
	{
		SetExitingFlag(false);
		if (!_detectThread || !_detectThread->joinable())
		{
			_detectThread = std::make_unique<std::thread>(std::thread(&CDetectFaces::DetectFacesThread, this));
		}
	}

	return true;
}

// Stop face detection thread
bool CDetectFaces::Stop(std::wstring &error)
{
	SetExitingFlag(true);
	_imageQueueWait.notify_all();

	if (_detectThread && _detectThread->joinable())
	{
		_detectThread->join();
	}
	return true;
}

// Get set for _detectMethod
void CDetectFaces::GetDetectMethod(std::string&value)
{
	_detectMethodLock.lock();
	value = _detectMethod;
	_detectMethodLock.unlock();
}

void CDetectFaces::SetDetectMethod(std::string value)
{
	_detectMethodLock.lock();
	_detectMethod = value;
	_detectMethodLock.unlock();
}

// Get set for _addRectToFace
void CDetectFaces::GetAddRectToFace(bool &value)
{
	_addRectToFaceLock.lock();
	value = _addRectToFace;
	_addRectToFaceLock.unlock();
}

void CDetectFaces::SetAddRectToFace(bool value)
{
	_addRectToFaceLock.lock();
	_addRectToFace = value;
	_addRectToFaceLock.unlock();
}

// Get set for _exitingFlag
void CDetectFaces::GetExitingFlag(bool &value)
{
	_exitingFlagLock.lock();
	value = _exitingFlag;
	_exitingFlagLock.unlock();
}

void CDetectFaces::SetExitingFlag(bool value)
{
	_exitingFlagLock.lock();
	_exitingFlag = value;
	_exitingFlagLock.unlock();
}

// Get set for _imagesPerSecond
void CDetectFaces::GetImagesPerSecond(int &value)
{
	_imagesPerSecondLock.lock();
	value = _imagesPerSecond;
	_imagesPerSecondLock.unlock();
}

void CDetectFaces::SetImagesPerSecond(int value)
{
	_imagesPerSecondLock.lock();
	_imagesPerSecond = value;
	_imagesPerSecondLock.unlock();
}

// Add new image to queue
bool CDetectFaces::AddImageToQueue(cv::Mat image)
{
	bool imageQueued = false;
	_imageQueueLock.lock();
	if (_imageQueue.size() <= MAX_QUEUE_SIZE)
	{
		cv::Mat imgCopy = image.clone();
		_imageQueue.push_back(image);
		imageQueued = true;
		_imageQueueWait.notify_all();
	}
	_imageQueueLock.unlock();

	return imageQueued;
}

void CDetectFaces::DetectFacesThread(CDetectFaces *pThis)
{
	assert(pThis);
	std::string method;
	pThis->GetDetectMethod(method);
	auto spFaceDetectAi = DetectFaceFactory::Create(method);

	// Initialize AI library
	std::wstring error;
	bool bOK = spFaceDetectAi->Initialize(error);
	assert(bOK);

	bool exiting = false;
	int imagesProcessed = 0;
	auto start = std::chrono::system_clock::now();
	do
	{
		bool hasImage = false;
		pThis->GetExitingFlag(exiting);
		if (!exiting)
		{
			// Load image from front of queue
			cv::Mat image;
			pThis->_imageQueueLock.lock();
			hasImage = !pThis->_imageQueue.empty();
			if (hasImage)
			{
				image = pThis->_imageQueue[0];
				pThis->_imageQueue.erase(pThis->_imageQueue.begin(), pThis->_imageQueue.begin() + 1);
			}
			pThis->_imageQueueLock.unlock();

			if (hasImage)
			{
				bool addRectToImage;
				pThis->GetAddRectToFace(addRectToImage);
				bool bFoundFace = bFoundFace = spFaceDetectAi->DetectFace(image, addRectToImage, error);
				if (bFoundFace)
				{
					pThis->_detectedCallback(image);
				}
				imagesProcessed++;

				// Set processed count every 1 second
				auto end = std::chrono::system_clock::now();
				std::chrono::duration<double> secs = end - start;
				if (secs.count() >= 1)
				{
					pThis->SetImagesPerSecond(imagesProcessed);
					imagesProcessed = 0;
					start = std::chrono::system_clock::now();
				}
			}
			else
			{
				std::unique_lock<std::mutex> lock(pThis->_imageQueueWaitLock);
				pThis->_imageQueueWait.wait(lock);
			}
		}
	} while (!exiting);
}


