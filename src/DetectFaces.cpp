#include "defines.h"
#include "helpers.h"
#include "DetectFaces.h"
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
	SetDetectMethod(none);
	_addRectToFace = true;
}

CDetectFaces::~CDetectFaces()
{
	std::wstring error;
	bool bOK = Stop(error);
	assert(bOK);
}

// Set initial settings
bool CDetectFaces::Initialize(std::string method, std::function<void(cv::Mat)> callback, std::wstring &error)
{
	_detectedCallback = callback;

	// Supported methods are OpenCV, Dnn, Hog, Mod
	if (method == "OpenCV")
	{
		InitOpenCV();
		SetDetectMethod(OpenCV);
	}
	else if (method == "Dnn")
	{
		InitDNN();
		SetDetectMethod(Dnn);
	}
	else if (method == "Hog")
	{
		InitHog();
		SetDetectMethod(Hog);
	}
	else if (method == "Mod")
	{
		InitMod();
		SetDetectMethod(Mod);
	}
	else if (method == "Off")
	{
		InitMod();
		SetDetectMethod(none);
	}
	else
	{
		std::wstringstream szErr;
		szErr << L"Error, DetectFaces method is unsupported. Method = " << Helpers::Utf8ToWide(method) << ".";
		error = szErr.str();
		return false;
	}

	// Start thread to process images
	if (method != "Off" && !_detectThread.joinable())
	{
		_signalNewImage = std::make_unique<std::promise<void>>();
		auto futureObj = _signalNewImage->get_future();
		_detectThread = std::thread(&CDetectFaces::DetectFacesThread, this, std::move(futureObj));
	}

	return true;
}

// Stop face detection thread
bool CDetectFaces::Stop(std::wstring &error)
{
	SetExitingFlag(true);
	if (_signalNewImage)
	{
		_signalNewImage->set_value();
		_signalNewImage.reset(nullptr);
	}

	if (_detectThread.joinable())
		_detectThread.join();
	return true;
}

// Get set for _detectMethod
void CDetectFaces::GetDetectMethod(DetectMethods &value)
{
	_detectMethodLock.lock();
	value = _detectMethod;
	_detectMethodLock.unlock();
}

void CDetectFaces::SetDetectMethod(DetectMethods value)
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

// Return face detect AI method as string
std::string CDetectFaces::GetDetectMethod()
{
	std::string result = "Unknown";
	DetectMethods method;
	GetDetectMethod(method);
	switch (method)
	{
	case OpenCV:
		result = "OpenCV";
		break;
	case Dnn:
		result = "Dnn";
		break;
	case Hog:
		result = "Hog";
		break;
	case Mod:
		result = "Mod";
		break;
	case none:
		result = "Off";
		break;
	default:
		assert(false);
		break;
	}
	return result;
}

void CDetectFaces::InitOpenCV()
{
	auto filePath = Helpers::AppendToRunPath(Helpers::AppendPath("assets", "haarcascade_frontalface_default.xml"));
	bool bOK = _faceFront.load(filePath);
	assert(bOK);
	filePath = Helpers::AppendToRunPath(Helpers::AppendPath("assets", "haarcascade_profileface.xml"));
	bOK = _faceProfile.load(filePath);
	assert(bOK);
}

void CDetectFaces::InitDNN()
{
	auto caffeConfigFile = Helpers::AppendToRunPath(Helpers::AppendPath("assets", "deploy.prototxt"));
	auto caffeWeightFile = Helpers::AppendToRunPath(Helpers::AppendPath("assets", "res10_300x300_ssd_iter_140000_fp16.caffemodel"));
	auto tensorflowConfigFile = Helpers::AppendToRunPath(Helpers::AppendPath("assets", "opencv_face_detector.pbtxt"));
	auto tensorflowWeightFile = Helpers::AppendToRunPath(Helpers::AppendPath("assets", "opencv_face_detector_uint8.pb"));
#ifdef CAFFE
	_networkFace = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
#else
	_networkFace = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
#endif
}

void CDetectFaces::InitHog()
{
	_hogFaceDetector = dlib::get_frontal_face_detector();
}

void CDetectFaces::InitMod()
{
	std::string mmodModelPath = Helpers::AppendToRunPath(Helpers::AppendPath("assets", "mmod_human_face_detector.dat"));
	dlib::deserialize(mmodModelPath) >> _mmodFaceDetector;
}

// Add new image to queue
bool CDetectFaces::AddImageToQueue(cv::Mat image)
{
	bool signal = false;
	_imageQueueLock.lock();
	if (_imageQueue.empty())
	{
		cv::Mat imgCopy = image;
		_imageQueue.push_back(imgCopy);
		signal = true;
	}
	_imageQueueLock.unlock();

	// Signal new image available
	if (signal && _signalNewImage)
	{
		_signalNewImage->set_value();
		_signalNewImage.reset(nullptr);
	}
	return signal;
}

void CDetectFaces::DetectFacesThread(CDetectFaces *pThis, std::future<void> futureObj)
{
	bool exiting = false;
	do
	{
		futureObj.wait();
		bool hasImage;
		do
		{
			hasImage = false;
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
					pThis->_imageQueue.erase(pThis->_imageQueue.begin());
				}
				pThis->_imageQueueLock.unlock();

				if (hasImage)
				{
					std::wstring error;
					DetectMethods method;
					pThis->GetDetectMethod(method);
					bool addRectToImage;
					pThis->GetAddRectToFace(addRectToImage);
					bool bFoundFace = false;
					switch (method)
					{
					case OpenCV:
						bFoundFace = pThis->DetectFaceOpenCV(image, addRectToImage, error);
						break;
					case Dnn:
						bFoundFace = pThis->DetectFaceDNN(image, addRectToImage, error);
						break;
					case Hog:
						bFoundFace = pThis->DetectFaceDlibHog(image, addRectToImage, error);
						break;
					case Mod:
						bFoundFace = pThis->DetectFaceDlibMod(image, addRectToImage, error);
						break;
					default:
						std::cerr << "DetectFaces unsupported method, " << method << std::endl;
						break;
					}
					if (bFoundFace)
					{
						pThis->_detectedCallback(image);
					}
				}
			}
		} while (hasImage);
	} while (!exiting);
}

// Look for a face in image
bool CDetectFaces::DetectFaceOpenCV(cv::Mat &image, bool addRectToFace, std::wstring &error)
{
	bool bFoundFace = false;
	if (!_faceFront.empty())
	{
		cv::Mat imgGrey;
		cv::cvtColor(image, imgGrey, cv::COLOR_BGR2GRAY);

		std::vector<cv::Rect> faces;
		_faceFront.detectMultiScale(image, faces, 1.3, 5);
		std::for_each(faces.begin(), faces.end(), [&](cv::Rect face) {
			cv::rectangle(image, face, (255, 255, 255), 2);
		});

		std::vector<cv::Rect> profiles;
		_faceProfile.detectMultiScale(image, profiles, 1.3, 5);
		std::for_each(profiles.begin(), profiles.end(), [&](cv::Rect profile) {
			cv::rectangle(image, profile, (255, 255, 255), 2);
		});

		bFoundFace = !faces.empty() || !profiles.empty();
	}
	return bFoundFace;
}

// Look for a face in image
bool CDetectFaces::DetectFaceDNN(cv::Mat &image, bool addRectToFace, std::wstring &error)
{
	bool bFoundFace = false;
	const cv::Scalar meanVal(104.0, 177.0, 123.0);
	const float confidenceThreshold = 0.7;

	if (!_networkFace.empty())
	{
#ifdef CAFFE
		cv::Mat inputBlob = cv::dnn::blobFromImage(image, 1.0, image.size(), meanVal, false, false);
#else
		cv::Mat inputBlob = cv::dnn::blobFromImage(image, 1.0, image.size(), meanVal, true, false);
#endif

		_networkFace.setInput(inputBlob, "data");
		cv::Mat detection = _networkFace.forward("detection_out");
		cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);
			if (confidence > confidenceThreshold)
			{
				int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
				int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
				int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
				int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);

				cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
				bFoundFace = true;
			}
		}
	}
	return bFoundFace;
}

bool CDetectFaces::DetectFaceDlibHog(cv::Mat &image, bool addRectToFace, std::wstring &error)
{
	bool bFoundFace = false;
	int inHeight = 300;
	int inWidth = 0;

	int frameHeight = image.rows;
	int frameWidth = image.cols;
	if (!inWidth)
		inWidth = (int)((frameWidth / (float)frameHeight) * inHeight);

	float scaleHeight = frameHeight / (float)inHeight;
	float scaleWidth = frameWidth / (float)inWidth;

	cv::Mat frameDlibHogSmall;
	cv::resize(image, frameDlibHogSmall, cv::Size(inWidth, inHeight));

	// Convert OpenCV image format to Dlib's image format
	dlib::cv_image<dlib::bgr_pixel> dlibIm(frameDlibHogSmall);

	// Detect faces in the image
	std::vector<dlib::rectangle> faceRects = _hogFaceDetector(dlibIm);

	for (size_t i = 0; i < faceRects.size(); i++)
	{
		int x1 = (int)(faceRects[i].left() * scaleWidth);
		int y1 = (int)(faceRects[i].top() * scaleHeight);
		int x2 = (int)(faceRects[i].right() * scaleWidth);
		int y2 = (int)(faceRects[i].bottom() * scaleHeight);
		cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), (int)(frameHeight / 150.0), 4);
	}
	bFoundFace = !faceRects.empty();
	return bFoundFace;
}

bool CDetectFaces::DetectFaceDlibMod(cv::Mat &image, bool addRectToFace, std::wstring &error)
{
	bool bFoundFace = false;
	int inHeight = 300;
	int inWidth = 0;

	int frameHeight = image.rows;
	int frameWidth = image.cols;
	if (!inWidth)
		inWidth = (int)((frameWidth / (float)frameHeight) * inHeight);

	float scaleHeight = frameHeight / (float)inHeight;
	float scaleWidth = frameWidth / (float)inWidth;

	cv::Mat imageSmall;
	cv::resize(image, imageSmall, cv::Size(inWidth, inHeight));

	// Convert OpenCV image format to Dlib's image format
	dlib::cv_image<dlib::bgr_pixel> dlibIm(imageSmall);
	dlib::matrix<dlib::rgb_pixel> dlibMatrix;
	dlib::assign_image(dlibMatrix, dlibIm);

	// Detect faces in the image
	std::vector<dlib::mmod_rect> faceRects = _mmodFaceDetector(dlibMatrix);

	for (size_t i = 0; i < faceRects.size(); i++)
	{
		int x1 = (int)(faceRects[i].rect.left() * scaleWidth);
		int y1 = (int)(faceRects[i].rect.top() * scaleHeight);
		int x2 = (int)(faceRects[i].rect.right() * scaleWidth);
		int y2 = (int)(faceRects[i].rect.bottom() * scaleHeight);
		cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), (int)(frameHeight / 150.0), 4);
	}
	bFoundFace = !faceRects.empty();
	return bFoundFace;
}
