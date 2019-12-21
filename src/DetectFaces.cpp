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
	_detectMethod = none;
}

CDetectFaces::~CDetectFaces()
{
}

// Set initial settings
void CDetectFaces::Initialize(std::string method)
{
	// Supported methods are OpenCV, Dnn, Hog, Mod
	if (method == "OpenCV")
	{
		_detectMethod = OpenCV;
		InitOpenCV();
	}
	else if (method == "Dnn")
	{
		_detectMethod = Dnn;
		InitDNN();
	}
	else if (method == "Hog")
	{
		_detectMethod = Hog;
		InitHog();
	}
	else if (method == "Mod")
	{
		_detectMethod = Mod;
		InitMod();
	}
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

void CDetectFaces::DetectFaces(cv::Mat image, bool addRectToFace)
{
	std::wstring error;
	bool bOK = true;
	switch (_detectMethod)
	{
	case OpenCV:
		bOK = DetectFaceOpenCV(image, addRectToFace, error);
		break;
	case Dnn:
		bOK = DetectFaceDNN(image, addRectToFace, error);
		break;
	case Hog:
		bOK = DetectFaceDlibHog(image, addRectToFace, error);
		break;
	case Mod:
		bOK = DetectFaceDlibMod(image, addRectToFace, error);
		break;
	default:
		assert(false);
		break;
	}
	if(!bOK)
	{
		std::cerr << "DetectFaces failed, " << error.c_str() << std::endl;
	}
	assert(bOK);
}

// Look for a face in image
bool CDetectFaces::DetectFaceOpenCV(cv::Mat image, bool addRectToFace, std::wstring &error)
{
	bool bOK = false;
	if (!_faceFront.empty())
	{
		bOK = true;
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
	}
	return bOK;
}

// Look for a face in image
bool CDetectFaces::DetectFaceDNN(cv::Mat image, bool addRectToFace, std::wstring &error)
{
	bool bOK = false;
	const cv::Scalar meanVal(104.0, 177.0, 123.0);
	const float confidenceThreshold = 0.7;

	if (!_networkFace.empty())
	{
		bOK = true;
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
			}
		}
	}
	return bOK;
}

bool CDetectFaces::DetectFaceDlibHog(cv::Mat image, bool addRectToFace, std::wstring &error)
{
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
	return true;
}

bool CDetectFaces::DetectFaceDlibMod(cv::Mat image, bool addRectToFace, std::wstring &error)
{
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
	return true;
}
