#pragma once
/*
	DetectFace - abstract class for face detection methods
*/
#include <mutex>
#include <opencv2/opencv.hpp>

class IDetectFace
{
public:

	// Detection method
	enum class DetectMethods 
	{
		none,
		OpenCV,
		Dnn,
		Hog,
		Mod
	};

	virtual DetectMethods GetMethod() = 0;

	// Set initial settings
	virtual bool Initialize(std::wstring& error) = 0;

	// Look for a face in image
	virtual bool DetectFace(cv::Mat& image, bool addRectToFace, std::wstring& error) = 0;
};