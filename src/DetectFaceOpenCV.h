#pragma once
/*
	CDetectFaceOpenCV - OpenCV implementation of face detection methods
*/
#include "IDetectFace.h"
#include <dlib/opencv.h>

class CDetectFaceOpenCV : public IDetectFace
{
private:
	cv::CascadeClassifier _faceFront;
	cv::CascadeClassifier _faceProfile;

public:

	// Return enum for this AI detect method
	DetectMethods GetMethod()
	{
		return DetectMethods::OpenCV;
	}

	// Set initial settings
	bool Initialize(std::wstring& error);

	// Look for a face in image
	bool DetectFace(cv::Mat& image, bool addRectToFace, std::wstring& error);
};