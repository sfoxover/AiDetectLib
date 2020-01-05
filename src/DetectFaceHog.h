#pragma once
/*
	CDetectFaceHog - Hog implementation of face detection methods
*/
#include "IDetectFace.h"
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

class CDetectFaceHog : public IDetectFace
{
private:
	dlib::frontal_face_detector _hogFaceDetector;

public:

	// Return enum for this AI detect method
	DetectMethods GetMethod()
	{
		return IDetectFace::DetectMethods::Hog;
	}

	// Set initial settings
	bool Initialize(std::wstring& error);

	// Look for a face in image
	bool DetectFace(cv::Mat& image, bool addRectToFace, std::wstring& error);
};