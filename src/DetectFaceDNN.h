#pragma once
/*
	CDetectFaceDNN - DNN implementation of face detection methods
*/
#include "IDetectFace.h"
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

class CDetectFaceDNN : public IDetectFace
{
private:
	cv::dnn::Net _networkFace;

public:

	// Return enum for this AI detect method
	DetectMethods GetMethod()
	{
		return IDetectFace::DetectMethods::Dnn;
	}

	// Set initial settings
	bool Initialize(std::wstring& error);

	// Look for a face in image
	bool DetectFace(cv::Mat& image, bool addRectToFace, std::wstring& error);
};