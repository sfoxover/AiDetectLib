#pragma once
/*
	CDetectFaceMod - Mod implementation of face detection methods, can be hardware accelerated, fastest on Pi4.
*/
#include "IDetectFace.h"
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>

template <long num_filters, typename SUBNET>
using con5d = dlib::con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET>
using con5 = dlib::con<num_filters, 5, 5, 1, 1, SUBNET>;
template <typename SUBNET>
using downsampler = dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET>
using rcon5 = dlib::relu<dlib::affine<con5<45, SUBNET>>>;
using net_type = dlib::loss_mmod<dlib::con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;

class CDetectFaceMod : public IDetectFace
{
private:
	net_type _mmodFaceDetector;

public:

	// Return enum for this AI detect method
	DetectMethods GetMethod()
	{
		return DetectMethods::Mod;
	}

	// Set initial settings
	bool Initialize(std::wstring& error);

	// Look for a face in image
	bool DetectFace(cv::Mat& image, bool addRectToFace, std::wstring& error);
};