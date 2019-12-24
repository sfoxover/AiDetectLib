#pragma once
/*
	Scan a frame with OpenCV for faces and detection
*/
#include <opencv2/opencv.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <dlib/dnn.h>
#include <dlib/data_io.h>

template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = dlib::con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = dlib::relu<dlib::affine<con5<45,SUBNET>>>;

using net_type = dlib::loss_mmod<dlib::con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;


class CDetectFaces
{
private:
	CDetectFaces();
	~CDetectFaces();

	// Initialize detection routines
	void InitOpenCV();
	void InitDNN();	
	void InitHog();
	void InitMod();

private:
// Properties
	// Detect faces
	cv::dnn::Net _networkFace;
	cv::CascadeClassifier _faceFront;
	cv::CascadeClassifier _faceProfile;
	dlib::frontal_face_detector _hogFaceDetector;
	net_type _mmodFaceDetector;

	// Detection method
	enum DetectMethods
	{
		none, OpenCV, Dnn, Hog, Mod
	};
	DetectMethods _detectMethod;

// Methods
	// Look for a face in image
	bool DetectFaceOpenCV(cv::Mat image, bool addRectToFace, std::wstring& error);
	bool DetectFaceDNN(cv::Mat image, bool addRectToFace, std::wstring& error);
	bool DetectFaceDlibHog(cv::Mat image, bool addRectToFace, std::wstring &error);
	// Can be hardware accelerated, fastest on Pi4
	bool DetectFaceDlibMod(cv::Mat image, bool addRectToFace, std::wstring &error);	

public:
// Methods
	static CDetectFaces& Instance()
	{
		static CDetectFaces instance;
		return instance;
	}

	// Return face detect AI method as string
	std::string GetDetectMethod();

	// Set initial settings
	bool Initialize(std::string method, std::wstring& error);

	// Look for a face in image
	void DetectFaces(cv::Mat image, bool addRectToFace);
};