#pragma once
/*
	Scan a frame with OpenCV for faces and detection
*/

#include <mutex>
#include <condition_variable>

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

	// Current selected AI method
	DetectMethods _detectMethod;
	std::mutex _detectMethodLock;

	// Flag to add rectangle to image around detected face
	bool _addRectToFace;
	std::mutex _addRectToFaceLock;

	// Queue of cv::Mat images to run detection on 
	std::vector<cv::Mat> _imageQueue;
	std::mutex _imageQueueLock;

	// Exit thread flag
	bool _exitingFlag;
	std::mutex _exitingFlagLock;

	// Detect face thread
	std::thread _detectThread;

	// Callback method to pass detected face images to
	std::function<void(cv::Mat)> _detectedCallback;

	// Signal when new image is ready to be processed
	std::mutex _mutexNewImage;
	std::condition_variable _signalNewImage;

// Methods

	// Look for a face in image
	bool DetectFaceOpenCV(cv::Mat& image, bool addRectToFace, std::wstring& error);
	bool DetectFaceDNN(cv::Mat& image, bool addRectToFace, std::wstring& error);
	bool DetectFaceDlibHog(cv::Mat& image, bool addRectToFace, std::wstring &error);
	// Can be hardware accelerated, fastest on Pi4
	bool DetectFaceDlibMod(cv::Mat& image, bool addRectToFace, std::wstring &error);	

public:
// Methods
	static CDetectFaces& Instance()
	{
		static CDetectFaces instance;
		return instance;
	}
	
	// Set initial settings
	bool Initialize(std::string method, std::function<void(cv::Mat)> callback, std::wstring& error);

	// Stop face detection thread
	bool Stop(std::wstring& error);

	// Get set for _detectMethod
	void GetDetectMethod(DetectMethods& value);
	void SetDetectMethod(DetectMethods value);

	// Get set for _addRectToFace
	void GetAddRectToFace(bool& value);
	void SetAddRectToFace(bool value);

	// Get set for _exitingFlag
	void GetExitingFlag(bool& value);
	void SetExitingFlag(bool value);

	// Return face detect AI method as string
	std::string GetDetectMethod();

	// Add new image to queue
	bool AddImageToQueue(cv::Mat image);

	// Look for a face in image on separate thread
	static void DetectFacesThread(CDetectFaces* pThis);
};