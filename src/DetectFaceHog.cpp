#include "DetectFaceHog.h"

// Set initial settings
bool CDetectFaceHog::Initialize(std::wstring& error)
{
	_hogFaceDetector = dlib::get_frontal_face_detector();
	return true;
}

// Look for a face in image
bool CDetectFaceHog::DetectFace(cv::Mat& image, bool addRectToFace, std::wstring& error)
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
