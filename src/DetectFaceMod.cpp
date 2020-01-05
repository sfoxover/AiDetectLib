#include "DetectFaceMod.h"
#include "helpers.h"

// Set initial settings
bool CDetectFaceMod::Initialize(std::wstring& error)
{
	std::string mmodModelPath = Helpers::AppendToRunPath(Helpers::AppendPath("assets", "mmod_human_face_detector.dat"));
	dlib::deserialize(mmodModelPath) >> _mmodFaceDetector;
	return true;
}

// Look for a face in image
bool CDetectFaceMod::DetectFace(cv::Mat& image, bool addRectToFace, std::wstring& error)
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
