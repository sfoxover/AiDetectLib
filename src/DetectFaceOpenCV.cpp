#include "DetectFaceOpenCV.h"
#include "helpers.h"

// Set initial settings
bool CDetectFaceOpenCV::Initialize(std::wstring& error)
{
	auto filePath = Helpers::AppendToRunPath(Helpers::AppendPath("assets", "haarcascade_frontalface_default.xml"));
	bool bOK = _faceFront.load(filePath);
	assert(bOK);
	if (!bOK)
	{
		error = L"Failed to load haarcascade_frontalface_default.xml file.";
	}
	else
	{
		filePath = Helpers::AppendToRunPath(Helpers::AppendPath("assets", "haarcascade_profileface.xml"));
		bOK = _faceProfile.load(filePath);
		assert(bOK);
		if (!bOK)
		{
			error = L"Failed to load haarcascade_profileface.xml file.";
		}
	}
	return bOK;
}

// Look for a face in image
bool CDetectFaceOpenCV::DetectFace(cv::Mat& image, bool addRectToFace, std::wstring& error)
{
	bool bFoundFace = false;
	if (!_faceFront.empty())
	{
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

		bFoundFace = !faces.empty() || !profiles.empty();
	}
	return bFoundFace;
}
