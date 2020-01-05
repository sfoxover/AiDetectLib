#include "DetectFaceDNN.h"
#include "helpers.h"

// Set initial settings
bool CDetectFaceDNN::Initialize(std::wstring& error)
{
	auto tensorflowConfigFile = Helpers::AppendToRunPath(Helpers::AppendPath("assets", "opencv_face_detector.pbtxt"));
	auto tensorflowWeightFile = Helpers::AppendToRunPath(Helpers::AppendPath("assets", "opencv_face_detector_uint8.pb"));
	_networkFace = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
	return !_networkFace.empty();
}

// Look for a face in image
bool CDetectFaceDNN::DetectFace(cv::Mat& image, bool addRectToFace, std::wstring& error)
{
	bool bFoundFace = false;
	const cv::Scalar meanVal(104.0, 177.0, 123.0);
	const float confidenceThreshold = 0.7;

	if (!_networkFace.empty())
	{
		cv::Mat inputBlob = cv::dnn::blobFromImage(image, 1.0, image.size(), meanVal, true, false);
		_networkFace.setInput(inputBlob, "data");
		cv::Mat detection = _networkFace.forward("detection_out");
		cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);
			if (confidence > confidenceThreshold)
			{
				int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
				int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
				int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
				int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);

				cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
				bFoundFace = true;
			}
		}
	}
	return bFoundFace;
}
