#pragma once

#include "IDetectFace.h"
#include "DetectFaceOpenCV.h"
#include "DetectFaceDNN.h"
#include "DetectFaceMod.h"
#include "DetectFaceHog.h"

class DetectFaceFactory
{
public:

	// Factory method to create face detection object
	static std::shared_ptr<IDetectFace> Create(std::string method)
	{
		std::shared_ptr<IDetectFace> spFace;

		// Supported methods are OpenCV, Dnn, Hog, Mod 
		if (method == "OpenCV")
		{
			spFace = std::make_shared<CDetectFaceOpenCV>();
		}
		else if (method == "Dnn")
		{
			spFace = std::make_shared<CDetectFaceDNN>();
		}
		else if (method == "Hog")
		{
			spFace = std::make_shared<CDetectFaceHog>();
		}
		else if (method == "Mod")
		{
			spFace = std::make_shared<CDetectFaceMod>();
		}
		else
		{
			assert(false);
		}
		return spFace;
	}
};