# AiDetectLib
Submodule for OpenCV and Dlib facial detection in C++.

This is a library that is used to detect faces from an OpenCV cv::Mat image video frame. It is built with cmake 3.8 for Windows and Linux with C++ 17. 

AiDetect is currenly 1 of 3 git submodules used to stream video with facial detection from a video file sample or RPi camera to a client using ZeroMQ. It allows dynamic testing of various facial AI detection methods. The other modules include ZeroMQ pub/sub messages that transmit video frames, CPU usage, frames per second etc. These other libraries will be made public after they are fully completed. 

CMakeLists.txt is set with:
```
cmake_minimum_required (VERSION 3.8)
set(CMAKE_CXX_STANDARD 17) 

find_package(OpenCV REQUIRED)
find_package(ZeroMQ REQUIRED)
find_package(dlib REQUIRED)

add_subdirectory("Imports/JsoncppLib")
add_subdirectory("Imports/MessagesLib")
add_subdirectory("Imports/AiDetectLib")
add_subdirectory("Imports/GTestLib")
add_subdirectory ("CameraService")
add_subdirectory("test")
```
This library is used as follows to detect faces and add a rectange around the faces in the video frame. 

- Start a thread that reads from a queue of OpenCV Mat images and does facial detection with the current detection method. Once processed the callback method will pass along the processed image.

```
// Initialize facial detection. Supported methods are OpenCV, Dnn, Hog, Mod and none
if (CSettings::Instance().GetUseFaceDetect())
{
  	bOK = CDetectFaces::Instance().Start(CSettings::Instance().GetFaceDetectMethod(), &CVideoSource::PublishDetectedFaces, error);
	assert(bOK);
}
```
- This is how a thread would read from the video source and queue some of the frames for facial detection. It is not practical to process 30 fps in real time unless you use Dlib Mod that is built with CUDA GPU support. So frames are skipped based on how polite you want to be on CPU usage.
```
// Get next frame from thread that reads from video source
cv::Mat image;
pThis->_opencvCaputure.read(image);
if (!image.empty())
{
  	frameCount++;
  	// Do face detection
	if (CSettings::Instance().GetUseFaceDetect())
	{
	  	if (SKIP_FRAME_NUM == 0 || (frameCount % SKIP_FRAME_NUM) == 0)
		{
		  	CDetectFaces::Instance().AddImageToQueue(image);
		}
	}
}
```
