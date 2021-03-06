# AiDetectLib shared library

cmake_minimum_required (VERSION 3.8)

set(AI_LIB_NAME "AiDetectLib")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")
file(GLOB_RECURSE LIB_HEADERS "src/*.h include/*.hpp")

# The variable 'dlib_INCLUDE_DIRS' is deprecated
include_directories(include ${OpenCV_INCLUDE_DIRS})

# Add source to this project's executable.
add_library(${AI_LIB_NAME} STATIC ${LIB_SOURCES} ${LIB_HEADERS})

target_include_directories(${AI_LIB_NAME} PUBLIC include PUBLIC src)

target_link_libraries(${AI_LIB_NAME} MessagesLib dlib::dlib)

install(TARGETS ${AI_LIB_NAME} DESTINATION lib)
install(FILES ${LIB_HEADERS} DESTINATION include)

