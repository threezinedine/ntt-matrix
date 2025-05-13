project(landmark)

set(CMAKE_CXX_STANDARD 14)

add_executable(${PROJECT_NAME} landmark_test.cpp)

# find_package(OpenCV REQUIRED PATHS E:/OpenCV/opencv-4.11.0/build)
find_package(OpenCV REQUIRED PATHS C:/opencv/build)

message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    ${PROJECT_NAME} 
    PUBLIC
    ${OpenCV_LIBS}
)
