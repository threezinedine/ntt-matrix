project(landmark)

set(CMAKE_CXX_STANDARD 14)

add_executable(${PROJECT_NAME} landmark_test.cpp)

find_package(OpenCV REQUIRED PATHS C:/opencv/build)

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
