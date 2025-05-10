set(TEST_PROJECT_NAME "NTTMatrixTests")

file(GLOB TEST_SOURCES "*.cpp")

add_executable(${TEST_PROJECT_NAME} ${TEST_SOURCES})

target_link_libraries(${TEST_PROJECT_NAME} PUBLIC gtest gmock)

target_include_directories(${TEST_PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/ntt_matrix)


