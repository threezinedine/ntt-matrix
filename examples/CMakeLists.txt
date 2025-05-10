file(
    GLOB 
    EXAMPLE_SOURCES 
    *.cpp
)

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    target_include_directories(${EXAMPLE_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()
