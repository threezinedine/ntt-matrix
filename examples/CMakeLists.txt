file(
    GLOB 
    EXAMPLE_SOURCES 
    **/*.cpp
)

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    # check if the folder contains CMakeLists.txt
    get_filename_component(EXAMPLE_DIR ${EXAMPLE_SOURCE} PATH)
    if (EXISTS ${EXAMPLE_DIR}/CMakeLists.txt)
        message(STATUS "Building ${EXAMPLE_DIR}")
        if (MSVC)
            add_subdirectory(${EXAMPLE_DIR})
        endif()
    else()
        get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
        add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
        target_include_directories(${EXAMPLE_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)
    endif()
endforeach()
