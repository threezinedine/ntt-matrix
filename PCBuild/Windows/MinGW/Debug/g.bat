@echo off

set current=%~dp0
set base=%current%\..\..\..\..\
set build=%base%\build\Windows\MinGW\Debug

set CMAKE_GENERATOR="MinGW Makefiles"
set CMAKE_BUILD_TYPE=Debug

cmake -G %CMAKE_GENERATOR% -B %build% -S %base% -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE%
