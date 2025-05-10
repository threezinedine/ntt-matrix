@echo off

set current=%~dp0
set base=%current%\..\..\..\..\
set build=%base%\build\Windows\VC\2022

set CMAKE_GENERATOR="Visual Studio 17 2022"

cmake -G %CMAKE_GENERATOR% -B %build% -S %base%
