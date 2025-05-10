@echo off

set current=%~dp0
set base=%current%\..\..\..\..\
set build=%base%\build\Windows\MinGW\Release
set test=%build%\tests\

cd %test%
NTTMatrixTests.exe
cd %current%
