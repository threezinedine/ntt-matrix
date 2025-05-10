@REM @echo off

set current=%~dp0
set base=%current%\..\..\..\..\
set build=%base%\build\Windows\MinGW\Debug
set examples=%build%\examples\

cd %examples%
for %%f in (*.exe) do (
    %%f
)

cd %current%