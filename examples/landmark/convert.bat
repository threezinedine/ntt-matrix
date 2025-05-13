@echo off 

set current=%~dp0
set base=%current%..\..

set python=%base%\venv\Scripts\python.exe
set utils=%base%\utils\npy_convert.py

for %%f in (data/*.npy) do (
    %python% %utils% data/%%f
)
