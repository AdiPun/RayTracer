@echo off
setlocal

echo === BuildProject.cmd ===

:: Start in script's directory, then go to repo root
cd /d "%~dp0..\.."

:: Now go into the actual source folder that contains CMakeLists.txt
cd RayTracer

echo [CMake] Generating build files in x64...
cmake -B ../x64 -S . -DCMAKE_BUILD_TYPE=Release

if errorlevel 1 (
    echo [Error] CMake configuration failed!
    pause
    exit /b 1
)

echo [CMake] Building project (Release mode)...
cmake --build ../x64 --config Release

if errorlevel 1 (
    echo [Error] Build failed!
    pause
    exit /b 1
)

echo [Success] Project built successfully.
pause
