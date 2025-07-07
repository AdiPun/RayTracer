@echo off
echo Running build and run script...
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%..\.."

echo Current directory: %CD%

echo Starting build...
cmake --build x64 --config Release
echo Build finished with errorlevel %ERRORLEVEL%

echo Running executable...
x64\Release\RayTracer.exe > image.ppm
echo Executable finished with errorlevel %ERRORLEVEL%

echo Output saved to image.ppm
pause
