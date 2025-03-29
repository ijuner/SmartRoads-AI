@echo off
echo Starting Face Detection System
echo ==============================

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python not found! Please install Python and try again.
    pause
    exit /b
)

:: Install dependencies
echo Installing required packages...
python -m pip install fastapi streamlit opencv-python numpy uvicorn requests python-multipart pillow

:: Start the server in a new command window
echo Starting the FastAPI server...
start cmd /k python face_detection_server.py

:: Wait for server to start
timeout /t 3 /nobreak

:: Start the Streamlit app
echo Starting the Streamlit frontend...
start cmd /k python -m streamlit run streamlit_frontend.py

echo Both services are now running.
echo - FastAPI server: http://localhost:8000
echo - Check the Streamlit window for the frontend URL
echo.
echo Press any key to stop all services when you're done.
pause

:: Kill all python processes when done (be careful with this on a dev machine)
echo Stopping services...
taskkill /f /im python.exe
echo Done.