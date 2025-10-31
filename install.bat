@echo off
REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Python...
    powershell -Command "Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe -OutFile python_installer.exe"
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    del python_installer.exe
)

echo Installing Streamlit...
pip install streamlit

echo Starting app...
streamlit run app5
pause
