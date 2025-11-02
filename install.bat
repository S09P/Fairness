@echo off
REM === Check for Python installation ===
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Python 3.12.7...
    powershell -Command "Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe -OutFile python_installer.exe"
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    del python_installer.exe
)

echo Checking Python version...
python --version

echo.
echo Upgrading pip if needed...
python -m pip install --upgrade pip >nul 2>&1

echo.
echo Checking required packages...

REM === Function-like block to install only if missing ===
for %%p in (numpy pandas scikit-learn aif360 streamlit) do (
    python -m pip show %%p >nul 2>&1
    if %errorlevel% neq 0 (
        echo Installing %%p...
        python -m pip install %%p --quiet
    ) else (
        echo %%p already installed.
    )
)

REM === Add User Scripts folder to PATH ===
setx PATH "%PATH%;%AppData%\Python\Python311\Scripts" >nul
set PATH=%PATH%;%AppData%\Python\Python311\Scripts

echo.
echo Starting Streamlit app...
python -m streamlit run app5.py

echo.
pause
