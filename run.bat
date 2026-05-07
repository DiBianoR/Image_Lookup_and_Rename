@echo off
echo [*] Initializing Art Metadata Pipeline...

:: Check if the virtual environment exists, if not, create it
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [*] Creating virtual environment...
    python -m venv venv
)

:: Activate the environment
call venv\Scripts\activate

:: Install requirements silently (only shows errors or warnings)
echo [*] Checking requirements...
pip install -r requirements.txt -q

:: Run the script
echo [*] Launching pipeline...
python image_name_finder.py

:: Keep the window open when finished
echo.
pause