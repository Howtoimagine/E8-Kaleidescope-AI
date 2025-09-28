@echo off
REM Start script for E8 Mind M24 Kaleidoscope
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Starting E8 Mind M24...
python main.py