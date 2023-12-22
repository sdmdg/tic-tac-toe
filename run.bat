@echo off

set PYTHON_EXE=python
set VENV_NAME=venv

if not exist %VENV_NAME% (
	echo Creating virtual environment...
	%PYTHON_EXE% -m venv %VENV_NAME%
)
call %VENV_NAME%\Scripts\activate
echo Installing dependencies...
%PYTHON_EXE% -m pip install -r requirements.txt
echo .
echo .
echo .
python main.py
deactivate