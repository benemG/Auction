@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r rv_bonds_app\requirements.txt

echo.
echo Setup complete.
echo To activate the virtual environment in the future, run:
echo venv\Scripts\activate
pause
