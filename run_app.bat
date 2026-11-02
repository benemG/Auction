@echo off
cd /d "%~dp0"

echo =======================================================
echo         Lancement de l'application RV Bonds
echo =======================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [ERREUR] L'environnement virtuel n'existe pas encore.
    echo Veuillez d'abord executer 'setup_env.bat' pour installer l'environnement.
    echo.
    pause
    exit /b 1
)

echo Activation de l'environnement virtuel et demarrage de Streamlit...
call venv\Scripts\activate.bat
streamlit run rv_bonds_app\app.py

pause
