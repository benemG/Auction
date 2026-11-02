@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo    Installation de l'environnement virtuel - RV Bonds
echo =======================================================
echo.

:: Se positionner dans le dossier du script
cd /d "%~dp0"

:: Verification de la presence de Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe ou n'est pas present dans le PATH.
    echo Veuillez installer Python (>= 3.10) et cocher "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VER=%%i
echo Python detecte : %PYTHON_VER%
echo.

:: Creation du virtualenv si inexistant
if not exist "venv" (
    echo [1/3] Creation de l'environnement virtuel 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERREUR] Impossible de creer l'environnement virtuel.
        pause
        exit /b 1
    )
    echo Environnement virtuel cree avec succes.
) else (
    echo [1/3] L'environnement virtuel 'venv' existe deja.
)
echo.

:: Activation du virtualenv
echo [2/3] Activation de l'environnement virtuel...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERREUR] Echec de l'activation de l'environnement virtuel.
    pause
    exit /b 1
)
echo Environnement virtuel active.
echo.

:: Mise a jour de pip et installation des dependances
echo [3/3] Installation des librairies requises...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [AVERTISSEMENT] La mise a jour de pip a rencontre un probleme, continuation...
)

pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERREUR] Une erreur est survenue lors de l'installation des librairies.
    pause
    exit /b 1
)

echo.
echo =======================================================
echo             Installation terminee avec succes !
echo =======================================================
echo.
echo Pour activer l'environnement virtuel manuellement :
echo   venv\Scripts\activate
echo.
echo Pour lancer l'application Streamlit :
echo   streamlit run rv_bonds_app\app.py
echo   (ou executez simplement run_app.bat)
echo.
pause
