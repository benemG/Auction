@echo off
echo Lancement de l'application RV Bonds...
echo Veuillez patienter, l'application va s'ouvrir dans votre navigateur par defaut.
cd /d "%~dp0"
call streamlit run app.py
pause
