@echo off
echo Lancement de la mise a jour des donnees via Bloomberg...
echo Assurez-vous que le terminal Bloomberg est ouvert.
cd /d "%~dp0"
call python scripts/update_from_bloomberg.py
echo.
echo Mise a jour terminee.
pause
