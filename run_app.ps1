# =======================================================
#         Lancement de l'application RV Bonds
# =======================================================

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venvStreamlit = Join-Path $PSScriptRoot "venv\Scripts\streamlit.exe"

if (-not (Test-Path $venvStreamlit)) {
    Write-Host "[ERREUR] L'environnement virtuel n'existe pas encore ou streamlit n'y est pas installé." -ForegroundColor Red
    Write-Host "Veuillez d'abord exécuter '.\setup_env.ps1' ou 'setup_env.bat'." -ForegroundColor Yellow
    exit 1
}

$appPath = Join-Path $PSScriptRoot "rv_bonds_app\app.py"
Write-Host "Démarrage de l'application RV Bonds Streamlit..." -ForegroundColor Green
& $venvStreamlit run $appPath
