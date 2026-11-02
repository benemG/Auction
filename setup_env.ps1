# =======================================================
#    Installation de l'environnement virtuel - RV Bonds
#    Script PowerShell
# =======================================================

$ErrorActionPreference = "Stop"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "   Installation de l'environnement virtuel - RV Bonds  " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Se placer dans le répertoire du script
Set-Location -Path $PSScriptRoot

# 1. Vérification de Python
try {
    $pythonCmd = Get-Command python -ErrorAction Stop
    $pyVersion = (& python --version 2>&1)
    Write-Host "[OK] Python détecté : $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERREUR] Python n'est pas trouvé dans le PATH." -ForegroundColor Red
    Write-Host "Veuillez installer Python (>= 3.10) et l'ajouter à votre variable d'environnement PATH." -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# 2. Création de l'environnement virtuel
$venvPath = Join-Path $PSScriptRoot "venv"
if (-not (Test-Path -Path $venvPath)) {
    Write-Host "[1/3] Création de l'environnement virtuel 'venv'..." -ForegroundColor Yellow
    & python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERREUR] Échec de la création du venv." -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Environnement virtuel créé avec succès." -ForegroundColor Green
} else {
    Write-Host "[1/3] L'environnement virtuel 'venv' existe déjà." -ForegroundColor Cyan
}

Write-Host ""

# Chemins des exécutables dans le venv
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvPip = Join-Path $venvPath "Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[ERREUR] L'exécutable python.exe dans venv est introuvable." -ForegroundColor Red
    exit 1
}

# 3. Mise à niveau de pip
Write-Host "[2/3] Mise à niveau de pip..." -ForegroundColor Yellow
try {
    & $venvPython -m pip install --upgrade pip
} catch {
    Write-Host "[AVERTISSEMENT] Impossible de mettre à jour pip, poursuite de l'installation..." -ForegroundColor DarkYellow
}

Write-Host ""

# 4. Installation des dépendances
Write-Host "[3/3] Installation des librairies requises (requirements.txt)..." -ForegroundColor Yellow
$reqPath = Join-Path $PSScriptRoot "requirements.txt"
& $venvPython -m pip install -r $reqPath

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERREUR] Une erreur est survenue lors de l'installation des dépendances." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "             Installation terminée avec succès !       " -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Pour activer l'environnement virtuel dans PowerShell :" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "(Si l'exécution de scripts est bloquée, exécutez d'abord: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Pour lancer l'application Streamlit :" -ForegroundColor Cyan
Write-Host "  .\run_app.bat   OU   streamlit run rv_bonds_app\app.py" -ForegroundColor White
Write-Host ""
