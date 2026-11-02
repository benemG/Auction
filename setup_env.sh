#!/usr/bin/env bash
set -e

echo "======================================================="
echo "   Installation de l'environnement virtuel - RV Bonds  "
echo "======================================================="
echo ""

# Se positionner dans le dossier du script
cd "$(dirname "$0")"

# Vérification de Python
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo "[ERREUR] Python n'est pas installé ou n'est pas dans le PATH."
    exit 1
fi

echo "[OK] Python détecté : $($PYTHON_CMD --version)"
echo ""

# Création du venv
if [ ! -d "venv" ]; then
    echo "[1/3] Création de l'environnement virtuel 'venv'..."
    $PYTHON_CMD -m venv venv
    echo "[OK] Environnement virtuel créé avec succès."
else
    echo "[1/3] L'environnement virtuel 'venv' existe déjà."
fi
echo ""

# Activation du venv
echo "[2/3] Activation de l'environnement virtuel..."
source venv/bin/activate
echo "[OK] Environnement virtuel activé."
echo ""

# Installation des dépendances
echo "[3/3] Installation des librairies requises..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "======================================================="
echo "             Installation terminée avec succès !       "
echo "======================================================="
echo ""
echo "Pour activer l'environnement virtuel :"
echo "  source venv/bin/activate"
echo ""
echo "Pour lancer l'application Streamlit :"
echo "  streamlit run rv_bonds_app/app.py"
echo ""
