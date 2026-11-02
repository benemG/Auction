# Relative Value Obligataire (RV Bonds)

Plateforme d'analyse et d'exploration d'opportunités de Relative Value sur la dette souveraine européenne.

## 🚀 Installation rapide

Un script d'installation automatique crée l'environnement virtuel (`venv`) et installe toutes les dépendances requises.

### Sous Windows :
- **PowerShell** :
  ```powershell
  .\setup_env.ps1
  ```
- **Invite de commande (CMD) ou double-clic** :
  ```cmd
  setup_env.bat
  ```

### Sous Linux / macOS / WSL :
- **Bash** :
  ```bash
  chmod +x setup_env.sh
  ./setup_env.sh
  ```

---

## 💻 Démarrage de l'application

Une fois l'installation terminée :

- **Via le script de lancement** :
  - Windows CMD / Double-clic : `run_app.bat`
  - Windows PowerShell : `.\run_app.ps1`
- **Manuellement** :
  ```powershell
  # Activation du venv (Windows)
  .\venv\Scripts\Activate.ps1
  # ou sous CMD: venv\Scripts\activate.bat

  # Lancement de l'UI
  streamlit run rv_bonds_app/app.py
  ```

---

## 📁 Structure du projet

- `requirements.txt` : Dépendances globales de l'application.
- `setup_env.ps1` / `setup_env.bat` / `setup_env.sh` : Scripts de création du venv et d'installation.
- `run_app.bat` / `run_app.ps1` : Lanceurs rapides de l'application Streamlit.
- `rv_bonds_app/` : Code source de l'application Streamlit (pages, modèles mathématiques P-spline & OU, scripts batch).
- `bonds/` : Données historiques obligataires par émetteur.
- `mapping_indiv_bond.csv` : Référentiel des obligations et métadonnées.
