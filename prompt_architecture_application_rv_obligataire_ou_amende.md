# Prompt architecte — Application Streamlit de Relative Value Obligataire

## Mission

Tu es un **LLM architecte logiciel** coordonnant une équipe d'agents (data engineering, recherche quantitative et développement Streamlit). Conçois et spécifie une application Streamlit légère, locale et extrêmement réactive, destinée à une équipe de stratégistes taux qui prépare des adjudications obligataires.

L'objectif opérationnel est d'identifier, pour chaque obligation mise aux enchères, au moins **trois trades de relative value** pertinents :

- Spread à deux jambes ;
- Fly à trois jambes ;
- Credit fly, le cas échéant inter-émetteurs ;
- Box à quatre jambes.

L'application exploite un historique quotidien de prix, rendement actuariel (YTM), asset swap spread (ASW) et z-spread par obligation. Elle doit rendre les opportunités explorables immédiatement, sans recalcul statistique coûteux dans l'interface.

## Contraintes non négociables

Conçois une solution volontairement légère :

- Application Python/Streamlit exécutée localement ou sur un serveur interne simple.
- Aucun système d'authentification ni gestion de rôles.
- Aucun Docker, Kubernetes, GitHub, CI/CD ou couche cloud imposée.
- Aucun serveur de base de données requis : Apache Parquet est le stockage analytique canonique.
- Pas de calcul massif lors de l'interaction utilisateur : les fits, combinaisons et métriques de mean reversion sont calculés en amont et persistés.
- Dépendances limitées à Python, Streamlit, pandas, NumPy, SciPy, statsmodels, Plotly et PyArrow/DuckDB si nécessaire pour filtrer Parquet rapidement.
- Interface sobre, optimisée pour un analyste/trader : peu de clics, tableau filtrable, graphiques immédiatement lisibles, export simple.

## Utilisateurs et décisions

### Utilisateurs

- Stratège taux : sélectionne les RV trades autour d'une obligation à émettre.
- Trader : consulte le niveau courant, la cible et le hedge ratio.
- Quant/researcher : contrôle le fit de courbe, les diagnostics et les paramètres de mean reversion.

### Cas d'usage

1. **Pré-adjudication** : sélectionner l'émetteur et l'obligation nouvelle ou réouverte, puis faire ressortir les trois meilleures expressions RV.
2. **Scan quotidien** : consulter les opportunités classées selon qualité statistique, potentiel de convergence et liquidité.
3. **Analyse d'un trade** : visualiser le spread observé, son équivalent fitté, la cible, l'historique, le z-score et les risques.
4. **Contrôle quantitatif** : vérifier la qualité du P-spline, la cointégration, la vitesse de mean reversion et la robustesse hors échantillon.
5. **Production de fiche** : exporter une fiche de trade concise, en CSV ou Markdown, pour diffusion interne.

## Architecture fonctionnelle

L'architecture doit séparer strictement les traitements batch de l'interface :

```text
CSV bruts par émetteur
      |
      v
Ingestion + validation + normalisation
      |
      v
Parquet historique normalisé
      |
      v
Fit quotidien P-spline par émetteur / métrique
      |
      v
Parquet des valeurs fittées et résidus
      |
      v
Génération exhaustive des combinaisons de trades
      |
      v
Tests MR + modèle OU de référence + diagnostics VECM optionnels
      |
      v
Parquet opportunités et classement quotidien
      |
      v
Application Streamlit de lecture / filtrage / visualisation
```

Les traitements en amont doivent être lancés par scripts Python explicites, par exemple après mise à jour des CSV ou via une tâche planifiée simple du système d'exploitation. L'application Streamlit est essentiellement en lecture des Parquet précalculés.

## Arborescence cible

Propose une arborescence similaire à celle-ci, en l'adaptant si nécessaire :

```text
rv_bonds_app/
├── app.py
├── pages/
│   ├── 1_Scanner.py
│   ├── 2_Trade_Explorer.py
│   ├── 3_Curve_Diagnostics.py
│   └── 4_Parameters.py
├── src/
│   ├── config.py
│   ├── data_io.py
│   ├── data_validation.py
│   ├── pspline_curve.py
│   ├── curve_pipeline.py
│   ├── trade_definitions.py
│   ├── opportunity_generation.py
│   ├── mean_reversion.py
│   ├── hedge_ratio.py
│   ├── scoring.py
│   ├── charts.py
│   └── exports.py
├── scripts/
│   ├── ingest_csv_to_parquet.py
│   ├── fit_daily_curves.py
│   ├── generate_opportunities.py
│   └── refresh_all.py
├── data/
│   ├── raw/
│   │   ├── OAT/
│   │   ├── OLO/
│   │   └── BTP/
│   └── parquet/
│       ├── market_data/
│       ├── fitted_curves/
│       ├── opportunities/
│       └── reference/
├── tests/
├── requirements.txt
└── README.md
```

## Données d'entrée

Les fichiers CSV bruts sont stockés dans un sous-dossier par émetteur, par exemple `data/raw/OAT/`. Chaque ligne est une observation quotidienne d'une obligation.

Le pipeline d'ingestion doit :

- Harmoniser les noms de colonnes et les formats de dates ;
- Convertir les identifiants en chaînes (`isin`, `issuer`, `bond_name`) ;
- Dédupliquer sur `(date, isin)` ;
- Contrôler les valeurs manquantes et aberrantes ;
- Conserver les champs nécessaires à la construction des hedges ;
- Écrire les données partitionnées par `issuer` et idéalement par année, avec compression `zstd`.

### Schéma minimal : données de marché

Fichier : `market_data/issuer={issuer}/year={year}/data.parquet`

| Colonne | Type | Description |
|---|---:|---|
| `date` | datetime | Date de valorisation |
| `issuer` | string | Émetteur, ex. OAT, OLO, BTP |
| `isin` | string | Identifiant unique de l'obligation |
| `bond_name` | string | Libellé de marché |
| `maturity` | datetime | Date de maturité |
| `coupon` | float | Coupon annuel, en pourcentage |
| `price_clean` | float | Prix clean, en pourcentage du nominal |
| `ytm` | float | Rendement actuariel, en pourcentage ou décimal selon convention explicitée |
| `asset_swap_spread` | float | ASW, en points de base |
| `z_spread` | float | Z-spread, en points de base |
| `modified_duration` | float | Duration modifiée |
| `dv01` | float | DV01 par unité de nominal ; calculé si absent et si les intrants sont suffisants |
| `convexity` | float | Convexité ; optionnelle mais souhaitable |
| `outstanding_amount` | float | Encours ; facultatif, utile comme proxy de liquidité |
| `volume` | float | Volume/turnover si disponible |

Les conventions de signe, unités et day-count doivent être centralisées dans `src/config.py` et testées. En particulier, les spreads doivent être normalisés en points de base, et les rendements stockés selon une convention unique documentée.

## Fit de courbe : P-spline obligatoire

Le fit quotidien par émetteur et par métrique doit reposer sur un **P-spline** de B-splines pénalisées. Ne pas substituer Nelson-Siegel, Svensson ou spline cubique non pénalisée comme méthode primaire.

Le modèle doit minimiser :

\[
\min_{\theta} \lVert y - B\theta \rVert^2 + \lambda \lVert D\theta \rVert^2
\]

avec :

- `B` : matrice de base B-spline ;
- `theta` : coefficients B-spline ;
- `D` : matrice de différences d'ordre `penalty_order` ;
- `lambda` : intensité du lissage.

### Implémentation de référence

La classe de référence à intégrer, nettoyer, tester et étendre est la suivante dans son principe :

```python
class PSplineCurveFitter:
    def load_market_data(self, maturity_dates, rates):
        # Conversion maturités -> TTM en Actual/365.25,
        # exclusion des observations invalides et tri croissant.
        ...

    def fit_curve(self, n_knots=20, degree=3, penalty_order=2, lambda_param=1.0):
        # B-splines + matrice de pénalité des différences,
        # résolution : (B.T @ B + lambda * D.T @ D) theta = B.T @ y.
        ...

    def compute_fitted_yield(self, year_fraction):
        # Évaluation de la base B-spline et projection B_new @ theta.
        ...
```

Les utilitaires attendus sont :

- `BSpline.design_matrix` de SciPy pour construire la base ;
- des nœuds régulièrement espacés sur l'intervalle de maturité ;
- une matrice de pénalité par différences successives ;
- une évaluation des valeurs fittées par `B_new @ theta` ;
- un avertissement explicite, ou mieux un statut retourné, en cas d'extrapolation hors de la plage calibrée.

### Exigences de robustesse du P-spline

1. Utiliser la date de valorisation comme `reference_date`, jamais `Timestamp.today()` dans les traitements historiques.
2. Écarter les obligations échues, les maturités non positives, les valeurs non finies et les doublons de maturité selon une règle documentée.
3. Appliquer des garde-fous si le nombre d'observations est insuffisant ou si la matrice normale est mal conditionnée.
4. Prévoir une résolution stable : régularisation minimale, solveur robuste ou pseudo-inverse comme solution de repli documentée.
5. Sélectionner `n_knots` et `lambda_param` via une grille modeste ou une heuristique stable ; journaliser les paramètres retenus et la qualité du fit.
6. Produire les diagnostics : RMSE, MAE, résidu médian, nombre d'observations, plage de TTM, paramètres du spline et drapeau d'extrapolation.
7. Ne pas utiliser les points extrapolés pour classer une opportunité, sauf paramètre explicite d'override.
8. Définir si le fit cible le YTM, l'ASW ou le z-spread : l'implémentation doit pouvoir fitter chaque métrique séparément par émetteur et par date.

### Sortie du fit quotidien

Fichier : `fitted_curves/issuer={issuer}/year={year}/fitted_values.parquet`

| Colonne | Description |
|---|---|
| `date`, `issuer`, `isin` | Clés de jointure |
| `ttm_years` | Maturité résiduelle en années Actual/365.25 |
| `ytm_market`, `ytm_fitted`, `ytm_residual_bps` | Observé, fitté et résidu YTM |
| `asw_market`, `asw_fitted`, `asw_residual_bps` | Observé, fitté et résidu ASW |
| `zspread_market`, `zspread_fitted`, `zspread_residual_bps` | Observé, fitté et résidu z-spread |
| `fit_rmse_bps`, `fit_mae_bps` | Diagnostics de qualité de la courbe du jour |
| `n_observations`, `n_knots`, `degree`, `penalty_order`, `lambda_param` | Paramètres du fit |
| `is_extrapolated` | Indicateur hors bornes de calibration |

Les unités doivent être cohérentes : les résidus et métriques de spread utilisés par les trades doivent être en points de base.

## Définition des trades

Toutes les combinaisons admissibles sont générées préalablement, puis enregistrées. L'interface ne construit pas exhaustivement les combinaisons à chaque clic.

### Spread

Pour deux obligations A et B :

\[
S_t = x_{A,t} - x_{B,t}
\]

ou `x` représente, au choix, le YTM converti en bp, l'ASW ou le z-spread.

La valeur fittée équivalente est :

\[
S_t^{fit} = x_{A,t}^{fit} - x_{B,t}^{fit}
\]

### Fly

Pour ailes A et C, et belly B :

\[
F_t = x_{A,t} - 2x_{B,t} + x_{C,t}
\]

Prévoir également une pondération de belly basée sur DV01/duration lorsque nécessaire afin de rendre la structure de risque taux approximativement neutre.

### Credit fly

Définir explicitement une structure entre deux ou plusieurs émetteurs : par exemple un fly de spreads crédit/ASW autour de maturités comparables. Toute comparaison inter-émetteurs doit contrôler devise, indice de swap de référence, seniorité, type d'instrument et segment de maturité.

### Box

Pour quatre obligations A, B, C et D :

\[
B_t = (x_{A,t} - x_{B,t}) - (x_{C,t} - x_{D,t})
\]

La convention de signe et les coefficients de chaque jambe doivent être persistés dans les métadonnées de trade, pas reconstruits implicitement dans l'interface.

### Filtres d'admissibilité

Les combinaisons doivent être filtrées avant tests statistiques par :

- dates communes suffisantes ;
- période historique minimale configurable, par défaut 12 mois de données quotidiennes ;
- qualité de données minimale ;
- même devise et benchmark de swap pour les comparaisons pertinentes ;
- maturités cohérentes avec le type de structure ;
- liquidité minimale si données de volume ou encours disponibles ;
- exclusion d'obligations arrivant très proche de maturité, selon seuil configurable ;
- exclusion ou étiquetage des observations dont le fit est extrapolé ou de faible qualité.

## Mean reversion : spécification OU obligatoire

L'implémentation de référence du modèle de mean reversion est une classe Python `MeanReversionModel` qui représente un processus d'Ornstein-Uhlenbeck (OU). Les spécifications ci-dessous sont obligatoires et priment sur toute formulation OU générique.

### Processus et paramètres

Le processus est :

\[
dX_t = \theta(\mu - X_t)dt + \sigma dW_t
\]

avec :

- `theta` : vitesse de retour vers la moyenne ;
- `mu` : moyenne de long terme ;
- `sigma` : volatilité instantanée du processus ;
- `dt` : pas temporel utilisé durant la calibration, par exemple `1/252` pour une série quotidienne exprimée en années.

Le modèle est calibré séparément pour chaque opportunité, sur la série de spread de marché ou sur un résidu explicitement documenté. Le choix de la série calibrée doit être enregistré dans le Parquet (`ou_input_series`).

### Implémentation de référence obligatoire

Intégrer une version production, testée et numériquement robuste de l'API suivante :

```python
class MeanReversionModel:
    def __init__(self):
        self.theta = None
        self.mu = None
        self.sigma = None
        self.dt = None
        self.is_calibrated = False

    def calibrate_ou_model(self, time_series: pd.Series, dt: float) -> dict:
        # Estimation MLE fermée à partir de X_t et X_{t+1}.
        ...

    def compute_future_values(self, current_value: float, horizon_t: float) -> float:
        # E[X_t|X_0] = X_0 exp(-theta t) + mu (1 - exp(-theta t)).
        ...

    def compute_future_volatility(self, horizon_t: float) -> float:
        # sqrt[(sigma^2/(2 theta)) * (1 - exp(-2 theta t))].
        ...

    def compute_first_hitting_time(self, current_value: float) -> dict:
        # Temps de premier franchissement de mu par intégration numérique.
        ...
```

### Calibration MLE : formule et convention

La calibration utilise des observations consécutives `X = time_series.values`, avec :

```python
N = len(X) - 1
XX = X[:-1]
YY = X[1:]

Sx = np.sum(XX)
Sy = np.sum(YY)
Sxx = np.dot(XX, XX)
Sxy = np.dot(XX, YY)
Syy = np.dot(YY, YY)
```

Les estimateurs de référence sont :

\[
\hat{\mu} = \frac{S_y S_{xx} - S_x S_{xy}}{N(S_{xx} - S_{xy}) - (S_x^2 - S_x S_y)}
\]

\[
\hat{\theta} = -\frac{1}{dt}\ln\left(\frac{S_{xy}-\hat{\mu}S_x-\hat{\mu}S_y+N\hat{\mu}^2}{S_{xx}-2\hat{\mu}S_x+N\hat{\mu}^2}\right)
\]

\[
\hat{\sigma} = \sqrt{\hat{\sigma}_{\varepsilon}^2\frac{2\hat{\theta}}{1-\exp(-2\hat{\theta}dt)}}
\]

avec :

\[
\hat{\sigma}_{\varepsilon}^2 = \frac{1}{N}\left[S_{yy} - 2e^{-\hat{\theta}dt}S_{xy} + e^{-2\hat{\theta}dt}S_{xx} -2\hat{\mu}(1-e^{-\hat{\theta}dt})(S_y-e^{-\hat{\theta}dt}S_x) + N\hat{\mu}^2(1-e^{-\hat{\theta}dt})^2\right]
\]

Le code de production doit rester algébriquement fidèle à ces spécifications. Il doit toutefois contrôler rigoureusement les cas dégénérés décrits ci-dessous au lieu de retourner silencieusement des `NaN`, des valeurs complexes ou des paramètres économiquement incohérents.

### Contrôles de robustesse nécessaires

Avant et après calibration :

1. Imposer une série `pd.Series` numérique, triée chronologiquement, sans valeurs manquantes après traitement, avec au moins un nombre minimum configurable d'observations ; par défaut, 63 observations quotidiennes.
2. Documenter et contrôler `dt > 0`. Pour une série quotidienne de jours de trading et des horizons exprimés en années, fixer par défaut `dt = 1/252`.
3. Détecter un dénominateur de `mu_mle` nul ou presque nul, un ratio à l'intérieur du logarithme non strictement positif, et une `sigma2_hat` négative au-delà d'une tolérance numérique.
4. Refuser ou étiqueter une calibration lorsque `theta <= 0`, `sigma <= 0`, ou lorsqu'un paramètre est non fini.
5. Faire la distinction entre une petite `theta` économiquement plausible et une valeur numériquement indéterminée ; stocker le statut et la raison d'échec ou d'avertissement.
6. Ne pas annualiser ou désannualiser implicitement : `theta`, `sigma`, `dt`, `horizon_t`, half-life et first hitting time doivent partager une convention temporelle explicitement stockée.
7. Prévoir un fallback explicite, par exemple une estimation AR(1) équivalente ou un statut `calibration_failed`, mais ne jamais masquer la méthode réellement utilisée.
8. Journaliser la taille d'échantillon, la période de calibration, les statistiques descriptives, les paramètres et les drapeaux de qualité.

### Valeur future attendue

La prévision conditionnelle doit suivre strictement :

\[
\mathbb{E}[X_t\mid X_0] = X_0\exp(-\theta t) + \mu\left(1-\exp(-\theta t)\right)
\]

La méthode `compute_future_values(current_value, horizon_t)` doit accepter une valeur courante et un horizon dans l'unité temporelle cohérente avec `theta`. La page de détail doit pouvoir afficher la trajectoire d'espérance pour plusieurs horizons : 1 semaine, 1 mois, 3 mois, 6 mois et 12 mois, avec conversion explicite en années lorsque `dt=1/252`.

### Volatilité future

La volatilité conditionnelle à horizon `t` est :

\[
\sqrt{\frac{\sigma^2}{2\theta}\left(1-\exp(-2\theta t)\right)}
\]

La méthode `compute_future_volatility(horizon_t)` doit retourner cette valeur et fournir les bandes de confiance conditionnelles du spread. L'interface doit pouvoir tracer l'espérance OU entourée de bandes à ±1 et ±2 écarts-types à plusieurs horizons.

### Half-life

Lorsque `theta > 0`, calculer :

\[
\text{half-life} = \frac{\ln(2)}{\theta}
\]

Convertir ensuite la valeur dans l'unité affichée, typiquement les jours de trading. Si `theta` est calibré en années, alors :

\[
\text{half-life}_{days} = \frac{\ln(2)}{\theta} \times 252
\]

La half-life ne doit pas être affichée comme fiable si le statut de calibration est invalide, si elle dépasse un plafond configurable ou si elle est inférieure à un seuil non exploitable.

### First hitting time vers mu

Le **first hitting time** demandé est le temps de premier franchissement du niveau de mean reversion `mu`, pas une approximation linéaire vers la cible P-spline.

Pour une valeur initiale différente de `mu`, définir :

\[
C = (X_0-\mu)\frac{\sqrt{2\theta}}{\sigma}
\]

La densité auxiliaire est :

\[
f(t,C) = \sqrt{\frac{2}{\pi}}\frac{|C|e^{-t}}{(1-e^{-2t})^{3/2}}\exp\left(-\frac{C^2e^{-2t}}{2(1-e^{-2t})}\right)
\]

La spécification de référence implique une intégration numérique par `scipy.integrate.quad`, sur la borne `[0, 1000]`, avec la transformation utilisée dans la classe fournie :

```python
_density_T_to_theta(t, C) = (
    np.sqrt(2 / np.pi)
    * np.abs(C)
    * np.exp(-t)
    / (1 - np.exp(-2 * t)) ** (3 / 2)
    * np.exp(-((C**2) * np.exp(-2 * t)) / (2 * (1 - np.exp(-2 * t))))
)

theoretical_T = quad(
    lambda t: t * theta * _density_T_to_theta(theta * t, C),
    0,
    1000,
)[0]

theoretical_std = np.sqrt(
    quad(
        lambda t: (t - theoretical_T) ** 2
        * theta
        * _density_T_to_theta(theta * t, C),
        0,
        1000,
    )[0]
)
```

La méthode `compute_first_hitting_time(current_value)` retourne obligatoirement :

```python
{
    "expected_time": theoretical_T,
    "standard_deviation": theoretical_std,
}
```

Règles de présentation et de robustesse :

- Si `current_value == mu`, retourner `expected_time = 0.0` et `standard_deviation = 0.0`.
- Utiliser une tolérance configurable plutôt qu'une égalité flottante stricte en production ; persister cette tolérance.
- Si l'intégrale ne converge pas, produit une valeur non finie ou dépasse une durée maximale admissible, retourner un statut/flag clair et ne pas classer ce résultat comme exploitable.
- Les valeurs affichées sont converties en jours de trading si l'intégration est réalisée en années.
- Conserver `expected_time` et `standard_deviation` séparément ; ne pas réduire la métrique à une seule estimation ponctuelle.
- Distinguer ce first hitting time vers `mu` de la convergence attendue vers `spread_fitted_current` : les deux niveaux ont des rôles analytiques différents.

### Série calibrée, cible et signal

L'application doit toujours afficher trois objets distincts :

1. `spread_market_bps` : combinaison des valeurs de marché ;
2. `spread_fitted_bps` : combinaison des valeurs P-spline fittées à la date courante ;
3. `ou_mu_bps` : moyenne long terme du processus OU calibré.

La cible de trading principale est par défaut `spread_fitted_bps`. La moyenne OU est un outil statistique complémentaire et la cible du calcul de first hitting time. Ces deux niveaux ne doivent jamais être confondus.

La direction recommandée découle du signe de :

\[
\text{target gap} = spread_{fitted} - spread_{market}
\]

avec convention de legs persistée. Le modèle OU doit compléter, non remplacer, le diagnostic de dislocation vis-à-vis de la courbe P-spline.

### VECM

Pour fly, credit fly ou box lorsque l'analyse porte sur plusieurs niveaux de spreads plutôt que sur une combinaison prédéfinie, implémenter un VECM à titre de diagnostic et/ou de construction de vecteur de cointégration. Stocker :

- nombre de relations de cointégration ;
- vecteur(s) beta ;
- coefficients d'ajustement alpha ;
- lags retenus ;
- statistiques de test ;
- statut de convergence.

Éviter le VECM lorsque la taille d'échantillon est insuffisante. Dans ce cas, afficher `not_available` plutôt que de produire une métrique non fiable.

## Ratio de nominal et neutralité de risque

Le ratio de nominal doit être calculé avec une convention claire :

- Pour un spread A-B, neutralité DV01 :

\[
N_B = -N_A \times \frac{DV01_A}{DV01_B}
\]

- Si seul le duration-adjustment est disponible, utiliser une approximation documentée sur duration modifiée et prix/nominal.
- Pour fly et box, résoudre les poids sous contraintes de neutralité DV01, convention de structure et normalisation du nominal brut.
- Produire à la fois les coefficients signés par jambe et les nominaux relatifs absolus.
- Afficher l'exposition DV01 résiduelle estimée et alerter si elle dépasse une tolérance configurable.

## Score et classement

Construire un `confidence_score` entre 0 et 100 uniquement à partir de métriques disponibles et interprétables. Le score doit être décomposable : l'interface doit montrer les sous-scores, les seuils et les exclusions éventuelles.

Exemple de composantes, avec pondérations configurables :

- qualité de cointégration ;
- Hurst inférieur à 0,5 ;
- half-life dans une plage exploitable ;
- amplitude du z-score absolu ;
- distance entre spread marché et cible fittée ;
- ratio rendement/risque ou Sharpe de backtest ;
- qualité du fit P-spline ;
- liquidité et ancienneté de la série ;
- validité de la calibration OU et qualité/dispersion du first hitting time.

Ne pas sélectionner mécaniquement une opportunité uniquement parce que son z-score est extrême. Une opportunité ayant une mauvaise cointégration, un Hurst non mean-reverting, une half-life irréaliste, un mauvais fit, une calibration OU invalide ou une liquidité insuffisante doit être pénalisée ou exclue.

## Parquet des opportunités

Créer un fichier de scan compact, partitionné au besoin par date, émetteur et type : `opportunities/date={date}/issuer={issuer}/data.parquet`.

| Colonne | Description |
|---|---|
| `as_of_date` | Date de calcul |
| `trade_id` | Identifiant stable et lisible |
| `issuer` | Émetteur principal |
| `trade_type` | `spread`, `fly`, `credit_fly`, `box` |
| `metric` | `ytm`, `asw`, `zspread` |
| `bond1_isin` à `bond4_isin` | Jambes du trade ; null si non utilisées |
| `bond1_name` à `bond4_name` | Libellés des jambes |
| `leg_weights` | Coefficients signés persistés, sérialisés de façon simple |
| `spread_market_bps` | Valeur de marché actuelle |
| `spread_fitted_bps` | Valeur à partir des courbes P-spline courantes |
| `spread_target_bps` | Cible principale ; par défaut valeur fittée courante |
| `ou_long_run_mean_bps` | `mu` OU, distinct de la cible fittée |
| `ou_input_series` | `market_spread`, `fitted_residual` ou définition explicite |
| `ou_dt` | Pas temporel de calibration |
| `ou_time_unit` | Convention temporelle, par exemple `years_252_trading_days` |
| `ou_calibration_status` | `valid`, `warning`, `failed`, `not_available` |
| `ou_calibration_message` | Message d'échec, avertissement ou diagnostic concis |
| `ou_sample_size` | Nombre d'observations de calibration |
| `ou_window_start`, `ou_window_end` | Bornes de l'échantillon OU |
| `target_gap_bps` | `spread_target_bps - spread_market_bps` |
| `daily_vol_bps` | Volatilité quotidienne |
| `zscore_3m`, `zscore_12m` | Z-scores historiques |
| `half_life_days` | Durée de demi-retour estimée |
| `momentum_signal` | Signal standardisé et conventionné |
| `first_hitting_time_days` | Espérance de premier passage vers `ou_mu` |
| `first_hitting_time_std_days` | Écart-type du premier passage vers `ou_mu` |
| `first_hitting_time_status` | `valid`, `warning`, `failed`, `not_available` |
| `sharpe_ratio` | Sharpe du backtest défini |
| `cointegration_pvalue` | P-value pertinente |
| `hurst_exponent` | Hurst |
| `ou_theta`, `ou_mu`, `ou_sigma` | Paramètres OU MLE |
| `vecm_status`, `vecm_alpha`, `vecm_beta` | Diagnostics VECM si applicables |
| `nominal_ratio` | Ratio principal, pour trades à deux jambes |
| `leg_notionals` | Nominaux relatifs par jambe |
| `residual_dv01` | Exposition DV01 résiduelle estimée |
| `fit_rmse_bps` | Qualité de fit pertinente pour le trade |
| `liquidity_score` | Score de liquidité si disponible |
| `recommended_direction` | `long_spread` ou `short_spread`, avec convention explicite |
| `confidence_score` | Score composite 0-100 |
| `eligibility_status` | `eligible`, `warning`, `excluded` |
| `exclusion_reason` | Raison si exclu ou avertissement |

## Fonctionnalités Streamlit

### Scanner d'opportunités

La page d'accueil doit charger le Parquet de scan le plus récent et afficher instantanément un tableau classé. Elle doit permettre :

- sélection de la date et de l'émetteur ;
- filtre par type de trade, métrique, maturité et obligation à émettre ;
- filtre par qualité statistique : p-value, Hurst, half-life, z-score, statut OU, first hitting time et score composite ;
- tri par `confidence_score`, `target_gap_bps`, `sharpe_ratio`, `abs(zscore_12m)` ou `first_hitting_time_days` ;
- affichage des trois meilleures opportunités par obligation sélectionnée ;
- accès en un clic à la fiche de trade.

Utiliser `st.dataframe` avec configuration de colonnes et un tableau volontairement compact. Ne charger que les colonnes nécessaires au scan.

### Explorateur de trade

L'utilisateur choisit :

- le type de trade ;
- l'émetteur ou la paire d'émetteurs ;
- `bond1`, `bond2`, et lorsque pertinent `bond3`, `bond4` ;
- la métrique : YTM, ASW ou z-spread ;
- la fenêtre temporelle : 3m, 6m, 12m, historique complet.

L'écran doit afficher :

1. Une **fiche de décision** immédiatement visible : direction recommandée, spread marché, cible P-spline, moyenne OU, écart à la cible, z-scores, vol, half-life, espérance et écart-type de first hitting time, Sharpe, hedge ratio et exposition DV01 résiduelle.
2. Un graphe temporel du spread de marché et du spread fitté, avec cible, bandes ±1/±2 écarts-types ou quantiles historiques, et marqueurs de signal.
3. Une trajectoire d'espérance OU, à partir du spread courant, avec bandes de volatilité conditionnelle ±1 et ±2 écarts-types, sur les horizons 1 semaine, 1 mois, 3 mois, 6 mois et 12 mois.
4. Un graphe des valeurs sous-jacentes de chaque jambe lorsque cela aide à interpréter la structure.
5. Une distribution historique du spread avec position courante, cible P-spline et moyenne OU.
6. Un tableau de coefficients/nominaux par jambe.
7. Les diagnostics : cointégration, Hurst, paramètres OU MLE, fenêtre et statut de calibration, first hitting time, qualité de fit et statut VECM.

Prévoir l'export de la fiche au format CSV et Markdown, ainsi que l'export des séries utilisées au format CSV. L'export doit être simple, sans système externe.

### Diagnostics de courbe

Cette page doit permettre de sélectionner une date, un émetteur et une métrique, puis d'afficher :

- points de marché et P-spline fitté en fonction de la TTM ;
- résidus en bp par obligation ;
- RMSE/MAE, plage de maturités, nombre de points, nœuds, degré, ordre de pénalité et lambda ;
- identification visuelle des points extrapolés et outliers ;
- tableau des obligations utilisées ou exclues du fit.

Cette page est essentielle pour éviter de prendre pour une opportunité de RV un artefact de mauvais fit.

### Paramètres

La page paramètres ne doit pas nécessiter de persistance complexe. Elle peut afficher les paramètres de configuration utilisés par les scripts batch et permettre de télécharger un fichier de configuration YAML/JSON mis à jour si cela est utile.

Paramètres à exposer :

- fenêtre historique ;
- seuil de données minimales ;
- paramètres P-spline : `n_knots`, `degree`, `penalty_order`, `lambda_param` ;
- paramètres OU : `dt`, taille minimale d'échantillon, tolérances numériques, seuils de validité de `theta` et `sigma`, durée maximale de first hitting time, tolérance de proximité à `mu` ;
- seuils de cointégration et Hurst ;
- fenêtres de z-score ;
- fourchette de half-life admissible ;
- tolérance de DV01 résiduelle ;
- seuils d'entrée, de sortie et de stop pour le backtest.

## Performance

Respecter les principes suivants :

- `@st.cache_data` pour les lectures Parquet et les transformations déterministes.
- `@st.cache_resource` seulement pour ressources lourdes réellement réutilisées.
- Lecture sélective de colonnes et filtres `pyarrow.dataset` ou DuckDB avant conversion en pandas pour les gros historiques.
- Chargement par défaut de 12 mois de données ; historique complet uniquement sur demande.
- Pré-calcul intégral des combinaisons et métriques par les scripts batch.
- Pré-calcul et persistance des résultats OU, y compris first hitting time ; ne pas lancer d'intégrales `quad` au fil des clics utilisateur.
- Pagination ou limite configurable du tableau principal.
- Temps cible : scanner en moins de 2 secondes sur les fichiers locaux usuels ; page de détail en moins de 3 secondes si les séries sont déjà stockées.

## Qualité, validation et tests

Écrire des tests `pytest` couvrant au minimum :

- ingestion CSV et normalisation des unités ;
- déduplication et traitement des données invalides ;
- P-spline sur données synthétiques connues ;
- gestion de maturités hors domaine ;
- cohérence entre valeurs fittées, résidus et unités bp ;
- calcul de spread/fly/box et conventions de signe ;
- neutralité DV01 et hedge ratios ;
- calibration OU MLE sur des séries simulées avec paramètres connus ;
- calcul de l'espérance future OU et de la volatilité conditionnelle ;
- relation entre `theta` et half-life ;
- first hitting time : point de départ égal à `mu`, convergence des intégrales, non-finitudes et échecs d'intégration ;
- cointégration, Hurst et gestion des échecs d'estimation ;
- cohérence du score et des exclusions ;
- temps de lecture Parquet et temps de chargement des pages critiques.

Chaque résultat stocké doit pouvoir être reproduit à partir de sa date, de ses paramètres et de la version de données, sans nécessiter d'infrastructure de versioning externe.

## Livrables demandés au LLM architecte

Produis, dans cet ordre :

1. Une architecture détaillée et légère, avec diagramme Mermaid des flux de données.
2. L'arborescence finale du projet avec responsabilité de chaque module.
3. Le schéma exact des Parquet et les conventions d'unités.
4. Le code de production, documenté mais concis, des fonctions critiques :
   - ingestion CSV vers Parquet ;
   - classe P-spline et pipeline quotidien de fit ;
   - génération de spread/fly/box ;
   - classe `MeanReversionModel` OU fidèle aux formules MLE et de first hitting time spécifiées dans ce document ;
   - calcul de hedge ratio DV01 ;
   - lecture rapide et cache dans Streamlit.
5. Les pages Streamlit et leur navigation.
6. Les tests unitaires, tests d'intégration et tests de performance.
7. Un `requirements.txt` minimal.
8. Un `README.md` avec instructions locales : installation, placement des CSV, lancement des scripts batch et démarrage de Streamlit.

## Principes métier à préserver

- Le système est une aide à la décision, pas une boîte noire d'exécution automatique.
- Le spread de marché, le spread P-spline fitté et la moyenne long terme OU sont trois objets distincts et toujours affichés séparément.
- La qualité du fit de courbe conditionne l'interprétation des résidus et donc le classement RV.
- Les statistiques de mean reversion doivent être traitées comme des filtres de robustesse, non comme une promesse de convergence.
- Le modèle OU doit être fidèle aux formules MLE, d'espérance conditionnelle, de volatilité conditionnelle et de first hitting time définies dans ce document.
- Chaque trade doit pouvoir être expliqué en quelques secondes : legs, signe, ratio de nominal, niveau courant, cible, horizon, risque et raisons de la sélection.
- Priorité absolue à la rapidité, la lisibilité, la reproductibilité et la simplicité de maintenance.

**Fin du prompt.**
