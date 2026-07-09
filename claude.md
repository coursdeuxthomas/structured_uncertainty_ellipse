# Structured Uncertainty — Ellipses

Reproduction de l'expérience **« Ellipses »** de l'article
*Structured Uncertainty Prediction Networks* (Dorta et al., CVPR 2018,
arXiv:1802.07079, PDF à la racine du dépôt : `1802.07079v2.pdf`).

Ce dépôt est le **pendant « images 2D »** d'un projet frère sur les splines
(signaux 1D) : https://github.com/coursdeuxthomas/structured_uncertainty
On réutilise volontairement la **même architecture de fichiers** et les mêmes
conventions que le projet splines, pour pouvoir comparer les deux facilement.

## Objectif

Entraîner un réseau qui, à partir d'une image moyenne `mu`, prédit la
**matrice de covariance structurée** `Sigma(mu)` du résidu `epsilon = x - mu`,
où `x ~ N(mu, Sigma)`. On n'utilise jamais la vraie covariance à
l'entraînement : on optimise la log-vraisemblance négative (NLL) gaussienne
(Eq. 4 de l'article). La vraie covariance ne sert qu'en évaluation.

## Le modèle mathématique (identique aux splines)

- On modélise le résidu par une gaussienne pleine : `epsilon ~ N(0, Sigma)`.
- Le réseau prédit la **matrice de précision** `Lambda = Sigma^{-1}` via sa
  décomposition de Cholesky `Lambda = L L^T`, `L` triangulaire inférieure.
- Positivité : le réseau estime `log(l_ii)` sur la diagonale (puis `exp`),
  ce qui garantit `Lambda` définie positive.
- NLL (à minimiser), calculable sans inverser `Sigma` :
  ```
  loss = log|Sigma| + (x-mu)^T Lambda (x-mu)
       = -2 * sum_i log(l_ii) + || L^T (x-mu) ||^2
  ```

## Différence clé avec les splines : la parcimonie (sparse Cholesky)

- Splines : signal 1D de `n = 50` points → `L` dense, `n(n+1)/2 = 1275` valeurs.
- Ellipses : image `16×16`, donc `n = 256` pixels → un `L` dense demanderait
  `32 896` valeurs par image, infaisable à prédire (l'article n'y arrive pas).
- L'article impose donc un **motif de parcimonie fixe** sur `L` : `l_ij` n'est
  non nul que si `i >= j` **et** les pixels `i, j` sont voisins dans le plan
  image (patch `f×f`, `f=5` pour les ellipses). `L` est alors triangulaire
  inférieure ET bande-diagonale (cf. Fig. 2). Le réseau ne prédit que les
  valeurs non nulles.
- Interprétation : un zéro dans `Lambda` = indépendance conditionnelle entre
  deux pixels → champ gaussien markovien sur le résidu.

## Le dataset synthétique des ellipses (§5.1.2 + annexes)

- Images `16×16` en niveaux de gris. Pour chaque exemple :
  - `mu` = une **ellipse** de largeur, hauteur, position et **angle de rotation**
    `theta` aléatoires.
  - La **covariance prototype « génère des lignes »** (corrélations fortes le
    long d'une direction, faibles en travers), **tournée du même angle `theta`**
    que l'ellipse → des lignes de corrélation **alignées avec l'ellipse**.
  - `Sigma` est une fonction de `mu` (comme pour les splines où l'on met à
    l'échelle une covariance prototype par `|mu|`).
- On tire `epsilon ~ N(0, Sigma)` et `x = mu + epsilon`.
- Tailles article : 35 000 exemples d'entraînement, 1 000 de test.
- Hyperparamètres article : voisinage `5×5`, 200 epochs, lr `1e-3`, batch 64,
  données normalisées dans `[-1, 1]`.
- Note d'implémentation : à `n = 256`, on **ne stocke pas** les 35 000 matrices
  `256×256` (≈ 9 Go). On stocke `mu`/`x` et les paramètres (`theta`, `|mu|`)
  permettant de **reconstruire `Sigma` à la volée** pour l'évaluation.

## Architecture des fichiers (calquée sur le projet splines)

- `dataset.py` — génération du dataset synthétique (ellipses + covariance).
- `model.py`   — réseau de prédiction de la Cholesky (creuse) de la précision.
- `loss.py`    — `vector_to_lower_triangular` + `structured_gaussian_nll`.
- `train.py`   — boucle d'entraînement + validation + checkpoints.
- `eval.py`    — métriques (NLL, MSE de covariance) + figures.
- `main.py`    — orchestration (config, dataloaders, train/eval).
- `results/`   — sorties d'expériences (ignoré par git).
- `venv/`      — environnement virtuel local (ignoré par git).

## Conventions

- Commentaires et docstrings en **français** (comme le projet splines).
- Le réseau prend `mu` **aplati** `[B, n]` ; `model.py` le remet en image
  `[B, 1, 16, 16]` en interne si besoin (modèle convolutionnel). `x`, `mu` et
  le résidu restent des vecteurs `[B, n]` pour la loss.
- `L` encode la précision `Lambda`, PAS la covariance : ne pas confondre.

## Environnement / commandes

- Python via le venv local : `./venv/Scripts/python.exe`.
- Dépendances : `numpy`, `torch` (CPU), `matplotlib`.
- Lancer une expérience : `./venv/Scripts/python.exe main.py`.
- Tester un module isolément : `./venv/Scripts/python.exe dataset.py`.

## État d'avancement

- [x] `dataset.py` — dataset des ellipses synthétiques.
- [ ] `model.py` — réseau (Cholesky creuse, voisinage 5×5).
- [ ] `loss.py` — NLL adaptée à la Cholesky creuse.
- [ ] `train.py` / `eval.py` / `main.py`.
