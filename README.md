# Structured Uncertainty — Ellipses (PyTorch)

Reproduction de l'expérience **« Ellipses »** de *Structured Uncertainty
Prediction Networks* (Dorta et al., CVPR 2018, arXiv:1802.07079).

À partir d'une image moyenne `mu` (une ellipse), un réseau prédit la **matrice de
précision structurée** `Lambda(mu) = L Lᵀ` du résidu `epsilon = x - mu`, via la
Cholesky **creuse** `L`. On n'optimise que la **NLL gaussienne** (Eq. 4) ; la
vraie covariance ne sert qu'à l'évaluation.

## Modèle

- Résidu gaussien `epsilon ~ N(0, Sigma)`, `Sigma = Lambda⁻¹`.
- `L` triangulaire inférieure, **bande-diagonale** : `L[i,j] ≠ 0` seulement si
  `j` est un voisin **causal** de `i` dans un patch `f×f` (`f = 5`). Le réseau
  prédit `1 + 12 = 13` valeurs par pixel (diagonale en `log`, 12 hors-diag).
- NLL (sans jamais inverser `Sigma`) :
  `loss = -2·Σᵢ log(lᵢᵢ) + ‖Lᵀ(x-mu)‖²  (+ n·log 2π)`.
  Le produit `Lᵀr` est calculé par `scatter_add` sur les seules valeurs non
  nulles — aucune matrice `n×n` à l'entraînement.

## Fichiers

| Fichier      | Rôle |
|--------------|------|
| `dataset.py` | Ellipses synthétiques + covariance « lignes » tournée de `theta`. |
| `loss.py`    | Motif de parcimonie, `apply_LT`, `structured_gaussian_nll`, reconstruction dense (éval). |
| `model.py`   | `SparseCholeskyNet` : petit U-Net `16×16 → 13` canaux. |
| `train.py`   | Boucle train/val + checkpoints. |
| `eval.py`    | Métriques (NLL, erreur de Frobenius, KL) + figures. |
| `main.py`    | Orchestration (config, datasets, train, éval). |

## Utilisation

```bash
# Config de l'article (35 000 exemples, 200 epochs) — long sur CPU (~qq heures).
./venv/Scripts/python.exe main.py

# Run rapide de bout en bout (validation du pipeline).
./venv/Scripts/python.exe main.py --smoke

# Nommer le run, surcharger le nombre d'epochs, ou évaluer un checkpoint.
./venv/Scripts/python.exe main.py --tag exp1 --epochs 50
./venv/Scripts/python.exe main.py --eval-only results/<run>/best_model.pt

# Tests unitaires par module.
./venv/Scripts/python.exe loss.py
./venv/Scripts/python.exe model.py
./venv/Scripts/python.exe dataset.py
```

### Organisation des résultats

Chaque exécution crée un dossier de run **isolé et horodaté**, jamais écrasé :

```
results/
  <tag>_<AAAAMMJJ-HHMMSS>/     # ex. article_20260720-153000/  (--tag pour le préfixe)
    best_model.pt              # meilleur checkpoint (NLL val)
    learning_curve.png         # courbes NLL train/val
    eval_correlation_maps.png  # cartes de corrélation vraies vs prédites
    eval_samples.png           # échantillons de bruit vrai vs prédit
    metrics.json               # config + métriques finales de test
```

Le préfixe par défaut est `smoke` (avec `--smoke`) ou `article` sinon.

## Métriques d'évaluation

- **NLL** — vraisemblance négative gaussienne structurée moyenne.
- **Frobenius relatif** — `‖Sigma_pred − Sigma_true‖_F / ‖Sigma_true‖_F`.
- **KL** — `KL(N(0,Sigma_true) ‖ N(0,Sigma_pred))`.

## Environnement

Venv local (`./venv/Scripts/python.exe`), dépendances : `numpy`, `torch` (CPU),
`matplotlib`. La référence TensorFlow `references/tf_mvg/` n'est utilisée que
comme inspiration pour la paramétrisation ; tout le code du projet est en PyTorch.
