"""
Loss NLL gaussienne structurée pour les ellipses (Cholesky CREUSE de la précision).

Pendant « images 2D » de la loss des splines : là-bas `L` est dense et
reconstruite depuis un vecteur (`vector_to_lower_triangular`). Ici l'image
`16x16` donne `n = 256` pixels ; un `L` dense (32 896 valeurs) est infaisable,
donc on impose le motif de parcimonie de l'article (Dorta et al., 2018, §5.1) :

    L[i, j] != 0  seulement si  i >= j  ET  les pixels i, j sont voisins
    dans un patch f×f (f = 5).

Concrètement, pour chaque pixel `i`, `L` n'a de valeurs non nulles que sur :
- la diagonale `L[i, i]` (paramétrée par `log(l_ii)` -> `exp` garantit > 0) ;
- les `L[i, j]` pour `j` voisin CAUSAL de `i` (j vient avant i en ordre
  raster ET dans le patch f×f). Pour f = 5 il y a 12 voisins causaux.

Le réseau ne prédit donc que `1 + 12 = 13` valeurs par pixel (cf. `model.py`).

La précision est `Lambda = L L^T` (symétrique définie positive par
construction). La NLL gaussienne (Eq. 4 de l'article) se calcule SANS inverser
Sigma :

    nll = 0.5 * [ log|Sigma| + (x-mu)^T Lambda (x-mu) + n*log(2*pi) ]
        = 0.5 * [ -2*sum_i log(l_ii) + || L^T (x-mu) ||^2 + n*log(2*pi) ]

et `L^T (x-mu)` se calcule directement à partir des valeurs non nulles (pas de
matrice n×n en mémoire), via un `scatter_add`.
"""

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Motif de parcimonie : voisins causaux dans un patch f×f
# ---------------------------------------------------------------------------
def causal_offsets(f=5):
    """
    Décalages (dr, dc) des voisins CAUSAUX (hors pixel courant) d'un patch f×f.

    Un voisin `j = i + (dr, dc)` est causal si son indice raster est < celui de
    `i`, c.-à-d. `dr < 0`, ou (`dr == 0` et `dc < 0`). En ordre raster
    (index = row * S + col), cela garantit que `L` est bien triangulaire
    inférieure.

    Pour f = 5 (h = 2) il y a 12 décalages :
    - dr in {-2, -1}, dc in {-2, -1, 0, 1, 2}  -> 10
    - dr == 0,        dc in {-2, -1}           ->  2

    Retour :
        offsets : list[(dr, dc)] de longueur (f*f - 1) // 2
    """
    h = f // 2
    offsets = []
    for dr in range(-h, 1):
        for dc in range(-h, h + 1):
            if dr < 0 or (dr == 0 and dc < 0):
                offsets.append((dr, dc))
    return offsets


def build_neighbor_indices(image_size=16, f=5, device=None):
    """
    Précalcule, pour chaque pixel `i`, l'indice raster de ses voisins causaux et
    un masque de validité (pixels hors image = invalides).

    Retour :
        neighbor_idx : LongTensor [n, m]  (m = nb de voisins causaux, 12 pour f=5)
                       neighbor_idx[i, k] = indice raster du k-ième voisin causal
                       de i, ou `i` lui-même si le voisin sort de l'image (valeur
                       neutralisée par le masque, la diagonale étant écrasée
                       séparément).
        mask         : FloatTensor [n, m], 1.0 si le voisin est dans l'image.

    Ces tenseurs ne dépendent pas de l'exemple : on les calcule une fois et on
    les réutilise pour tout le batch / toute l'expérience.
    """
    S = image_size
    n = S * S
    offsets = causal_offsets(f)
    m = len(offsets)

    neighbor_idx = np.zeros((n, m), dtype=np.int64)
    mask = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        r, c = divmod(i, S)
        for k, (dr, dc) in enumerate(offsets):
            rr, cc = r + dr, c + dc
            if 0 <= rr < S and 0 <= cc < S:
                neighbor_idx[i, k] = rr * S + cc
                mask[i, k] = 1.0
            else:
                # Voisin hors image : on le renvoie sur `i` (col quelconque) et on
                # le neutralise via le masque (valeur 0 -> contribution nulle).
                neighbor_idx[i, k] = i
                mask[i, k] = 0.0

    neighbor_idx = torch.from_numpy(neighbor_idx)
    mask = torch.from_numpy(mask)
    if device is not None:
        neighbor_idx = neighbor_idx.to(device)
        mask = mask.to(device)
    return neighbor_idx, mask


# ---------------------------------------------------------------------------
# Produit L^T r sans matrice dense
# ---------------------------------------------------------------------------
def apply_LT(log_diag, offdiag, residual, neighbor_idx, mask):
    """
    Calcule `w = L^T r` à partir des valeurs non nulles de `L`, sans jamais
    construire la matrice n×n.

    Rappel : `L[i, i] = exp(log_diag[i])` et `L[i, neighbor_idx[i, k]] = offdiag[i, k]`.
    On a `w_c = (L^T r)_c = sum_i L[i, c] r_i`, d'où :
    - diagonale : `w_i += exp(log_diag_i) * r_i` ;
    - hors-diag : chaque `L[i, j] * r_i` s'ajoute à `w_j` (scatter_add sur j).

    Args :
        log_diag     : [B, n]      valeurs log de la diagonale de L.
        offdiag      : [B, n, m]   valeurs hors-diagonale (une par voisin causal).
        residual     : [B, n]      r = x - mu.
        neighbor_idx : [n, m]      indices des voisins causaux (cf. build_neighbor_indices).
        mask         : [n, m]      masque de validité des voisins.

    Retour :
        w : [B, n],  w = L^T r.
    """
    B, n = residual.shape
    m = offdiag.shape[2]

    # Terme diagonal.
    w = torch.exp(log_diag) * residual  # [B, n]

    # Termes hors-diagonale : contribution de chaque pixel i à ses voisins j.
    contrib = offdiag * mask.unsqueeze(0) * residual.unsqueeze(2)  # [B, n, m]

    # scatter_add sur la dimension des pixels : w[b, neighbor_idx[i, k]] += contrib[b, i, k].
    idx = neighbor_idx.reshape(1, n * m).expand(B, n * m)
    w = w.scatter_add(1, idx, contrib.reshape(B, n * m))
    return w


# ---------------------------------------------------------------------------
# NLL gaussienne structurée
# ---------------------------------------------------------------------------
def structured_gaussian_nll(
    log_diag,
    offdiag,
    residual,
    neighbor_idx,
    mask,
    include_const=True,
    mean_batch=True,
):
    """
    NLL gaussienne (Eq. 4 de l'article) pour la précision structurée creuse.

        nll = 0.5 * [ -2*sum_i log(l_ii) + ||L^T r||^2 + n*log(2*pi) ]

    avec `log(l_ii) = log_diag_i` (donc `log|Sigma| = -2 sum_i log_diag_i`) et
    `||L^T r||^2 = r^T Lambda r`. Aucune inversion, aucune matrice n×n.

    Args :
        log_diag, offdiag : sorties du réseau ([B, n] et [B, n, m]).
        residual          : r = x - mu, [B, n].
        neighbor_idx, mask: motif de parcimonie (cf. build_neighbor_indices).
        include_const     : ajoute le terme constant n*log(2*pi) (vraie NLL).
        mean_batch        : moyenne sur le batch (sinon retour par exemple [B]).

    Retour :
        nll : scalaire (mean_batch=True) ou [B].
    """
    n = residual.shape[1]

    w = apply_LT(log_diag, offdiag, residual, neighbor_idx, mask)  # [B, n]

    quad = (w ** 2).sum(dim=1)                   # ||L^T r||^2, [B]
    log_det_sigma = -2.0 * log_diag.sum(dim=1)   # log|Sigma|, [B]

    nll = log_det_sigma + quad
    if include_const:
        nll = nll + n * math.log(2.0 * math.pi)
    nll = 0.5 * nll

    if mean_batch:
        return nll.mean()
    return nll


# ---------------------------------------------------------------------------
# Reconstruction dense (uniquement pour l'évaluation, jamais pour l'entraînement)
# ---------------------------------------------------------------------------
def build_L_dense(log_diag, offdiag, neighbor_idx, mask):
    """
    Reconstruit la matrice de Cholesky creuse `L` sous forme DENSE [B, n, n].

    Réservé à l'évaluation (comparaison de covariances) : à n = 256 la matrice
    dense est petite (256×256), mais on ne l'utilise jamais dans la boucle
    d'entraînement (cf. `apply_LT`).

    Retour :
        L : [B, n, n], triangulaire inférieure, diagonale > 0.
    """
    B, n = log_diag.shape
    m = offdiag.shape[2]
    device = log_diag.device

    L = torch.zeros(B, n, n, dtype=log_diag.dtype, device=device)

    # Hors-diagonale : L[b, i, neighbor_idx[i, k]] += offdiag[b, i, k] (masqué).
    rows = torch.arange(n, device=device).view(n, 1).expand(n, m).reshape(-1)  # [n*m]
    cols = neighbor_idx.reshape(-1)                                            # [n*m]
    vals = (offdiag * mask.unsqueeze(0)).reshape(B, n * m)                     # [B, n*m]

    bidx = torch.arange(B, device=device).view(B, 1).expand(B, n * m).reshape(-1)
    ridx = rows.view(1, -1).expand(B, -1).reshape(-1)
    cidx = cols.view(1, -1).expand(B, -1).reshape(-1)
    L.index_put_((bidx, ridx, cidx), vals.reshape(-1), accumulate=True)

    # Diagonale : écrase toute valeur parasite (les voisins hors-image pointent
    # sur i, mais leur valeur masquée est nulle -> pas d'effet ici).
    diag = torch.exp(log_diag)  # [B, n]
    idx = torch.arange(n, device=device)
    L[:, idx, idx] = diag
    return L


def predicted_precision_and_covariance(log_diag, offdiag, neighbor_idx, mask):
    """
    Renvoie (Lambda, Sigma) prédites à partir des sorties du réseau (éval).

        Lambda = L L^T           (précision)
        Sigma  = Lambda^{-1}     (covariance)

    Retour :
        Lambda : [B, n, n]
        Sigma  : [B, n, n]
    """
    L = build_L_dense(log_diag, offdiag, neighbor_idx, mask)
    Lambda = L @ L.transpose(1, 2)
    Sigma = torch.linalg.inv(Lambda)
    return Lambda, Sigma


if __name__ == "__main__":
    # Test unitaire : cohérence entre apply_LT (creux) et le calcul dense, et
    # validité de la précision (définie positive), sur des valeurs aléatoires.
    torch.manual_seed(0)
    S, f, B = 16, 5, 4
    n = S * S
    neighbor_idx, mask = build_neighbor_indices(S, f)
    m = neighbor_idx.shape[1]

    print("n =", n, "| voisins causaux m =", m, "(attendu 12 pour f=5)")

    log_diag = 0.1 * torch.randn(B, n)
    offdiag = 0.1 * torch.randn(B, n, m)
    r = torch.randn(B, n)

    # 1) L^T r creux == L^T r dense.
    w_sparse = apply_LT(log_diag, offdiag, r, neighbor_idx, mask)
    L = build_L_dense(log_diag, offdiag, neighbor_idx, mask)
    w_dense = torch.bmm(L.transpose(1, 2), r.unsqueeze(2)).squeeze(2)
    err = (w_sparse - w_dense).abs().max().item()
    print("erreur max apply_LT vs dense : %.2e (doit etre ~0)" % err)

    # 2) L est bien triangulaire inférieure avec diagonale exp(log_diag).
    upper = torch.triu(L, diagonal=1).abs().max().item()
    diag_err = (torch.diagonal(L, dim1=1, dim2=2) - torch.exp(log_diag)).abs().max().item()
    print("masse triangle superieur : %.2e (doit etre 0)" % upper)
    print("erreur diagonale         : %.2e (doit etre 0)" % diag_err)

    # 3) Lambda = L L^T definie positive.
    Lambda, Sigma = predicted_precision_and_covariance(log_diag, offdiag, neighbor_idx, mask)
    eigmin = torch.linalg.eigvalsh(Lambda)[..., 0].min().item()
    print("valeur propre min de Lambda : %.3e (doit etre > 0)" % eigmin)

    # 4) NLL dense de reference vs structurée.
    nll_struct = structured_gaussian_nll(log_diag, offdiag, r, neighbor_idx, mask)
    log_det_sigma = torch.logdet(Sigma)
    quad = torch.einsum("bi,bij,bj->b", r, Lambda, r)
    nll_ref = 0.5 * (log_det_sigma + quad + n * math.log(2 * math.pi))
    print("NLL structuree : %.4f | NLL dense ref : %.4f" % (nll_struct.item(), nll_ref.mean().item()))
