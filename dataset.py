import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Génération de la moyenne mu : une ellipse
# ---------------------------------------------------------------------------
def generate_ellipse(
    image_size=16,
    semi_major_range=(4.0, 6.0),
    semi_minor_range=(2.0, 3.2),
    intensity_range=(0.7, 1.0),
    edge_softness=0.3,
):
    """
    Génère une image moyenne mu contenant une ellipse.

    L'ellipse a une largeur (demi-grand axe a), une hauteur (demi-petit axe b),
    une position (centre) et un angle de rotation theta aléatoires, comme dans
    l'article (§5.1.2).

    On force a > b (ellipse allongée) pour que l'orientation soit bien définie
    et donc apprenable à partir de mu.

    Retour :
        mu    : np.ndarray [image_size * image_size]  (image aplatie, valeurs dans [0, 1])
        theta : float, angle de rotation de l'ellipse (radians, dans [0, pi))

    Le fond vaut 0 et l'intérieur de l'ellipse vaut ~intensity. Le bord est
    légèrement adouci (anti-aliasing) pour que |mu| varie continûment.
    """
    S = image_size

    a = np.random.uniform(*semi_major_range)
    b = np.random.uniform(*semi_minor_range)
    theta = np.random.uniform(0.0, np.pi)
    intensity = np.random.uniform(*intensity_range)

    # Centre tiré dans la partie centrale de l'image (des ellipses partiellement
    # sorties du cadre sont acceptables et ajoutent de la variété).
    cx = np.random.uniform(0.30 * S, 0.70 * S)
    cy = np.random.uniform(0.30 * S, 0.70 * S)

    # Grille des coordonnées des pixels (col = x, row = y).
    rows, cols = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    X = cols - cx
    Y = rows - cy

    # Rotation des coordonnées dans le repère propre de l'ellipse (rotation de -theta).
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    Xr = X * cos_t + Y * sin_t
    Yr = -X * sin_t + Y * cos_t

    # Fonction implicite : <= 1 à l'intérieur de l'ellipse.
    f = (Xr / a) ** 2 + (Yr / b) ** 2

    # Bord adouci : plateau ~1 à l'intérieur, transition douce autour de f = 1.
    mask = 0.5 * (1.0 - np.tanh((f - 1.0) / edge_softness))

    mu = (intensity * mask).astype(np.float32)

    return mu.reshape(-1), float(theta)


# ---------------------------------------------------------------------------
# Covariance prototype : « génère des lignes », tournée par l'angle de l'ellipse
# ---------------------------------------------------------------------------
def pairwise_pixel_displacements(image_size=16):
    """
    Précalcule les déplacements pixel à pixel dans le plan image.

    Retour :
        dx, dy : np.ndarray [n, n] avec n = image_size**2
                 dx[i, j] = col_i - col_j, dy[i, j] = row_i - row_j

    Ces grilles ne dépendent pas de l'exemple : on les calcule une seule fois
    et on les réutilise pour construire toutes les covariances.
    """
    S = image_size
    rows, cols = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    rows = rows.reshape(-1).astype(np.float32)
    cols = cols.reshape(-1).astype(np.float32)

    dx = cols[:, None] - cols[None, :]
    dy = rows[:, None] - rows[None, :]

    return dx.astype(np.float32), dy.astype(np.float32)


def generate_line_covariance(dx, dy, theta, l_along=5.0, l_across=0.7):
    """
    Construit la covariance prototype « lignes », tournée de l'angle theta.

    Idée :
    - noyau gaussien anisotrope sur le déplacement entre deux pixels ;
    - corrélation forte LE LONG d'une direction (grande échelle l_along),
      faible EN TRAVERS (petite échelle l_across) ;
    - un tel noyau produit des échantillons en forme de lignes / stries,
      orientées selon la direction « along ».

    On tourne le repère de l'angle theta de l'ellipse, de sorte que les lignes
    soient alignées avec le grand axe de l'ellipse (comme dans l'article :
    « the prototype covariance ... is rotated by the same random rotation angle
    that was used for the ellipse »).

    Retour :
        K : np.ndarray [n, n], noyau symétrique semi-défini positif, K_ii = 1.
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Déplacement projeté dans le repère de la ligne (aligné avec l'ellipse).
    d_along = dx * cos_t + dy * sin_t
    d_across = -dx * sin_t + dy * cos_t

    K = np.exp(
        -0.5 * ((d_along / l_along) ** 2 + (d_across / l_across) ** 2)
    )

    return K.astype(np.float32)


def generate_covariance_from_mu(
    mu,
    theta,
    dx,
    dy,
    l_along=5.0,
    l_across=0.7,
    overall_var=0.04,
    min_scale=0.0,
    jitter=1e-3,
):
    """
    Sigma(mu) : covariance du résidu, fonction de la moyenne mu.

    Comme pour les splines (mise à l'échelle d'une covariance prototype par |mu|),
    on module ici la covariance « lignes » par l'amplitude locale de mu :

        Sigma_ij = overall_var * scale_i * scale_j * K_ij   (+ jitter sur la diagonale)

    avec scale_i = |mu_i| + min_scale et K le noyau « lignes » tourné de theta.

    Conséquence : le résidu corrélé (les lignes) se concentre là où l'ellipse
    est présente (|mu| grand) et disparaît sur le fond (|mu| ~ 0), tout en étant
    aligné avec l'ellipse. Le jitter garantit que Sigma est définie positive
    (Cholesky stable), y compris sur le fond.

    Retour :
        Sigma : np.ndarray [n, n], symétrique définie positive.
    """
    K = generate_line_covariance(dx, dy, theta, l_along=l_along, l_across=l_across)

    scale = np.abs(mu) + min_scale

    Sigma = overall_var * (scale[:, None] * scale[None, :]) * K
    Sigma += jitter * np.eye(len(mu), dtype=np.float32)

    return Sigma.astype(np.float32)


def sample_correlated_noise(Sigma):
    """
    Tire epsilon ~ N(0, Sigma) via la décomposition de Cholesky de Sigma.
    """
    z = np.random.randn(Sigma.shape[0]).astype(np.float32)
    L = np.linalg.cholesky(Sigma.astype(np.float64)).astype(np.float32)
    eps = L @ z
    return eps.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class EllipseDataset(Dataset):
    """
    Dataset synthétique des ellipses (§5.1.2 de l'article) :

        mu         : image 16x16 (aplatie) contenant une ellipse
        x          : mu + bruit corrélé epsilon,  x ~ N(mu, Sigma)
        Sigma_true : covariance ayant servi à générer epsilon (pour l'éval)

    À l'entraînement, on n'utilise que mu et x. Sigma_true ne sert qu'à
    l'évaluation, et n'est donc renvoyée que si return_sigma=True.

    Mémoire : à n = 256, stocker les 35 000 covariances 256x256 coûterait ~9 Go.
    On ne stocke donc que mu, x, l'angle theta et le facteur d'échelle |mu|
    (implicite via mu), ce qui suffit à reconstruire Sigma_true à la volée
    (méthode _build_sigma).

    resample_noise :
        False -> x est fixé une fois à la construction (val / test).
        True  -> un nouvel epsilon est tiré à chaque __getitem__ (train),
                 en reconstruisant Sigma puis sa Cholesky. Plus coûteux
                 (une Cholesky 256x256 par accès), donc désactivé par défaut.

    return_sigma :
        True  -> __getitem__ renvoie aussi Sigma_true (reconstruite à la volée).
                 À activer pour les jeux de validation / test.
    """

    def __init__(
        self,
        num_samples=35000,
        image_size=16,
        semi_major_range=(4.0, 6.0),
        semi_minor_range=(2.0, 3.2),
        intensity_range=(0.7, 1.0),
        edge_softness=0.3,
        l_along=5.0,
        l_across=0.7,
        overall_var=0.04,
        min_scale=0.0,
        jitter=1e-3,
        seed=None,
        resample_noise=False,
        return_sigma=False,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_pixels = image_size * image_size

        self.l_along = l_along
        self.l_across = l_across
        self.overall_var = overall_var
        self.min_scale = min_scale
        self.jitter = jitter

        self.resample_noise = resample_noise
        self.return_sigma = return_sigma

        if seed is not None:
            np.random.seed(seed)

        # Déplacements pixel à pixel, calculés une seule fois.
        self.dx, self.dy = pairwise_pixel_displacements(image_size)

        mus = []
        xs = []
        thetas = []

        for _ in range(num_samples):
            mu, theta = generate_ellipse(
                image_size=image_size,
                semi_major_range=semi_major_range,
                semi_minor_range=semi_minor_range,
                intensity_range=intensity_range,
                edge_softness=edge_softness,
            )

            Sigma_true = generate_covariance_from_mu(
                mu=mu,
                theta=theta,
                dx=self.dx,
                dy=self.dy,
                l_along=l_along,
                l_across=l_across,
                overall_var=overall_var,
                min_scale=min_scale,
                jitter=jitter,
            )

            eps = sample_correlated_noise(Sigma_true)
            x = mu + eps

            mus.append(mu)
            xs.append(x.astype(np.float32))
            thetas.append(theta)

        self.mus = torch.from_numpy(np.stack(mus).astype(np.float32))
        self.xs = torch.from_numpy(np.stack(xs).astype(np.float32))
        # theta et mu suffisent à reconstruire Sigma_true à la volée.
        self.thetas = torch.tensor(thetas, dtype=torch.float32)

    def _build_sigma(self, idx):
        """
        Reconstruit Sigma_true pour l'exemple idx (retour Tensor [n, n]).
        """
        mu = self.mus[idx].numpy()
        theta = float(self.thetas[idx])

        Sigma = generate_covariance_from_mu(
            mu=mu,
            theta=theta,
            dx=self.dx,
            dy=self.dy,
            l_along=self.l_along,
            l_across=self.l_across,
            overall_var=self.overall_var,
            min_scale=self.min_scale,
            jitter=self.jitter,
        )

        return torch.from_numpy(Sigma)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mu = self.mus[idx]

        if self.resample_noise:
            Sigma = self._build_sigma(idx)
            L = torch.linalg.cholesky(Sigma.double())
            z = torch.randn(self.num_pixels, dtype=torch.float64)
            eps = (L @ z).to(torch.float32)
            x = mu + eps
        else:
            x = self.xs[idx]

        item = {"mu": mu, "x": x}

        if self.return_sigma:
            item["Sigma_true"] = self._build_sigma(idx)

        return item


if __name__ == "__main__":
    # Petit jeu de test : on vérifie les formes, les statistiques et la
    # définie-positivité, puis on sauvegarde un aperçu visuel.
    dataset = EllipseDataset(num_samples=8, seed=0, return_sigma=True)

    sample = dataset[0]

    print("num pixels (n)      :", dataset.num_pixels)
    print("mu shape            :", tuple(sample["mu"].shape))
    print("x shape             :", tuple(sample["x"].shape))
    print("Sigma_true shape    :", tuple(sample["Sigma_true"].shape))
    print("mu range            : [%.3f, %.3f]" % (sample["mu"].min(), sample["mu"].max()))
    print("residual (x-mu) std : %.4f" % (sample["x"] - sample["mu"]).std().item())

    # Vérifications de cohérence sur Sigma_true.
    Sigma = sample["Sigma_true"].numpy()
    sym_err = np.abs(Sigma - Sigma.T).max()
    eigmin = np.linalg.eigvalsh(Sigma).min()
    print("Sigma symmetry err  : %.2e" % sym_err)
    print("Sigma min eigenvalue: %.2e (doit etre > 0)" % eigmin)

    # Aperçu visuel (facultatif, nécessite matplotlib).
    try:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs("results", exist_ok=True)
        S = dataset.image_size
        n_show = 4

        fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 9))
        for k in range(n_show):
            s = dataset[k]
            mu_img = s["mu"].numpy().reshape(S, S)
            x_img = s["x"].numpy().reshape(S, S)
            res_img = x_img - mu_img

            axes[0, k].imshow(mu_img, cmap="gray")
            axes[0, k].set_title(f"mu #{k}")
            axes[1, k].imshow(x_img, cmap="gray")
            axes[1, k].set_title("x = mu + eps")
            axes[2, k].imshow(res_img, cmap="RdBu")
            axes[2, k].set_title("residu eps")
            for row in range(3):
                axes[row, k].axis("off")

        plt.tight_layout()
        out = os.path.join("results", "dataset_preview.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print("apercu sauvegarde  :", out)
    except Exception as e:
        print("apercu non genere (matplotlib indisponible ?) :", e)
