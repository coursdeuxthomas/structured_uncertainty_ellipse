"""
Réseau de prédiction de la Cholesky CREUSE de la précision (ellipses).

Entrée  : la moyenne `mu` aplatie `[B, n]` (n = 16*16 = 256).
Sortie  : les seules valeurs non nulles de `L` (Cholesky de la précision
          Lambda = L L^T), pour le motif de parcimonie 5×5 de `loss.py` :
          - `log_diag` : `[B, n]`     -> diagonale `l_ii = exp(log_diag)` (> 0) ;
          - `offdiag`  : `[B, n, m]`  -> m = 12 valeurs hors-diag par pixel.

Comme l'article (Dorta et al., 2018), le réseau est CONVOLUTIONNEL : `mu` est
remis en image `[B, 1, 16, 16]`, passé dans un petit encodeur-décodeur, puis une
convolution 1×1 finale produit `1 + m = 13` canaux par pixel. Le canal 0 est
`log_diag`, les `m` canaux suivants sont les `offdiag`.

On travaille bien en PyTorch (la référence `references/tf_mvg` est en TensorFlow
et ne sert que d'inspiration pour la paramétrisation).
"""

import torch
import torch.nn as nn

from loss import causal_offsets


class ConvBlock(nn.Module):
    """Bloc conv 3×3 (padding 1) + BatchNorm + activation, taille d'image conservée."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SparseCholeskyNet(nn.Module):
    """
    Réseau convolutionnel prédisant les valeurs non nulles de la Cholesky creuse
    de la précision.

    Architecture (petit U-Net symétrique, images 16×16) :
        16×16 -> encodeur -> 8×8 (bottleneck) -> décodeur -> 16×16
    avec connexions de saut (skip) pour préserver les détails de position/bord de
    l'ellipse, qui conditionnent l'orientation locale des corrélations.

    Args :
        image_size : côté de l'image (16).
        f          : taille du voisinage (5) -> m = (f*f - 1)//2 canaux hors-diag.
        base_ch    : nombre de canaux de base.
        init_log_diag : biais initial du canal `log_diag` (contrôle l'échelle de
                        la précision au démarrage ; ~ -0.5*log(var) attendu).
    """

    def __init__(self, image_size=16, f=5, base_ch=32, init_log_diag=1.5):
        super().__init__()
        self.image_size = image_size
        self.f = f
        self.n = image_size * image_size
        self.m = len(causal_offsets(f))       # 12 pour f = 5
        self.out_ch = 1 + self.m              # 1 (log_diag) + m (offdiag)

        c = base_ch

        # Encodeur.
        self.enc1 = nn.Sequential(ConvBlock(1, c), ConvBlock(c, c))               # 16×16
        self.pool = nn.MaxPool2d(2)                                               # -> 8×8
        self.enc2 = nn.Sequential(ConvBlock(c, 2 * c), ConvBlock(2 * c, 2 * c))   # 8×8

        # Décodeur (upsample + concat skip).
        self.up = nn.Upsample(scale_factor=2, mode="nearest")                     # -> 16×16
        self.dec1 = nn.Sequential(ConvBlock(2 * c + c, c), ConvBlock(c, c))       # 16×16

        # Tête de sortie : conv 1×1 -> 13 canaux.
        self.head = nn.Conv2d(c, self.out_ch, kernel_size=1)

        # Initialisation : sortie ~ 0 partout, sauf un biais sur log_diag pour
        # démarrer avec une précision d'échelle raisonnable (évite d'exploser).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[0] = init_log_diag

    def forward(self, mu):
        """
        Args :
            mu : `[B, n]` (moyenne aplatie) ou `[B, 1, S, S]` (déjà en image).

        Retour :
            log_diag : `[B, n]`
            offdiag  : `[B, n, m]`
        """
        if mu.dim() == 2:
            B = mu.shape[0]
            x = mu.view(B, 1, self.image_size, self.image_size)
        else:
            B = mu.shape[0]
            x = mu

        e1 = self.enc1(x)                 # [B, c, 16, 16]
        e2 = self.enc2(self.pool(e1))     # [B, 2c, 8, 8]
        d = self.up(e2)                   # [B, 2c, 16, 16]
        d = torch.cat([d, e1], dim=1)     # skip connection
        d = self.dec1(d)                  # [B, c, 16, 16]
        out = self.head(d)                # [B, 13, 16, 16]

        out = out.view(B, self.out_ch, self.n)   # [B, 13, n]
        log_diag = out[:, 0, :]                   # [B, n]
        offdiag = out[:, 1:, :].transpose(1, 2)   # [B, n, m]
        return log_diag, offdiag


if __name__ == "__main__":
    # Test : formes de sortie et rétropropagation d'une NLL structurée.
    from loss import build_neighbor_indices, structured_gaussian_nll

    torch.manual_seed(0)
    S, f, B = 16, 5, 8
    n = S * S

    net = SparseCholeskyNet(image_size=S, f=f)
    n_params = sum(p.numel() for p in net.parameters())
    print("parametres du reseau :", n_params)

    mu = torch.rand(B, n)
    log_diag, offdiag = net(mu)
    print("log_diag :", tuple(log_diag.shape), "| offdiag :", tuple(offdiag.shape))
    assert log_diag.shape == (B, n)
    assert offdiag.shape == (B, n, net.m)

    neighbor_idx, mask = build_neighbor_indices(S, f)
    x = mu + 0.1 * torch.randn(B, n)
    loss = structured_gaussian_nll(log_diag, offdiag, x - mu, neighbor_idx, mask)
    loss.backward()

    grad_norm = sum(p.grad.abs().sum() for p in net.parameters() if p.grad is not None)
    print("NLL initiale : %.4f | somme |grad| : %.4f" % (loss.item(), grad_norm.item()))
    print("OK : forward/backward fonctionnels.")
