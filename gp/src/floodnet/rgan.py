"""R-GAN for the class-imbalance study, after "Advanced R-GAN" (Lee, Jung, Moon & Rho, Alexandria Eng. J. 111, 2025).

The paper's abstract names three components: spectral normalisation in the generator, a similarity-measure loss,
and CELU activations in the discriminator. Only the abstract was readable, so the exact formulation is assumed
(documented in docs/imbalance.md). Two uses:

* image R-GAN: a pix2pix-style mask-to-image generator (Isola et al. 2016) with those three components and the
  relativistic-average adversarial loss (Jolicoeur-Martineau 2018) - renders edited layouts for the U-Net/RF;
* tabular R-GAN: the paper's own setting (minority rows), generating 22-number superpixel rows for the RF.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from .gan_common import (
    CONDITION_CHANNELS,
    diff_augment,
    relativistic_d_loss,
    relativistic_g_loss,
    similarity_loss,
)

# ---------- image R-GAN ----------


class Down(nn.Module):
    def __init__(self, cin, cout, norm=True):
        super().__init__()
        layers = [spectral_norm(nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm))]
        if norm:
            layers.append(nn.InstanceNorm2d(cout, affine=True))
        layers.append(nn.LeakyReLU(.2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, cin, cout, dropout=False):
        super().__init__()
        layers = [spectral_norm(nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False)), nn.InstanceNorm2d(cout, affine=True), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.net(x), skip], 1)


class ImageGenerator(nn.Module):
    """8-level U-Net generator (256 -> 1 -> 256) with spectral normalisation on every convolution."""
    def __init__(self, ngf=48):
        super().__init__()
        widths = [ngf, ngf * 2, ngf * 4, ngf * 8, ngf * 8, ngf * 8, ngf * 8, ngf * 8]
        self.downs = nn.ModuleList([Down(CONDITION_CHANNELS, widths[0], norm=False)]
                                   + [Down(widths[i - 1], widths[i], norm=i < 7) for i in range(1, 8)])
        ups = []
        for i in range(7, 0, -1):
            cin = widths[i] if i == 7 else widths[i] * 2
            ups.append(Up(cin, widths[i - 1], dropout=i >= 5))
        self.ups = nn.ModuleList(ups)
        self.out = nn.Sequential(spectral_norm(nn.ConvTranspose2d(widths[0] * 2, 3, 4, 2, 1)), nn.Tanh())

    def forward(self, cond):
        if cond.shape[1:] != (CONDITION_CHANNELS, 256, 256):
            raise ValueError(f"Expected (B, {CONDITION_CHANNELS}, 256, 256) layouts, got {tuple(cond.shape)}")
        skips, x = [], cond
        for down in self.downs:
            x = down(x)
            skips.append(x)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        return self.out(x)


class ImageDiscriminator(nn.Module):
    """Conditional 70x70 PatchGAN with spectral normalisation and CELU activations."""
    def __init__(self, ndf=64):
        super().__init__()
        cin = 3 + CONDITION_CHANNELS
        self.blocks = nn.ModuleList([
            nn.Sequential(spectral_norm(nn.Conv2d(cin, ndf, 4, 2, 1)), nn.CELU()),
            nn.Sequential(spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)), nn.CELU()),
            nn.Sequential(spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)), nn.CELU()),
            nn.Sequential(spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1)), nn.CELU()),
        ])
        self.head = spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 1))

    def forward(self, image, cond):
        x = torch.cat([image, cond], 1)
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return self.head(x), features


def build(settings):
    return ImageGenerator(settings["ngf"]), ImageDiscriminator(settings["ndf"])


def step(generator, discriminator, opt_g, opt_d, real, cond, features, settings, sampler):
    fake = generator(cond)
    # Discriminator: relativistic average loss on DiffAugmented real and fake pairs.
    real_aug, cond_r = diff_augment(real, cond, sampler)
    fake_aug, cond_f = diff_augment(fake.detach(), cond, sampler)
    d_real, _ = discriminator(real_aug, cond_r)
    d_fake, _ = discriminator(fake_aug, cond_f)
    loss_d = relativistic_d_loss(d_real, d_fake)
    opt_d.zero_grad(set_to_none=True)
    loss_d.backward()
    opt_d.step()
    # Generator: relativistic adversarial + L1 + similarity-measure loss + discriminator feature matching.
    fake_aug, cond_f = diff_augment(fake, cond, sampler)
    with torch.no_grad():
        real_aug, cond_r = diff_augment(real, cond, sampler)
        d_real, _ = discriminator(real_aug, cond_r)
        _, real_features = discriminator(real, cond)
    d_fake, _ = discriminator(fake_aug, cond_f)
    _, fake_features = discriminator(fake, cond)
    adversarial = relativistic_g_loss(d_real, d_fake)
    l1 = (fake - real).abs().mean()
    similarity = similarity_loss(features, fake, real)
    matching = sum((a - b.detach()).abs().mean() for a, b in zip(fake_features, real_features))
    loss_g = adversarial + settings["lambda_l1"] * l1 + settings["lambda_similarity"] * similarity + settings["lambda_feature_matching"] * matching
    opt_g.zero_grad(set_to_none=True)
    loss_g.backward()
    opt_g.step()
    return {"d": loss_d.detach().item(), "g_adv": adversarial.detach().item(), "l1": l1.detach().item(), "similarity": similarity.detach().item(), "fm": matching.detach().item()}


# ---------- tabular R-GAN (22-number superpixel rows) ----------

COLOUR_DIMS, TEXTURE_DIMS = 12, 10


class TabularGenerator(nn.Module):
    """Class-conditional MLP with spectral normalisation. Output respects the feature ranges: 12 colour statistics
    in [0, 1] (sigmoid) and a 10-bin LBP histogram that sums to 1 (softmax)."""
    def __init__(self, noise=32, hidden=128, classes=2):
        super().__init__()
        self.noise, self.classes = noise, classes
        self.net = nn.Sequential(spectral_norm(nn.Linear(noise + classes, hidden)), nn.LeakyReLU(.2),
                                 spectral_norm(nn.Linear(hidden, hidden)), nn.LeakyReLU(.2),
                                 spectral_norm(nn.Linear(hidden, COLOUR_DIMS + TEXTURE_DIMS)))

    def forward(self, z, label):
        x = self.net(torch.cat([z, nn.functional.one_hot(label, self.classes).float()], 1))
        return torch.cat([torch.sigmoid(x[:, :COLOUR_DIMS]), torch.softmax(x[:, COLOUR_DIMS:], 1)], 1)


class TabularDiscriminator(nn.Module):
    def __init__(self, hidden=128, classes=2):
        super().__init__()
        self.classes = classes
        self.net = nn.Sequential(spectral_norm(nn.Linear(COLOUR_DIMS + TEXTURE_DIMS + classes, hidden)), nn.CELU(),
                                 spectral_norm(nn.Linear(hidden, hidden)), nn.CELU(), spectral_norm(nn.Linear(hidden, 1)))

    def forward(self, rows, label):
        return self.net(torch.cat([rows, nn.functional.one_hot(label, self.classes).float()], 1))


def torch_mmd(a, b):
    """RBF-kernel MMD^2 with the median-distance bandwidth (similarity-measure loss for rows)."""
    joint = torch.cat([a, b])
    distances = torch.cdist(joint, joint) ** 2
    bandwidth = distances.detach().median().clamp_min(1e-6)
    k = torch.exp(-distances / bandwidth)
    n = len(a)
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


def train_tabular(rows, labels, val_rows, val_labels, settings, seed, device="cuda"):
    """Train on one budget's target rows (labels 0 = flooded building, 1 = flooded road); select by validation MMD."""
    from .imbalance import rbf_mmd
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = TabularGenerator(settings["noise"], settings["hidden"]).to(device)
    d = TabularDiscriminator(settings["hidden"]).to(device)
    opt_g = torch.optim.Adam(g.parameters(), lr=2e-4, betas=(.5, .999))
    opt_d = torch.optim.Adam(d.parameters(), lr=2e-4, betas=(.5, .999))
    x = torch.as_tensor(rows, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    history, best = [], None
    for iteration in range(1, settings["iterations"] + 1):
        index = torch.as_tensor(rng.integers(0, len(x), settings["batch_size"]), device=device)
        real, label = x[index], y[index]
        fake = g(torch.randn(len(index), settings["noise"], device=device), label)
        loss_d = relativistic_d_loss(d(real, label), d(fake.detach(), label))
        opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()
        adversarial = relativistic_g_loss(d(real, label).detach(), d(fake, label))
        similarity = sum(torch_mmd(fake[label == c], real[label == c]) + (1 - nn.functional.cosine_similarity(fake[label == c].mean(0), real[label == c].mean(0), 0))
                         for c in (0, 1) if (label == c).sum() > 1)
        loss_g = adversarial + settings["lambda_similarity"] * similarity
        opt_g.zero_grad(set_to_none=True); loss_g.backward(); opt_g.step()
        if not (torch.isfinite(loss_d) and torch.isfinite(loss_g)):
            raise FloatingPointError(f"Tabular R-GAN diverged at iteration {iteration}")
        if iteration % settings["checkpoint_every"] == 0:
            score = float(np.mean([rbf_mmd(sample(g, {c: int((val_labels == c).sum())}, seed, device)[0], val_rows[val_labels == c]) for c in (0, 1)]))
            history.append({"iteration": iteration, "val_mmd": score})
            if best is None or score < best[0]:
                best = (score, iteration, {k: v.detach().cpu().clone() for k, v in g.state_dict().items()})
    g.load_state_dict(best[2])
    return g, {"selected_iteration": best[1], "selected_val_mmd": best[0], "history": history}


def sample(generator, counts, seed, device="cuda"):
    """Deterministic rows: counts = {class index: n}. Returns (rows, labels)."""
    noise = torch.Generator(device=device).manual_seed(seed)
    rows, labels = [], []
    generator.eval()
    with torch.no_grad():
        for c, n in counts.items():
            if n:
                label = torch.full((n,), c, dtype=torch.long, device=device)
                rows.append(generator(torch.randn(n, generator.noise, device=device, generator=noise), label).cpu().numpy())
                labels.append(np.full(n, c))
    generator.train()
    return np.concatenate(rows).astype(np.float32), np.concatenate(labels)
