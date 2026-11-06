"""OASIS (Sushko et al., "You Only Need Adversarial Supervision for Semantic Image Synthesis", ICLR 2021) for the
class-imbalance study. The discriminator is a U-Net *segmentation* network with N+1 outputs (the 11 layout classes
plus "fake"); its loss is cross-entropy weighted by inverse class frequency, so rare classes such as flooded roads
still drive the generator. The SPADE generator receives 3D noise with the label map; LabelMix regularises the
discriminator; no perceptual loss (as in the paper)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm

from .gan_common import CONDITION_CHANNELS, diff_augment
from .gan_spade import SPADEGeneratorBase

FAKE = CONDITION_CHANNELS  # index 11 of the (N+1)-way output


class OasisGenerator(SPADEGeneratorBase):
    uses_noise = True

    def __init__(self, nf=48, noise_channels=64):
        super().__init__(nf, CONDITION_CHANNELS + noise_channels)
        self.noise_channels = noise_channels

    def forward(self, cond, noise=None):
        if cond.shape[1:] != (CONDITION_CHANNELS, 256, 256):
            raise ValueError(f"Expected (B, {CONDITION_CHANNELS}, 256, 256) layouts, got {tuple(cond.shape)}")
        z = torch.randn(cond.shape[0], self.noise_channels, 1, 1, device=cond.device, generator=noise)
        return self.render(torch.cat([cond, z.expand(-1, -1, *cond.shape[2:])], 1))


def groups(channels):
    return next(g for g in (8, 4, 3, 2, 1) if channels % g == 0)


class ResBlock(nn.Module):
    """Pre-normalised residual block. Deviation from the paper (spectral norm only): without normalisation the
    activations grew 3-11x per block and the 12-block discriminator diverged (see audit/imbalance_ledger.md)."""
    def __init__(self, cin, cout, normalise_input=True):
        super().__init__()
        # The first block sees the RGB image itself (bounded in [-1, 1]); normalising it would erase absolute colour.
        self.norm_in = nn.GroupNorm(groups(cin), cin) if normalise_input else nn.Identity()
        self.main = nn.Sequential(nn.LeakyReLU(.2), spectral_norm(nn.Conv2d(cin, cout, 3, padding=1)),
                                  nn.GroupNorm(groups(cout), cout), nn.LeakyReLU(.2), spectral_norm(nn.Conv2d(cout, cout, 3, padding=1)))
        self.skip = spectral_norm(nn.Conv2d(cin, cout, 1))

    def residual(self, x):
        h = self.norm_in(x)
        return self.main(h) + self.skip(h)


class ResDown(ResBlock):
    def forward(self, x):
        return F.avg_pool2d(self.residual(x), 2)


class ResUp(ResBlock):
    def forward(self, x):
        return self.residual(F.interpolate(x, scale_factor=2, mode="nearest"))


class OasisDiscriminator(nn.Module):
    """U-Net segmentation discriminator: (B, 3, 256, 256) -> (B, N+1 = 12, 256, 256) logits."""
    def __init__(self, nf=48):
        super().__init__()
        widths = [3, nf, 2 * nf, 4 * nf, 8 * nf, 8 * nf, 8 * nf]
        self.downs = nn.ModuleList([ResDown(widths[i], widths[i + 1], normalise_input=i > 0) for i in range(6)])
        self.ups = nn.ModuleList([ResUp(widths[6], widths[5])]
                                 + [ResUp(widths[i + 1] * 2, widths[i]) for i in range(4, 0, -1)])
        self.ups.append(ResUp(widths[1] * 2, nf))
        self.head = nn.Conv2d(nf, CONDITION_CHANNELS + 1, 1)

    def forward(self, image):
        skips, x = [], image
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.ups[0](x)
        for up, skip in zip(self.ups[1:], reversed(skips[:-1])):
            x = up(torch.cat([x, skip], 1))
        return self.head(x)


def class_balanced_weights(labels, classes):
    """OASIS weighting: inverse pixel frequency of each class present in the batch (0 for absent classes)."""
    counts = torch.bincount(labels.flatten(), minlength=classes).float()
    present = counts > 0
    weights = torch.zeros(classes, device=labels.device)
    weights[present] = counts.sum() / (counts[present] * present.sum())
    return weights


def labels_from(cond):
    return cond.argmax(1)


def segmentation_loss(logits, labels, for_real):
    """Class-balanced (N+1)-way cross-entropy: real pixels -> their class (weighted), fake pixels -> FAKE."""
    if for_real:
        weights = class_balanced_weights(labels, CONDITION_CHANNELS)
        loss = F.cross_entropy(logits, labels, reduction="none")
        return (loss * weights[labels]).mean()
    return F.cross_entropy(logits, torch.full_like(labels, FAKE))


def build(settings):
    return OasisGenerator(settings["nf"], settings["noise_channels"]), OasisDiscriminator(settings["ndf"])


def step(generator, discriminator, opt_g, opt_d, real, cond, features, settings, sampler):
    fake = generator(cond, noise=sampler)
    real_aug, cond_r = diff_augment(real, cond, sampler)
    fake_aug, cond_f = diff_augment(fake.detach(), cond, sampler)
    loss_d = segmentation_loss(discriminator(real_aug), labels_from(cond_r), True) \
        + segmentation_loss(discriminator(fake_aug), labels_from(cond_f), False)
    # LabelMix: mix real and fake by a random subset of classes; the discriminator must mix its outputs the same way.
    labels = labels_from(cond)
    chosen = torch.rand(CONDITION_CHANNELS, device=cond.device, generator=sampler) < .5
    mix = chosen[labels].unsqueeze(1).float()
    mixed = mix * real + (1 - mix) * fake.detach()
    target = mix * discriminator(real) + (1 - mix) * discriminator(fake.detach())
    labelmix = F.mse_loss(discriminator(mixed), target.detach())
    loss_d_total = loss_d + settings["lambda_labelmix"] * labelmix
    opt_d.zero_grad(set_to_none=True)
    loss_d_total.backward()
    opt_d.step()
    fake_aug, cond_f = diff_augment(fake, cond, sampler)
    adversarial = segmentation_loss(discriminator(fake_aug), labels_from(cond_f), True)
    opt_g.zero_grad(set_to_none=True)
    adversarial.backward()
    opt_g.step()
    return {"d": loss_d.detach().item(), "labelmix": labelmix.detach().item(), "g_adv": adversarial.detach().item()}
