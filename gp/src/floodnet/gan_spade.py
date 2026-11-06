"""SPADE / GauGAN (Park et al., CVPR 2019) for the class-imbalance study: spatially-adaptive normalisation keeps the
layout through the generator; 2-scale PatchGAN discriminator, hinge loss, feature matching and a perceptual loss.
Deviation (offline): the perceptual loss uses the cached ImageNet ResNet-18 instead of VGG-19; no style encoder.
The SPADE blocks are shared with OASIS (gan_oasis.py)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm

from .gan_common import CONDITION_CHANNELS, diff_augment, perceptual_loss


class SPADE(nn.Module):
    def __init__(self, channels, label_channels, hidden=96):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.shared = nn.Sequential(nn.Conv2d(label_channels, hidden, 3, padding=1), nn.ReLU())
        self.gamma = nn.Conv2d(hidden, channels, 3, padding=1)
        self.beta = nn.Conv2d(hidden, channels, 3, padding=1)

    def forward(self, x, segmap):
        segmap = F.interpolate(segmap, size=x.shape[2:], mode="nearest")
        h = self.shared(segmap)
        return self.norm(x) * (1 + self.gamma(h)) + self.beta(h)


class SPADEResBlock(nn.Module):
    def __init__(self, fin, fout, label_channels):
        super().__init__()
        middle = min(fin, fout)
        self.norm_0, self.conv_0 = SPADE(fin, label_channels), spectral_norm(nn.Conv2d(fin, middle, 3, padding=1))
        self.norm_1, self.conv_1 = SPADE(middle, label_channels), spectral_norm(nn.Conv2d(middle, fout, 3, padding=1))
        self.learned_shortcut = fin != fout
        if self.learned_shortcut:
            self.norm_s, self.conv_s = SPADE(fin, label_channels), spectral_norm(nn.Conv2d(fin, fout, 1, bias=False))

    def forward(self, x, segmap):
        shortcut = self.conv_s(self.norm_s(x, segmap)) if self.learned_shortcut else x
        dx = self.conv_0(F.leaky_relu(self.norm_0(x, segmap), .2))
        dx = self.conv_1(F.leaky_relu(self.norm_1(dx, segmap), .2))
        return shortcut + dx


class SPADEGeneratorBase(nn.Module):
    """8x8 -> 256x256 with SPADE residual blocks; the input is the downsampled (label [+ noise]) map."""
    def __init__(self, nf, label_channels):
        super().__init__()
        widths = [8 * nf, 8 * nf, 8 * nf, 4 * nf, 2 * nf, nf, nf // 2]
        self.fc = nn.Conv2d(label_channels, widths[0], 3, padding=1)
        self.blocks = nn.ModuleList([SPADEResBlock(widths[i], widths[i + 1], label_channels) for i in range(6)])
        self.out = nn.Conv2d(widths[-1], 3, 3, padding=1)

    def render(self, segmap):
        x = self.fc(F.interpolate(segmap, size=(8, 8), mode="nearest"))
        for i, block in enumerate(self.blocks):
            x = block(x, segmap)
            if i < 5:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
        return torch.tanh(self.out(F.leaky_relu(x, .2)))


class SpadeGenerator(SPADEGeneratorBase):
    uses_noise = False

    def __init__(self, nf=48):
        super().__init__(nf, CONDITION_CHANNELS)

    def forward(self, cond):
        if cond.shape[1:] != (CONDITION_CHANNELS, 256, 256):
            raise ValueError(f"Expected (B, {CONDITION_CHANNELS}, 256, 256) layouts, got {tuple(cond.shape)}")
        return self.render(cond)


class PatchDiscriminator(nn.Module):
    def __init__(self, ndf):
        super().__init__()
        cin = 3 + CONDITION_CHANNELS
        self.blocks = nn.ModuleList([
            nn.Sequential(spectral_norm(nn.Conv2d(cin, ndf, 4, 2, 1)), nn.LeakyReLU(.2)),
            nn.Sequential(spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)), nn.InstanceNorm2d(ndf * 2), nn.LeakyReLU(.2)),
            nn.Sequential(spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)), nn.InstanceNorm2d(ndf * 4), nn.LeakyReLU(.2)),
            nn.Sequential(spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1)), nn.InstanceNorm2d(ndf * 8), nn.LeakyReLU(.2)),
            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 1)),
        ])

    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features  # last element = patch logits


class SpadeDiscriminator(nn.Module):
    """Two PatchGANs at full and half resolution; each returns its intermediate features, logits last."""
    def __init__(self, ndf=64):
        super().__init__()
        self.scales = nn.ModuleList([PatchDiscriminator(ndf), PatchDiscriminator(ndf)])

    def forward(self, image, cond):
        x = torch.cat([image, cond], 1)
        outputs = []
        for i, scale in enumerate(self.scales):
            outputs.append(scale(x))
            if i == 0:
                x = F.avg_pool2d(x, 3, stride=2, padding=1, count_include_pad=False)
        return outputs


def build(settings):
    return SpadeGenerator(settings["nf"]), SpadeDiscriminator(settings["ndf"])


def step(generator, discriminator, opt_g, opt_d, real, cond, features, settings, sampler):
    fake = generator(cond)
    real_aug, cond_r = diff_augment(real, cond, sampler)
    fake_aug, cond_f = diff_augment(fake.detach(), cond, sampler)
    loss_d = sum(F.relu(1 - out[-1]).mean() for out in discriminator(real_aug, cond_r)) \
        + sum(F.relu(1 + out[-1]).mean() for out in discriminator(fake_aug, cond_f))
    opt_d.zero_grad(set_to_none=True)
    loss_d.backward()
    opt_d.step()
    fake_aug, cond_f = diff_augment(fake, cond, sampler)
    adversarial = -sum(out[-1].mean() for out in discriminator(fake_aug, cond_f))
    with torch.no_grad():
        real_outputs = discriminator(real, cond)
    fake_outputs = discriminator(fake, cond)
    matching = sum((a - b.detach()).abs().mean() for fo, ro in zip(fake_outputs, real_outputs) for a, b in zip(fo[:-1], ro[:-1]))
    perceptual = perceptual_loss(features, fake, real)
    loss_g = adversarial + settings["lambda_feature_matching"] * matching + settings["lambda_perceptual"] * perceptual
    opt_g.zero_grad(set_to_none=True)
    loss_g.backward()
    opt_g.step()
    return {"d": loss_d.detach().item(), "g_adv": adversarial.detach().item(), "fm": matching.detach().item(), "perceptual": perceptual.detach().item()}
