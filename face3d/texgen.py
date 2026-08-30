"""A learned texture generator: photograph -> UV albedo, on-manifold by design.

The problem this solves. Projection (face3d/project.py) transfers the
photograph faithfully, which means it transfers whatever the photograph
contains -- an occluder the segmenter missed, a cast shadow, a smear where the
fitted geometry is wrong. The PCA basis never does any of that, because 50
linear coefficients cannot express it, but for the same reason it cannot
express a mole or an eyebrow either. Neither is a model of "what a face texture
can look like"; one is a transfer and the other is a plane.

So learn that model. Two properties are structural here rather than hoped for,
and they are the whole point:

  BOTTLENECK    The texture is reconstructed from `latent` numbers. Whatever
                the decoder emits is something it can reach from a 128-vector
                after training only on faces, so a pair of spectacle frames is
                not in its range. It cannot draw one; it was never able to.

  BOUNDED       The output is a RESIDUAL over the PCA basis, squashed through
                tanh and scaled by AMPLITUDE. No texel can move more than
                AMPLITUDE from a texture the basis already considered plausible.
                The worst failure available to this model is looking a bit
                generic, which is exactly the trade we want: sacrifice
                similarity, never produce something weird.

The residual formulation also matches what was already measured. Detail belongs
to the photograph and broad tone belongs to the basis (see
project.frequency_merge); predicting a residual keeps that split by
construction, and gives a defined, safe target -- zero -- everywhere the
photograph never saw, which is 80% of the UV map.

Two stages, so a failure is diagnosable:

  1. TextureAutoencoder     texture -> latent -> texture. Answers "can a latent
                            this small represent these textures at all?"
  2. TextureGenerator       photograph -> latent -> texture, decoder warm-started
                            from stage 1. Answers "can we predict that latent
                            from a photo?"

Train end-to-end only and a blurry result tells you nothing about which half
was at fault.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Hard ceiling on how far a texel may move from the basis, in [0,1] colour.
# 0.45 is enough for an eyebrow against forehead skin (measured ~0.25 in the
# corpus) with headroom, and far too little to paint a black spectacle frame
# across a cheek.
AMPLITUDE = 0.45


def _block(cin, cout, groups=8):
    """Upsample-then-convolve, never a transposed convolution.

    ConvTranspose2d with a kernel that does not divide its stride lays down
    overlapping output windows, and the seams show up as a fixed checkerboard
    the network then has to learn around. On a face texture that reads as a
    weave across the cheeks -- precisely the "weird stuff" this model exists to
    avoid, arriving from the architecture rather than the data.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(min(groups, cout), cout),
        nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(min(groups, cout), cout),
        nn.SiLU(inplace=True),
    )


class TextureDecoder(nn.Module):
    """latent (B,L) -> residual texture (B,3,R,R), bounded to +/-AMPLITUDE."""

    def __init__(self, latent=128, res=256, width=32):
        super().__init__()
        self.res, self.latent = res, latent
        # 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
        chans = [width * 16, width * 8, width * 8, width * 4,
                 width * 2, width, width // 2]
        self.fc = nn.Linear(latent, chans[0] * 4 * 4)
        self.blocks = nn.ModuleList(
            [_block(chans[i], chans[i + 1]) for i in range(len(chans) - 1)])
        self.out = nn.Conv2d(chans[-1], 3, 3, padding=1)
        # Start life predicting nothing. The identity of this model is "the
        # basis, plus a correction", and a random correction at step 0 is a
        # worse starting point than no correction at all.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z):
        h = self.fc(z).view(z.shape[0], -1, 4, 4)
        for b in self.blocks:
            h = b(h)
        if h.shape[-1] != self.res:
            h = F.interpolate(h, size=(self.res, self.res), mode="bilinear",
                              align_corners=False)
        return torch.tanh(self.out(h)) * AMPLITUDE


class TextureEncoder(nn.Module):
    """texture (B,3,R,R) -> latent (B,L). Stage 1 only."""

    def __init__(self, latent=128, width=32):
        super().__init__()
        c = [4, width, width * 2, width * 4, width * 8, width * 8, width * 16]
        layers = []
        for i in range(len(c) - 1):
            layers += [nn.Conv2d(c[i], c[i + 1], 4, stride=2, padding=1),
                       nn.GroupNorm(min(8, c[i + 1]), c[i + 1]),
                       nn.SiLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(c[-1] * 4 * 4, latent)

    def forward(self, residual, weight):
        # The weight rides along as a fourth channel. Without it the encoder
        # cannot tell "this texel is genuinely mid-grey" from "this texel was
        # never observed and is zero by convention", and it would spend latent
        # capacity encoding the shape of each subject's coverage hole.
        h = self.net(torch.cat([residual, weight], 1))
        return self.fc(h.flatten(1))


class TextureAutoencoder(nn.Module):
    """Stage 1: how small can the latent be and still hold these textures?"""

    def __init__(self, latent=128, res=256, width=32):
        super().__init__()
        self.enc = TextureEncoder(latent, width)
        self.dec = TextureDecoder(latent, res, width)

    def forward(self, residual, weight):
        return self.dec(self.enc(residual, weight))


class TextureGenerator(nn.Module):
    """Stage 2: photograph (B,3,224,224) in [0,1] -> residual texture.

    A separate trunk from the shape encoder rather than a second head on it.
    They want different things -- shape is invariant to the lighting and skin
    tone this has to predict -- and sharing would let texture gradients move
    weights that produce the one output measured to be fragile.
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, latent=128, res=256, width=32, arch="resnet34",
                 pretrained=True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        trunk = getattr(torchvision.models, arch)(weights=weights)
        self.feat_dim = trunk.fc.in_features
        trunk.fc = nn.Identity()
        self.trunk = trunk
        self.head = nn.Linear(self.feat_dim, latent)
        self.dec = TextureDecoder(latent, res, width)
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

    def latent(self, image):
        return self.head(self.trunk((image - self.mean) / self.std))

    def forward(self, image):
        return self.dec(self.latent(image))


def masked_loss(pred, target, weight, off_weight=0.08, grad_weight=0.5):
    """L1 on the residual, weighted by confidence, plus a gradient term.

    `off_weight` is what the model is told outside the observed region, and it
    is not a detail: 80% of the UV map is never seen by any one photograph. The
    target there is ZERO residual -- fall back to the basis -- which is the only
    honest answer and also the safe one. Left unsupervised the decoder would be
    free to emit anything over four fifths of its own output.

    The gradient term exists because plain L1 has a known and visible failure:
    the minimiser of an expected L1 error over a distribution of plausible
    faces is a blurred face. Matching image gradients as well pushes back
    toward edges -- eyebrows, lips, the nasolabial fold -- without the training
    instability of an adversarial term, which on a corpus this size would be a
    bad trade.
    """
    w = weight + off_weight * (1.0 - weight)
    l1 = ((pred - target).abs() * w).mean()

    def dx(t):
        return t[..., :, 1:] - t[..., :, :-1]

    def dy(t):
        return t[..., 1:, :] - t[..., :-1, :]

    g = (((dx(pred) - dx(target)).abs() * w[..., :, 1:]).mean()
         + ((dy(pred) - dy(target)).abs() * w[..., 1:, :]).mean())
    return l1 + grad_weight * g, l1.detach(), g.detach()
