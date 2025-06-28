from fastai.basics import torch, random, store_attr, Iterable
from fastai.vision.augment import PadMode, RandTransform
from mriaug.core import (crop3d, affine3d, warp3d, affinewarp3d, chi_noise3d, bias_field3d, contrast,
                         ringing3d, motion3d, ghosting3d, spike3d, downsample3d, flip3d, dihedral3d)

from niftiai.core import TensorImage3d, TensorMask3d


class Crop3d(RandTransform):
    order = 40
    def __init__(self, size: (int, tuple), max_translate: (float, tuple) = 1.,
                 p: float = .5, batch: bool = False, item: bool = False):
        super().__init__(p=p)
        max_translate = tuple(3 * [max_translate]) if isinstance(max_translate, float) else max_translate
        size = (size, size, size) if isinstance(size, int) else tuple(size)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        bs = 1 if self.item else len(b[0])
        self.translate = get_rand_affine_param(bs, self.max_translate, device=b[0].device, batch=self.batch)

    def encodes(self, x: (TensorImage3d, TensorMask3d)):
        return crop3d(x, self.translate, self.size)


class AffineWarp3d(RandTransform):
    order = 40
    def __init__(self, max_zoom: (float, tuple) = .1, max_rotate: (float, tuple) = .1,
                 max_translate: (float, tuple) = .1, max_shear: (float, tuple) = .02, p_affine: float = .5,
                 max_warp: float = .01, k_size: (int, tuple) = 2, p_warp: float = .5, upsample: float = 2,
                 size: (int, tuple) = None, image_mode: str = 'bilinear', mask_mode: str = 'nearest',
                 pad_mode: PadMode = PadMode.Zeros, batch: bool = False, item: bool = False):
        super().__init__()
        max_zoom = max_zoom if isinstance(max_zoom, Iterable) else tuple(3 * [max_zoom])
        max_rotate = max_rotate if isinstance(max_rotate, Iterable) else tuple(3 * [max_rotate])
        max_translate = max_translate if isinstance(max_translate, Iterable) else tuple(3 * [max_translate])
        max_shear = max_shear if isinstance(max_shear, Iterable) else tuple(3 * [max_shear])
        k_size = tuple(3 * [k_size]) if isinstance(k_size, int) else k_size
        size = None if size is None else (size, size, size) if isinstance(size, int) else tuple(size)
        store_attr()

    def before_call(self, b, split_idx):
        self.do_affine = random.random() < self.p_affine
        self.do_warp = random.random() < self.p_warp
        self.do = self.do_affine or self.do_warp
        bs = 1 if self.item else len(b[0])
        self.zoom = get_rand_affine_param(bs, self.max_zoom, device=b[0].device, batch=self.batch)
        self.rotate = get_rand_affine_param(bs, self.max_rotate, device=b[0].device, batch=self.batch)
        self.translate = get_rand_affine_param(bs, self.max_translate, device=b[0].device, batch=self.batch)
        self.shear = get_rand_affine_param(bs, self.max_shear, device=b[0].device, batch=self.batch)
        self.warp = self.max_warp * torch.rand(bs, device=b[0].device)
        self.k = torch.randn((bs, 3, *self.k_size), device=b[0].device)
        self.k[..., 0, 0, 0] = 0
        if self.batch and bs > 1:
            self.warp = self.warp[:1]
            self.k = self.k[:1]
        self.kws = {'do_affine': self.do_affine, 'do_warp': self.do_warp, 'zoom': self.zoom,
                    'rotate': self.rotate, 'translate': self.translate, 'shear': self.shear,
                    'warp': self.warp, 'k': self.k, 'upsample': self.upsample, 'size': self.size,
                    'mode': self.mask_mode if isinstance(b[0], TensorMask3d) else self.image_mode,
                    'pad_mode': self.pad_mode}

    def encodes(self, x: TensorImage3d):
        return apply_affinewarp3d(x[None], **self.kws)[0] if self.item else apply_affinewarp3d(x, **self.kws)

    def encodes(self, x: TensorMask3d):
        return apply_affinewarp3d(x[None, None], **self.kws)[0, 0] if self.item else apply_affinewarp3d(x[:, None], **self.kws)[:, 0]


class Warp3d(RandTransform):
    order = 40
    def __init__(self, max_magnitude: float = .01, k_size: (int, tuple) = 2, p: float = .5, upsample: float = 2,
                 size: (int, tuple) = None, image_mode: str = 'bilinear', mask_mode: str = 'nearest',
                 pad_mode: PadMode = PadMode.Zeros, batch: bool = False, item: bool = False):
        super().__init__(p=p)
        k_size = tuple(3 * [k_size]) if isinstance(k_size, int) else k_size
        size = None if size is None else (size, size, size) if isinstance(size, int) else tuple(size)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        bs = 1 if self.item else len(b[0])
        self.magnitude = self.max_magnitude * torch.rand(bs, device=b[0].device)
        self.k = torch.randn((bs, 3, *self.k_size), device=b[0].device)
        self.k[..., 0, 0, 0] = 0
        if self.batch and bs > 1:
            self.magnitude = self.magnitude[:1]
            self.k = self.k[:1]
        self.kws = {'magnitude': self.magnitude, 'k': self.k, 'upsample': self.upsample, 'size': self.size,
                    'mode': self.mask_mode if isinstance(b[0], TensorMask3d) else self.image_mode,
                    'pad_mode': self.pad_mode}

    def encodes(self, x: TensorImage3d):
        return warp3d(x[None], **self.kws)[0] if self.item else warp3d(x, **self.kws)

    def encodes(self, x: TensorMask3d):
        return warp3d(x[None, None], **self.kws)[0, 0] if self.item else warp3d(x[:, None], **self.kws)[:, 0]


class Affine3d(RandTransform):
    order = 40
    def __init__(self, max_zoom: (float, tuple) = .1, max_rotate: (float, tuple) = .1,
                 max_translate: (float, tuple) = .1, max_shear: (float, tuple) = .02, p: float = .5,
                 upsample: float = 2, size: (int, tuple) = None, image_mode: str = 'bilinear',
                 mask_mode: str = 'nearest', pad_mode: PadMode = PadMode.Zeros, batch: bool = False,
                 item: bool = False):
        super().__init__(p=p)
        max_zoom = max_zoom if isinstance(max_zoom, Iterable) else tuple(3 * [max_zoom])
        max_rotate = max_rotate if isinstance(max_rotate, Iterable) else tuple(3 * [max_rotate])
        max_translate = max_translate if isinstance(max_translate, Iterable) else tuple(3 * [max_translate])
        max_shear = max_shear if isinstance(max_shear, Iterable) else tuple(3 * [max_shear])
        size = None if size is None else (size, size, size) if isinstance(size, int) else tuple(size)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        bs = 1 if self.item else len(b[0])
        self.zoom = get_rand_affine_param(bs, self.max_zoom, device=b[0].device, batch=self.batch)
        self.rotate = get_rand_affine_param(bs, self.max_rotate, device=b[0].device, batch=self.batch)
        self.translate = get_rand_affine_param(bs, self.max_translate, device=b[0].device, batch=self.batch)
        self.shear = get_rand_affine_param(bs, self.max_shear, device=b[0].device, batch=self.batch)
        self.kws = {'zoom': self.zoom, 'rotate': self.rotate, 'translate': self.translate, 'shear': self.shear,
                    'upsample': self.upsample, 'size': self.size, 'pad_mode': self.pad_mode,
                    'mode': self.mask_mode if isinstance(b[0], TensorMask3d) else self.image_mode}

    def encodes(self, x: TensorImage3d):
        return affine3d(x[None], **self.kws)[0] if self.item else affine3d(x, **self.kws)

    def encodes(self, x: TensorMask3d):
        return affine3d(x[None, None], **self.kws)[0, 0] if self.item else affine3d(x[:, None], **self.kws)[:, 0]


class Zoom3d(Affine3d):
    def __init__(self, max_zoom: (float, tuple) = .1, p: float = .5, upsample: float = 2,
                 size: (int, tuple) = None, image_mode: str = 'bilinear', mask_mode: str = 'nearest',
                 pad_mode: PadMode = PadMode.Zeros, batch: bool = False, item: bool = False):
        super().__init__(max_zoom=max_zoom, max_rotate=0, max_translate=0, max_shear=0, p=p,
                         upsample=upsample, size=size, image_mode=image_mode, mask_mode=mask_mode,
                         pad_mode=pad_mode, batch=batch, item=item)


class Rotate3d(Affine3d):
    def __init__(self, max_rotate: (float, tuple) = .1, p: float = .5, upsample: float = 2,
                 size: (int, tuple) = None, image_mode: str = 'bilinear', mask_mode: str = 'nearest',
                 pad_mode: PadMode = PadMode.Zeros, batch: bool = False, item: bool = False):
        super().__init__(max_zoom=0, max_rotate=max_rotate, max_translate=0, max_shear=0, p=p,
                         upsample=upsample, size=size, image_mode=image_mode, mask_mode=mask_mode,
                         pad_mode=pad_mode, batch=batch, item=item)


class Translate3d(Affine3d):
    def __init__(self, max_translate: (float, tuple) = .1, p: float = .5, upsample: float = 2.,
                 size: (int, tuple) = None, image_mode: str = 'bilinear', mask_mode: str = 'nearest',
                 pad_mode: PadMode = PadMode.Zeros, batch: bool = False, item: bool = False):
        super().__init__(max_zoom=0, max_rotate=0, max_translate=max_translate, max_shear=0, p=p,
                         upsample=upsample, size=size, image_mode=image_mode, mask_mode=mask_mode,
                         pad_mode=pad_mode, batch=batch, item=item)


class Shear3d(Affine3d):
    def __init__(self, max_shear: (float, tuple) = .02, p: float = .5, upsample: float = 2,
                 size: (int, tuple) = None, image_mode: str = 'bilinear', mask_mode: str = 'nearest',
                 pad_mode: PadMode = PadMode.Zeros, batch: bool = False, item: bool = False):
        super().__init__(max_zoom=0, max_rotate=0, max_translate=0, max_shear=max_shear, p=p,
                         upsample=upsample, size=size, image_mode=image_mode, mask_mode=mask_mode,
                         pad_mode=pad_mode, batch=batch, item=item)


class ChiNoise3d(RandTransform):
    order = 50
    def __init__(self, max_intensity: float = .1, max_dof: int = 3, p: float = .5, batch: bool = False):
        super().__init__(p=p)
        store_attr()

    def encodes(self, x: TensorImage3d):
        return chi_noise3d(x, self.max_intensity, random.randint(1, self.max_dof), self.batch)


class BiasField3d(RandTransform):
    order = 55
    def __init__(self, max_intensity: float = .1, k_size: (int, tuple) = 2, p: float = .5, batch: bool = False):
        super().__init__(p=p)
        k_size = tuple(3 * [k_size]) if isinstance(k_size, int) else k_size
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.intensity = self.max_intensity * torch.randn(len(b[0]), device=b[0].device)
        self.k = torch.randn((*b[0].shape[:-3], *self.k_size), device=b[0].device)
        self.k[..., 0, 0, 0] = 0
        if self.batch:
            self.intensity = self.intensity[:1]
            self.k = self.k[:1]

    def encodes(self, x: TensorImage3d):
        return bias_field3d(x, self.intensity, k=self.k)


class Contrast3d(RandTransform):
    order = 60
    def __init__(self, max_lighting: float = .2, p: float = .5, batch: bool = False):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.lighting = self.max_lighting * torch.rand(len(b[0]), dtype=b[0].dtype, device=b[0].device)
        if self.batch:
            self.lighting = self.lighting[:1]

    def encodes(self, x: TensorImage3d):
        return contrast(x, self.lighting)


class Ringing3d(RandTransform):
    order = 65
    def __init__(self, max_intensity: float = .2, frequency: float = .7,
                 dims: (int, tuple) = 2, p: float = .5, batch: bool = False):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.intensity = self.max_intensity * torch.rand(len(b[0]), dtype=b[0].dtype, device=b[0].device)
        self.dim = self.dims if isinstance(self.dims, int) else self.dims[-random.randint(0, len(self.dims) - 1)]
        if self.batch and len(b[0]) > 1:
            self.intensity = self.intensity[:1]

    def encodes(self, x: TensorImage3d):
        return ringing3d(x, self.intensity, self.frequency, self.dim)


class Motion3d(RandTransform):
    order = 70
    def __init__(self, max_intensity: float = .2, max_move: float = .02,
                 p: float = .5, batch: bool = False, item: bool = False):
        super().__init__(p=p)
        max_move = tuple(3 * [max_move]) if isinstance(max_move, float) else max_move
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        bs = 1 if self.item else len(b[0])
        self.intensity = self.max_intensity * torch.rand(bs, dtype=b[0].dtype, device=b[0].device)
        self.move = get_rand_affine_param(bs, self.max_move, b[0].device, self.batch)
        if self.batch and len(b[0]) > 1:
            self.intensity = self.intensity[:1].repeat(bs)

    def encodes(self, x: TensorImage3d):
        return motion3d(x[None], self.intensity, self.move)[0] if self.item else motion3d(x, self.intensity, self.move)


class Ghosting3d(RandTransform):
    order = 75
    def __init__(self, max_intensity: float = .2, n_ghosts: int = 4,
                 dims: (int, tuple) = 2, p: float = .5, batch: bool = False):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.intensity = self.max_intensity * torch.rand(len(b[0]), dtype=b[0].dtype, device=b[0].device)
        self.dim = self.dims if isinstance(self.dims, int) else self.dims[-random.randint(0, len(self.dims) - 1)]
        if self.batch and len(b[0]) > 1:
            self.intensity = self.intensity[:1]

    def encodes(self, x: TensorImage3d):
        return ghosting3d(x, self.intensity, self.n_ghosts, self.dim)


class Spike3d(RandTransform):
    order = 80
    def __init__(self, max_intensity: float = .5, max_frequency: float = 1., p: float = .5, batch: bool = False):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.intensity = self.max_intensity * torch.rand(len(b[0]), dtype=b[0].dtype, device=b[0].device)
        self.frequencies = self.max_frequency * torch.rand((len(b[0]), 3), dtype=b[0].dtype, device=b[0].device)
        if self.batch and len(b[0]) > 1:
            self.intensity = self.intensity[:1].repeat(len(b[0]))
            self.frequencies = self.frequencies[:1].repeat(len(b[0]), 1)

    def encodes(self, x: TensorImage3d):
        return spike3d(x, self.intensity, self.frequencies)


class Downsample3d(RandTransform):
    order = 85
    def __init__(self, max_downsample: float = 2., dims: (int, tuple) = 2,
                 p: float = .5, batch: bool = False, item: bool = False):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        bs = 1 if self.item else len(b[0])
        self.scale = 1 / (1 + self.max_downsample * torch.rand(bs, device=b[0].device))
        self.dim = self.dims if isinstance(self.dims, int) else random.choice(self.dims)

    def encodes(self, x: TensorImage3d):
        kws = {'scale': self.scale, 'dim': self.dim, 'mode': 'nearest'}
        return downsample3d(x[None], **kws)[0] if self.item else downsample3d(x, **kws)


class Flip3d(RandTransform):
    order = 90
    def __init__(self, p: float = .5):
        super().__init__(p=p)

    def encodes(self, x: (TensorImage3d, TensorMask3d)):
        return flip3d(x)


class Dihedral3d(RandTransform):
    order = 95
    def __init__(self, p: float = .5, ks: tuple = None):
        super().__init__(p=p)
        store_attr()
        self.ks = tuple(range(24)) if ks is None else ks

    def encodes(self, x: (TensorImage3d, TensorMask3d)):
        return dihedral3d(x, k=random.choice(self.ks))


def apply_affinewarp3d(x, do_affine, do_warp, zoom, rotate, translate, shear, warp, k, upsample, size, mode, pad_mode):
    if do_affine and do_warp:
        return affinewarp3d(x, zoom=zoom, rotate=rotate, translate=translate, shear=shear, magnitude=warp,
                            k=k, size=size, mode=mode, upsample=upsample, pad_mode=pad_mode)
    elif do_warp:
        return warp3d(x, magnitude=warp, k=k, size=size, mode=mode, upsample=upsample, pad_mode=pad_mode)
    else:
        return affine3d(x, zoom=zoom, rotate=rotate, translate=translate, shear=shear,
                        size=size, mode=mode, upsample=upsample, pad_mode=pad_mode)


def get_rand_affine_param(bs, max_param, device, batch):
    param = (2 * torch.rand((bs, 3), device=device) - 1) * torch.tensor(max_param, device=device)
    return param[:1].repeat(bs, 1) if batch and bs > 1 else param


def aug_transforms3d(mult: float = 1., max_warp: float = .01, warp_k_size: int = 2, p_warp: float = .2,
                     max_zoom: (float, tuple) = .1, max_rotate: (float, tuple) = .1,
                     max_translate: (float, tuple) = .1, max_shear: float = .02, p_affine: float = .5,
                     max_noise: float = .1, max_dof_noise: int = 4, p_noise: float = .1, max_bias: float = .1,
                     bias_k_size: int = 2, p_bias: float = .1, max_contrast: float = .2, p_contrast: float = .1,
                     max_ring: float = .2, freq_ring: float = .7, dims_ring: (int, tuple) = 2, p_ring: float = .1,
                     max_motion: float = .2, max_move: float = .02, p_motion: float = .1, max_ghost: float = .2,
                     n_ghosts: int = 4, dims_ghost: (int, tuple) = 2, p_ghost: float = .1, max_spike: float = .1,
                     freq_spike: float = 1., p_spike: float = .01, max_down: float = 2., dims_down: (int, tuple) = 2,
                     p_down: float = .1, p_flip: float = .5, p_dihedral: float = .0, batch: bool = False,
                     item: bool = False, **kwargs):
    do_zoom = all(p != 0 for p in max_zoom) if isinstance(max_zoom, Iterable) else max_zoom != 0
    do_rotate = all(p != 0 for p in max_rotate) if isinstance(max_rotate, Iterable) else max_rotate != 0
    do_translate = all(p != 0 for p in max_translate) if isinstance(max_translate, Iterable) else max_translate != 0
    do_shear = all(p != 0 for p in max_shear) if isinstance(max_shear, Iterable) else max_shear != 0
    mult_max_zoom = tuple(mult * p for p in max_zoom) if isinstance(max_zoom, Iterable) else mult * max_zoom
    mult_max_rotate = tuple(mult * p for p in max_rotate) if isinstance(max_rotate, Iterable) else mult * max_rotate
    mult_max_translate = tuple(mult * p for p in max_translate) if isinstance(max_translate, Iterable) else mult * max_translate
    mult_max_shear = tuple(mult * p for p in max_shear) if isinstance(max_shear, Iterable) else mult * max_shear
    res = []
    if max_warp:
        if do_zoom or do_rotate or do_translate or do_shear:
            res.append(AffineWarp3d(mult_max_zoom, mult_max_rotate, mult_max_translate, mult_max_shear,
                                    p_affine=p_affine, max_warp=mult * max_warp, k_size=warp_k_size,
                                    p_warp=p_warp, batch=batch, item=item, **kwargs))
        else:
            res.append(Warp3d(mult * max_warp, k_size=warp_k_size, p=p_warp, batch=batch, item=item))
    else:
        if do_translate or do_rotate or do_zoom or do_shear:
            res.append(Affine3d(mult_max_zoom, mult_max_rotate, mult_max_translate, mult_max_shear,
                                p=p_affine, batch=batch, item=item, **kwargs))
    if max_noise: res.append(ChiNoise3d(mult * max_noise, max_dof=max_dof_noise, p=p_noise, batch=batch))
    if max_bias: res.append(BiasField3d(mult * max_bias, k_size=bias_k_size, p=p_bias, batch=batch))
    if max_contrast: res.append(Contrast3d(mult * max_contrast, p=p_contrast, batch=batch))
    if max_ring: res.append(Ringing3d(mult * max_ring, freq_ring, dims=dims_ring, p=p_ring, batch=batch))
    if max_motion: res.append(Motion3d(mult * max_motion, max_move=max_move, p=p_motion, batch=batch, item=item))
    if max_ghost: res.append(Ghosting3d(mult * max_ghost, n_ghosts=n_ghosts, dims=dims_ghost, p=p_ghost, batch=batch))
    if max_spike: res.append(Spike3d(mult * max_spike, max_frequency=freq_spike, p=p_spike, batch=batch))
    if max_down: res.append(Downsample3d(mult * max_down, dims=dims_down, p=p_down, batch=batch, item=item))
    if p_flip: res.append(Flip3d(p=p_flip))
    if p_dihedral: res.append(Dihedral3d(p=p_dihedral))
    return res
