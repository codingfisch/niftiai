from fastai.basics import torch, random, store_attr
from fastai.vision.augment import PadMode, RandTransform
from mriaug.core import (crop3d, affine3d, warp3d, affinewarp3d, chi_noise3d, bias_field3d, contrast,
                         ringing3d, motion3d, ghosting3d, spike3d, downsample3d, flip3d, dihedral3d)
from mriaug.utils import is_zero, get_mode_grid

from niftiai.core import TensorImage3d, TensorMask3d


class Crop3d(RandTransform):
    order = 10
    def __init__(self, size: (int, tuple), max_translate: (float, tuple) = 1., p: float = .5, batch: bool = False):
        super().__init__(p=p)
        max_translate = tuple(3 * [max_translate]) if isinstance(max_translate, float) else max_translate
        size = (size, size, size) if isinstance(size, int) else tuple(size)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.translate = get_rand_affine_param(b[0], self.max_translate, self.batch)

    def encodes(self, x: (TensorImage3d, TensorMask3d)):
        return crop3d(x, self.translate, self.size)


class AffineWarp3d(RandTransform):
    order = 10
    def __init__(self, max_translate: (float, tuple) = .1, max_rotate: (float, tuple) = (.1, .02, .02),
                 max_zoom: (float, tuple) = .1, max_shear: (float, tuple) = .02, p_affine: float = .5,
                 max_warp: float = .01, steps: int = 2, nodes: int = 2, p_warp: float = .5, upsample: float = 2.,
                 size: (int, tuple) = None, pad_mode: PadMode = PadMode.Zeros, batch: bool = False):
        super().__init__()
        max_translate = tuple(3 * [max_translate]) if isinstance(max_translate, float) else max_translate
        max_rotate = tuple(3 * [max_rotate]) if isinstance(max_rotate, float) else max_rotate
        max_zoom = tuple(3 * [max_zoom]) if isinstance(max_zoom, float) else max_zoom
        max_shear = tuple(3 * [max_shear]) if isinstance(max_shear, float) else max_shear
        size = None if size is None else (size, size, size) if isinstance(size, int) else tuple(size)
        store_attr()

    def before_call(self, b, split_idx):
        self.do_affine = random.random() < self.p_affine
        self.do_warp = random.random() < self.p_warp
        self.do = self.do_affine or self.do_warp
        self.translate = get_rand_affine_param(b[0], self.max_translate, self.batch)
        self.rotate = get_rand_affine_param(b[0], self.max_rotate, self.batch)
        self.zoom = get_rand_affine_param(b[0], self.max_zoom, self.batch)
        self.shear = get_rand_affine_param(b[0], self.max_shear, self.batch)
        self.warp = self.max_warp * torch.rand(len(b[0]))
        if self.batch and len(b[0]) > 1:
            self.warp= self.warp[:1].repeat(len(b[0]))
        self.x_randn = torch.randn(len(b[0]) * 10000, dtype=b[0].dtype, device=b[0].device)

    def encodes(self, x: TensorImage3d):
        mode = 'bilinear'
        return smart_affinewarp3d(x, self.do_affine, self.do_warp, self.translate, self.rotate, self.zoom, self.shear,
                                  self.steps, self.warp, self.nodes, self.x_randn, self.upsample, self.size, mode, self.pad_mode)

    def encodes(self, x: TensorMask3d):
        mode = 'nearest'
        return smart_affinewarp3d(x[:, None], self.do_affine, self.do_warp, self.translate, self.rotate, self.zoom, self.shear,
                                  self.steps, self.warp, self.nodes, self.x_randn, self.upsample, self.size, mode, self.pad_mode)[:, 0]

class Warp3d(RandTransform):
    order = 10
    def __init__(self, max_magnitude: float = .01, steps: int = 2, nodes: int = 2, p: float = .5,
                 size: (int, tuple) = None, pad_mode: PadMode = PadMode.Zeros, batch: bool = False):
        super().__init__(p=p)
        size = None if size is None else (size, size, size) if isinstance(size, int) else tuple(size)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.magnitude = self.max_magnitude * torch.rand(len(b[0]))
        if self.batch and len(b[0]) > 1:
            self.magnitude = self._magnitude[:1].repeat(len(b[0]))
        self.x_randn = torch.randn(len(b[0]) * 10000, dtype=b[0].dtype, device=b[0].device)

    def encodes(self, x: TensorImage3d):
        mode = 'bilinear'
        return warp3d(x, self.magnitude, self.steps, self.nodes, self.x_randn, self.size, mode, self.pad_mode)

    def encodes(self, x: TensorMask3d):
        mode = 'nearest'
        return warp3d(x[:, None], self.magnitude, self.steps, self.nodes, self.x_randn, self.size, mode, self.pad_mode)[:, 0]


class Affine3d(RandTransform):
    order = 10
    def __init__(self, max_translate: (float, tuple) = .1, max_rotate: (float, tuple) = (.1, .02, .02),
                 max_zoom: (float, tuple) = .1, max_shear: (float, tuple) = .02, p: float = .5, upsample: float = 2.,
                 size: (int, tuple) = None, pad_mode: PadMode = PadMode.Zeros, batch: bool = False):
        super().__init__(p=p)
        max_translate = tuple(3 * [max_translate]) if isinstance(max_translate, float) else max_translate
        max_rotate = tuple(3 * [max_rotate]) if isinstance(max_rotate, float) else max_rotate
        max_zoom = tuple(3 * [max_zoom]) if isinstance(max_zoom, float) else max_zoom
        max_shear = tuple(3 * [max_shear]) if isinstance(max_shear, float) else max_shear
        size = None if size is None else (size, size, size) if isinstance(size, int) else tuple(size)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.translate = get_rand_affine_param(b[0], self.max_translate, self.batch)
        self.rotate = get_rand_affine_param(b[0], self.max_rotate, self.batch)
        self.zoom = get_rand_affine_param(b[0], self.max_zoom, self.batch)
        self.shear = get_rand_affine_param(b[0], self.max_shear, self.batch)

    def encodes(self, x: TensorImage3d):
        mode = 'bilinear'
        return affine3d(x, self.translate, self.rotate, self.zoom, self.shear, self.size, mode, self.upsample, self.pad_mode)

    def encodes(self, x: TensorMask3d):
        mode = 'nearest'
        return affine3d(x[:, None], self.translate, self.rotate, self.zoom, self.shear, self.size, mode, self.upsample, self.pad_mode)[:, 0]


class Translate3D(Affine3d):
    def __init__(self, max_translate: (float, tuple) = .1, p: float = .5, upsample: float = 2.,
                 size: (int, tuple) = None, pad_mode: PadMode = PadMode.Zeros, batch: bool = False):
        super().__init__(max_translate=max_translate, max_rotate=0, max_zoom=0, max_shear=0, p=p,
                         upsample=upsample, size=size, pad_mode=pad_mode, batch=batch)


class Rotate3D(Affine3d):
    def __init__(self, max_rotate: (float, tuple) = (.1, .02, .02), p: float = .5, upsample: float = 2.,
                 size: (int, tuple) = None, pad_mode: PadMode = PadMode.Zeros, batch: bool = False):
        super().__init__(max_translate=0, max_rotate=max_rotate, max_zoom=0, max_shear=0, p=p,
                         upsample=upsample, size=size, pad_mode=pad_mode, batch=batch)


class Zoom3D(Affine3d):
    def __init__(self, max_zoom: (float, tuple) = .1, p: float = .5, upsample: float = 2.,
                 size: (int, tuple) = None, pad_mode: PadMode = PadMode.Zeros, batch: bool = False):
        super().__init__(max_translate=0, max_rotate=0, max_zoom=max_zoom, max_shear=0, p=p,
                         upsample=upsample, size=size, pad_mode=pad_mode, batch=batch)


class Shear3D(Affine3d):
    def __init__(self, max_shear: (float, tuple) = .02, p: float = .5, upsample: float = 2.,
                 size: (int, tuple) = None, pad_mode: PadMode = PadMode.Zeros, batch: bool = False):
        super().__init__(max_translate=0, max_rotate=0, max_zoom=0, max_shear=max_shear, p=p,
                         upsample=upsample, size=size, pad_mode=pad_mode, batch=batch)


class ChiNoise3d(RandTransform):
    order = 20
    def __init__(self, max_noise: float = .1, max_dof: int = 3, p: float = .5, batch: bool = False):
        super().__init__(p=p)
        store_attr()

    def encodes(self, x: TensorImage3d):
        return chi_noise3d(x, self.max_noise, random.randint(1, self.max_dof), self.batch)


class BiasField3d(RandTransform):
    order = 30
    def __init__(self, max_bias: float = .2, order: int = 4, p: float = .5, batch: bool = False):
        super().__init__(p=p)
        store_attr()
        self.mode_grid = None

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        if self.mode_grid is None:
            self.mode_grid = get_mode_grid(self.order, x=b[0])
        self.intensity = self.max_bias * torch.rand(self.mode_grid.shape[0], dtype=b[0].dtype, device=b[0].device)
        self.x_rand = torch.rand((len(b[0]), self.mode_grid.shape[1]), dtype=b[0].dtype, device=b[0].device)
        if self.batch and len(b[0]) > 1:
            self.intensity = self.intensity[:1].repeat(len(b[0]))
            self.x_rand = self.x_rand[:1].repeat(len(b[0]), 1)

    def encodes(self, x: TensorImage3d):
        return bias_field3d(x, self.intensity, self.order, mode_grid=self.mode_grid, x_randn=self.x_rand)


class Contrast3d(RandTransform):
    order = 40
    def __init__(self, max_lighting: float = .2, p: float = .5, batch: bool = False):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.lighting = self.max_lighting * torch.rand(len(b[0]), dtype=b[0].dtype, device=b[0].device)
        if self.batch and len(b[0]) > 1:
            self.lighting = self.lighting[:1].repeat(len(b[0]))

    def encodes(self, x: TensorImage3d):
        return contrast(x, self.lighting)


class Ringing3d(RandTransform):
    order = 50
    def __init__(self, max_ringing: float = .2, frequency: float = .7, band: float = .05,
                 dims: (int, tuple) = 2, p: float = .5, batch: bool = False):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.intensity = self.max_ringing * torch.rand(len(b[0]), dtype=b[0].dtype, device=b[0].device)
        self.dim = self.dims if isinstance(self.dims, int) else self.dims[-random.randint(0, len(self.dims) - 1)]
        if self.batch and len(b[0]) > 1:
            self.intensity = self.intensity[:1].repeat(len(b[0]))

    def encodes(self, x: TensorImage3d):
        return ringing3d(x, self.intensity, self.frequency, self.band, self.dim)


class Motion3d(RandTransform):
    order = 60
    def __init__(self, max_motion: float = .2, max_move: float = .02, p: float = .5, batch: bool = False):
        super().__init__(p=p)
        max_move = tuple(3 * [max_move]) if isinstance(max_move, float) else max_move
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.intensity = self.max_motion * torch.rand(len(b[0]), dtype=b[0].dtype, device=b[0].device)
        self.move = get_rand_affine_param(b[0], self.max_move, self.batch)
        if self.batch and len(b[0]) > 1:
            self.intensity = self.intensity[:1].repeat(len(b[0]))

    def encodes(self, x: TensorImage3d):
        return motion3d(x, self.intensity, self.move)


class Ghosting3d(RandTransform):
    order = 70
    def __init__(self, max_ghosting: float = .2, num_ghosts: int = 4,
                 dims: (int, tuple) = 2, p: float = .5, batch: bool = False):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.intensity = self.max_ghosting * torch.rand(len(b[0]), dtype=b[0].dtype, device=b[0].device)
        self.dim = self.dims if isinstance(self.dims, int) else self.dims[-random.randint(0, len(self.dims) - 1)]
        if self.batch and len(b[0]) > 1:
            self.intensity = self.intensity[:1].repeat(len(b[0]))

    def encodes(self, x: TensorImage3d):
        return ghosting3d(x, self.intensity, self.num_ghosts, self.dim)


class Spike3d(RandTransform):
    order = 75
    def __init__(self, max_spike: float = .5, max_frequency: float = 1., p: float = .5, batch: bool = False):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.intensity = self.max_spike * torch.rand(len(b[0]), dtype=b[0].dtype, device=b[0].device)
        self.frequencies = self.max_frequency * torch.rand((len(b[0]), 3), dtype=b[0].dtype, device=b[0].device)
        if self.batch and len(b[0]) > 1:
            self.intensity = self.intensity[:1].repeat(len(b[0]))
            self.frequencies = self.frequencies[:1].repeat(len(b[0]), 1)

    def encodes(self, x: TensorImage3d):
        return spike3d(x, self.intensity, self.frequencies)


class Downsample3d(RandTransform):
    order = 80
    def __init__(self, max_downsample: float = 2., dims: (int, tuple) = 2, p: float = .5, batch: bool = False):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.scale = 1 / (1 + self.max_downsample * random.random())
        self._dim = self.dims if isinstance(self.dims, int) else self.dims[-random.randint(0, len(self.dims) - 1)]

    def encodes(self, x: TensorImage3d):
        return downsample3d(x, self.scale, self._dim, mode='nearest')


class Flip3d(RandTransform):
    order = 90
    def __init__(self, p: float = .5):
        super().__init__(p=p)

    def encodes(self, x: (TensorImage3d, TensorMask3d)):
        return flip3d(x)


class Dihedral3d(RandTransform):
    order = 95
    def __init__(self, p: float = .5, batch: bool = False):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.ks = [random.randint(0,23) for _ in range(len(b[0]))]

    def encodes(self, x: (TensorImage3d, TensorMask3d)):
        return dihedral3d(x, self.ks[0]) if self.batch else torch.stack([dihedral3d(u, k) for u, k in zip(x, self.ks)])


def smart_affinewarp3d(x, do_affine, do_warp, translate, rotate, zoom, shear,
                       steps, warp, nodes, x_randn, upsample, size, mode, pad_mode):
    if do_affine and do_warp:
        return affinewarp3d(x, translate, rotate, zoom, shear, warp, steps, nodes, x_randn, size, mode, upsample, pad_mode)
    elif do_warp:
        return warp3d(x, warp, steps, nodes, x_randn, size, mode, upsample, pad_mode)
    else:
        return affine3d(x, translate, rotate, zoom, shear, size, mode, upsample, pad_mode)


def get_rand_affine_param(x, max_param, batch):
    param = (2 * torch.rand((len(x), 3)) - 1) * torch.tensor(max_param)
    param = param.to(x.device)
    return param[:1].repeat(len(x), 1) if batch and len(x) > 1 else param


def aug_transforms3d(mult: float = 1., max_warp: float = .01, warp_steps: int = 2, warp_nodes: int = 2,
                     p_warp: float = .2, p_affine: float = .5, max_translate: (float, tuple) = .1,
                     max_zoom: (float, tuple) = .1, max_rotate: (float, tuple) = (.1, .02, .02), max_shear: float = .02,
                     max_noise: float = .1, max_dof_noise: int = 4, p_noise: float = .1, max_bias: float = .1,
                     order_bias: int = 5, p_bias: float = .1, max_contrast: float = .2, p_contrast: float = .1,
                     max_ring: float = .2, freq_ring: float = .7, band_ring: float = .05, dims_ring: (int, tuple) = 2,
                     p_ring: float = .1, max_motion: float = .2, max_move: float = .02, p_motion: float = .1,
                     max_ghost: float = .2, num_ghost: int = 4, dims_ghost: (int, tuple) = 2, p_ghost: float = .1,
                     max_down: float = 2., dims_down: (int, tuple) = 2, p_down: float = .1, do_flip: bool = True,
                     do_dihedral: bool = False, batch: bool = False, **affine_kwargs):
    do_translate, do_rotate, do_zoom, do_shear = [not is_zero(p) for p in [max_translate, max_rotate, max_zoom, max_shear]]  # TODO: mult should also work with tuple args
    res = []
    if max_warp:
        if do_translate or do_rotate or do_zoom or do_shear:
            res.append(AffineWarp3d(mult * max_translate, max_rotate, mult * max_zoom, mult * max_shear, p_affine,
                                    mult * max_warp, warp_steps, warp_nodes, p_warp, batch=batch, **affine_kwargs))
        else:
            res.append(Warp3d(mult * max_warp, warp_steps, warp_nodes, p=p_warp, batch=batch))
    else:
        if do_translate or do_rotate or do_zoom or do_shear:
            res.append(Affine3d(mult * max_translate, mult * max_rotate, mult * max_zoom, mult * max_shear, p=p_affine, batch=batch, **affine_kwargs))
    if max_noise: res.append(ChiNoise3d(mult * max_noise, max_dof_noise, p_noise, batch))
    if max_bias: res.append(BiasField3d(mult * max_bias, order_bias, p_bias, batch))
    if max_contrast: res.append(Contrast3d(mult * max_contrast, p_contrast, batch))
    if max_ring: res.append(Ringing3d(mult * max_ring, freq_ring, band_ring, dims_ring, p_ring, batch))
    if max_motion: res.append(Motion3d(mult * max_motion, max_move, p_motion, batch))
    if max_ghost: res.append(Ghosting3d(mult * max_ghost, num_ghost, dims_ghost, p_ghost, batch))
    #if max_spike: res.append(Spike3d(mult * max_spike, p_spike, batch))
    if max_down: res.append(Downsample3d(mult * max_down, dims_down, p_down))
    if do_flip: res.append(Flip3d())
    if do_dihedral: res.append(Dihedral3d())
    return res
