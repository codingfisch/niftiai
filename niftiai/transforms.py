from resize_right import resize, interp_methods
from fastai.basics import np, noop, torch, to_cpu, store_attr, Normalize, DisplayedTransform

from .core import TensorMask3d, TensorImage3d


class Crop(DisplayedTransform):
    order = 10

    def __init__(self, slices, **kwargs):
        store_attr()
        super().__init__(**kwargs)

    def encodes(self, x: (TensorMask3d, TensorImage3d)):
        return x[self.slices]


class ReplaceNan(DisplayedTransform):
    order = 20
    def __init__(self, nan: float = 0., **kwargs):
        store_attr()
        super().__init__(**kwargs)

    def encodes(self, x: TensorImage3d):
        return torch.nan_to_num(x, nan=self.nan)


class Scale(DisplayedTransform):
    order = 30
    def __init__(self, maximum: float = .99, minimum: float = .0, are_values: bool = False, **kwargs):
        store_attr()
        super().__init__(**kwargs)

    def encodes(self, x: TensorImage3d):
        if self.are_values:
            return (x - self.minimum) / (self.maximum - self.minimum)
        else:
            minimum, maximum = np.quantile(x.cpu().numpy(), [self.minimum, self.maximum])
            return (x - minimum) / (maximum - minimum)


class Resize(DisplayedTransform):
    order = 40

    def __init__(self, size: (int, tuple), pad_mode: str = 'constant', antialiasing: bool = True, **kwargs):
        size = (size, size, size) if isinstance(size, int) else size
        store_attr()
        super().__init__(**kwargs)

    def encodes(self, x: TensorImage3d):
        return resize(x, out_shape=[*x.shape[:-3], *self.size], pad_mode=self.pad_mode,
                      antialiasing=self.antialiasing, interp_method=interp_methods.lanczos3, by_convs=True)

    def encodes(self, x: TensorMask3d):
        return resize(x, out_shape=[*x.shape[:-3], *self.size], pad_mode=self.pad_mode,
                      antialiasing=False, interp_method=interp_methods.linear, by_convs=True)


class Normalize3D(Normalize):
    def __init__(self, mean=None, std=None, axes=(0, 2, 3, 4)):
        super().__init__(mean=mean, std=std, axes=axes)

    def encodes(self, x:TensorImage3d):
        return (x-self.mean) / self.std

    def decodes(self, x:TensorImage3d):
        f = to_cpu if x.device.type=='cpu' else noop
        return (x*f(self.std) + f(self.mean))
