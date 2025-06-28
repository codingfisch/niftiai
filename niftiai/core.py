import nibabel as nib
from niftiview import NiftiImageGrid
from niftiview.image import CMAP, QRANGE
from fastai.basics import np, plt, torch, ndarray, warnings, Path, Tensor, TensorImageBW


class TensorImageBase3d(TensorImageBW):
    _show_args = {'cmap': CMAP, 'transp_if': None, 'qrange': QRANGE[0], 'resizing': 1}

    def show(self, use_affine=None, figsize=None, ctx=None, **kwargs):
        im = self.get_image(use_affine=use_affine, **kwargs)
        return plot(im, figsize=figsize, ctx=ctx)

    def get_image(self, use_affine=True, **kwargs):
        if use_affine:
            use_affine = self.org_shape[-3:] == self.shape[-3:] if hasattr(self, 'org_shape') else False
        for k, v in self._show_args.items():
            if k not in kwargs:
                kwargs.update({k: v})
        arrays = channels_last(self.detach().cpu().numpy())
        arrays = [arrays] if self.ndim < 5 else arrays
        affines = len(self) * [self.affine if hasattr(self, 'affine') and use_affine else None]
        return NiftiImageGrid(arrays=arrays, affines=affines).get_image(**kwargs)

    def save(self, fn: (str, Path), keep_shape=False, squeeze=True, no_warning=False, cls=nib.Nifti1Image, **kwargs):
        array = self.detach().cpu().numpy()
        if str(fn).endswith('.npy'):
            np.save(fn, array, **kwargs)
        else:
            if not keep_shape:
                array = channels_last(array)
                if squeeze and array.ndim > 3:
                    array = array.squeeze()
                if array.ndim > 4 and not no_warning:
                    warnings.warn(f'Could not squeeze array into 4D. Saved {array.ndim}D array', UserWarning)
            array = orient_array(array, self.affine)
            cls(array, affine=self.affine, header=self.header).to_filename(fn, **kwargs)

    @classmethod
    def create(cls, fn: (str, Path, ndarray, Tensor), affine=None, header=None, filepath=None, slices=None, **kwargs):
        if isinstance(fn, (str, Path)):
            filepath = str(fn)
            if '[' in filepath and ':' in filepath and filepath.endswith(']'):
                filepath, slices = extract_slices(filepath)
            fn, affine, header = load_np(filepath, slices) if filepath.endswith('.npy') else load_nib(filepath, slices)
            fn = tensor_reshape(fn, is_mask='TensorMask3d' in str(cls))
        if isinstance(fn, ndarray):
            fn = torch.from_numpy(fn.copy())
        return cls.__new__(cls=cls, x=fn, affine=affine, header=header,
                           filepath=filepath, org_shape=fn.shape, slices=slices, **kwargs)


class TensorImage3d(TensorImageBase3d):
    pass


class TensorMask3d(TensorImageBase3d):
    _show_args = {'cmap': 'colorbrewer:reds', 'transp_if': '=0', 'qrange': QRANGE[1], 'resizing': 0}

    def show(self, **kwargs):
        codes = getattr(self, 'codes', None)
        if codes is not None:
            vmax = max(codes)
            kwargs['vrange'] = (0, vmax)
            kwargs['cmap'] = 'colorbrewer:' + ('reds' if vmax < 3 else f'set1_{vmax}_r' if vmax < 10 else 'set1_r')
        return super().show(**kwargs)


def load_np(filepath, slices=None, header=None):
    array = np.load(filepath)
    array = array if slices is None else array[*slices]
    affine = get_dummy_affine(array.shape)
    return array, affine, header


def get_dummy_affine(shape):
    affine = np.eye(4)
    affine[:3, 3] -= np.array(shape) / 2
    return affine


def load_nib(filepath, slices=None):
    im = nib.load(filepath)
    array = nib.as_closest_canonical(im).get_fdata() if slices is None else im.dataobj[*slices]
    return array, im.affine, im.header


def extract_slices(filepath):
    filepath, bounds = str(filepath).split('[')
    slices = []
    for b in bounds[:-1].replace(' ', '').split(','):
        if b == ':':
            slices.append(slice(None))
        elif b.startswith(':'):
            slices.append(slice(int(float(b[1:]))))
        elif b.endswith(':'):
            slices.append(slice(int(float(b[:-1])), None))
        else:
            start, stop = b.split(':')
            slices.append(slice(int(float(start)), int(float(stop))))
    return filepath, slices


def plot(im, ax=None, figsize=None, ctx=None):
    ax = ctx if ax is None else ctx
    if figsize is None:
        figsize = (im.size[0] / 100, im.size[1] / 100)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.imshow(im.convert('RGBA'), cmap=None)
    ax.axis('off')
    return ax


def tensor_reshape(x, is_mask=False):
    if x.ndim == 3  and is_mask:
        return x#.astype(np.uint8)
    x = x[..., None] if x.ndim == 3 else x
    return channels_second(x)


def get_blended_image(xs, use_affine=False, **kwargs):
    use_affine = use_affine and all(x.org_shape[-3:] == x.shape[-3:] for x in xs)
    for k in TensorImageBase3d._show_args:
        kwargs.update({k: kwargs[k] if k in kwargs else [x._show_args[k] for x in xs]})
    arrays, affines = [], []
    for i in range(len(xs[0])):
        arrays.append([x[i].detach().cpu().numpy() for x in xs])
        affines.append([x[i].affine if use_affine else None for x in xs])
    return NiftiImageGrid(arrays=arrays, affines=affines).get_image(**kwargs)


def orient_array(array, affine):
    ornt = nib.orientations.ornt_transform(start_ornt=[[0, 1], [1, 1], [2, 1]], end_ornt=nib.io_orientation(affine))
    return nib.apply_orientation(array, ornt)


def channels_last(array):
    return array if array.ndim < 4 else array.transpose(*list(range(array.ndim - 4)), -3, -2, -1, -4)


def channels_second(array):
    return array if array.ndim < 4 else array.transpose(*list(range(array.ndim - 4)), -1, -4, -3, -2)
