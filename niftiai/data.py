from fastai.basics import os, np, pd, csv, torch, is_listy, delegates, dispatch, Path, Tensor, TitledFloat, ColSplitter
from fastai.data.all import DataBlock, CategoryBlock, MultiCategoryBlock, DataLoaders, TransformBlock, ColReader, RandomSplitter
from fastai.vision.all import Image, TensorCategory, AddMaskCodes

from niftiai.core import plot, get_blended_image, TensorMask3d, TensorImage3d


def ImageBlock3d(cls=TensorImage3d):
    return TransformBlock(type_tfms=cls.create)


def MaskBlock3d(codes=None):
    return TransformBlock(type_tfms=TensorMask3d.create, item_tfms=AddMaskCodes(codes=codes))


class ImageDataLoaders3d(DataLoaders):
    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_df(cls, df, path='.', valid_pct=0.2, seed=None, fn_col=0, folder=None, suff='', label_col=1, label_delim=None,
                y_block=None, valid_col=None, item_tfms=None, batch_tfms=None, img_cls=TensorImage3d, bs=2, **kwargs):
        pref = f'{Path(path) if folder is None else Path(path)/folder}{os.path.sep}'
        if y_block is None:
            is_multi = (is_listy(label_col) and len(label_col) > 1) or label_delim is not None
            y_block = MultiCategoryBlock if is_multi else CategoryBlock
        splitter = RandomSplitter(valid_pct, seed=seed) if valid_col is None else ColSplitter(valid_col)
        dblock = DataBlock(blocks=(ImageBlock3d(img_cls), y_block),
                           get_x=ColReader(fn_col, pref=pref, suff=suff),
                           get_y=ColReader(label_col, label_delim=label_delim),
                           splitter=splitter,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        return cls.from_dblock(dblock, df, path=path, bs=bs, **kwargs)

    @classmethod
    def from_csv(cls, path, csv_fname='labels.csv', header='infer', delimiter=None, quoting=csv.QUOTE_MINIMAL, **kwargs):
        df = pd.read_csv(Path(path)/csv_fname, header=header, delimiter=delimiter, quoting=quoting)
        return cls.from_df(df, path=path, **kwargs)


@dispatch
def show_batch(x: TensorImage3d, y: Tensor, samples, max_n=10, ctxs=None, **kwargs):
    title = [str(item) for item in samples.itemgot(1)]
    return x[:max_n].show(title=title, **kwargs)


@dispatch
def show_results(x: TensorImage3d, y: Tensor, samples, outs, max_n=10, ctxs=None, **kwargs):
    title = []
    for item, pred, _ in zip(samples.itemgot(1), outs.itemgot(0), range(max_n)):
        item_str, pred_str = (str(pred)[1:6], str(item)[:5]) if isinstance(item, TitledFloat) else (str(item), str(pred))
        extra_str = '' if isinstance(item, TitledFloat) else [' ✖', ' ✔'][item == pred]
        title.append(f'Pred. {pred_str} is {item_str}' + extra_str)
    return x[:max_n].show(title=title, **kwargs)


@dispatch
def plot_top_losses(x: TensorImage3d, y: TensorCategory, samples, outs, raws, losses, **kwargs):
    title = []
    for item, pred, r, loss in zip(samples.itemgot(1), outs.itemgot(0), raws, losses):
        title.append(f'Pred. {pred} is {item} (Loss={loss.item():.2g}, Prob.={r.max().item():.2g})')
    return x.show(title=title, **kwargs)


class SegmentationDataLoaders3d(ImageDataLoaders3d):
    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_df(cls, df, codes=None, y_block=None, **kwargs):
        y_block = MaskBlock3d(codes) if y_block is None else y_block
        return ImageDataLoaders3d.from_df(df, y_block=y_block, **kwargs)


@dispatch
def show_batch(x: TensorImage3d, y: TensorMask3d, samples, ctxs=None, figsize=None, ctx=None, max_n=16, **kwargs):
    return plot(get_blended_image([x[:max_n, 0], y[:max_n]], **kwargs), figsize=figsize, ctx=ctx)


@dispatch
def show_results(x: TensorImage3d, y: TensorMask3d, samples, outs, ctxs=None, max_n=16, figsize=None, ctx=None, nrows=None, **kwargs):
    nrows = len(x) if nrows is None else nrows
    targ_title, pred_title = (['Target'] + [None] * (nrows - 1), ['Prediction'] + [None] * (nrows - 1))
    im = get_results_image(x, y, samples, outs, max_n, nrows, targ_title, pred_title, **kwargs)
    return plot(im, figsize=figsize, ctx=ctx)


@dispatch
def plot_top_losses(x: TensorImage3d, y: TensorMask3d, samples, outs, raws, losses, max_n=16, figsize=None, ctx=None, nrows=None, **kwargs):
    targ_title, pred_title = [], []
    for loss in losses:
        targ_title.append('Target')
        pred_title.append(f'Prediction loss={loss.item():.2g}')
    im = get_results_image(x, y, samples, outs, max_n, nrows, targ_title, pred_title, **kwargs)
    return plot(im, figsize=figsize, ctx=ctx)


def get_results_image(x, y, samples, outs, max_n, nrows, targ_title, pred_title, **kwargs):
    nrows = min(max_n, len(samples)) if nrows is None else nrows
    im0 = get_blended_image([x[:max_n, 0], y[:max_n]], nrows=nrows, title=targ_title, **kwargs)
    im1 = get_blended_image([x[:max_n, 0], torch.stack([*outs.itemgot(0)[:max_n]])], nrows=nrows, title=pred_title, **kwargs)
    return Image.fromarray(np.hstack([np.array(im0), np.array(im1)]))
