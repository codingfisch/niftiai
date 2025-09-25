from fastai.basics import delegates, Learner, DiceLoss

from .models import TinyCNN3d, UNet3d, LinkNet3d


@delegates(Learner.__init__)
def cnn_learner3d(dls, model=None, c_in=None, n_out=None, **kwargs):
    if model is None:
        n_in = dls.one_batch()[0].shape[1] if c_in is None else c_in
        n_out = dls.c if n_out is None else n_out
        assert n_out, '`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`'
        model = TinyCNN3d(n_out=n_out, c_in=n_in)
    return Learner(dls, model, **kwargs)


@delegates(Learner.__init__)
def unet_learner3d(dls, c_out, linknet=False, model=None, c_in=None, loss_func=None, **kwargs):
    if model is None:
        c_in = dls.one_batch()[0].shape[1] if c_in is None else c_in
        model = LinkNet3d(c_in=c_in, c_out=c_out) if linknet else UNet3d(c_in=c_in, c_out=c_out)
    loss_func = DiceLoss() if loss_func is None else loss_func
    return Learner(dls=dls, model=model, loss_func=loss_func, **kwargs)
