# niftiai

`niftiai` aims to be the **easiest framework to train neural nets on 3D images** (often [NIfTIs](https://brainder.org/2012/09/23/the-nifti-file-format/)), using

- [`fastai`](https://github.com/fastai/fastai) *easy neural net training*
- [`niftiview`](https://github.com/codingfisch/niftiview) *easy 3D image viewing*
- [`mriaug`](https://github.com/codingfisch/mriaug) *easy 3D image (+MRI-specific) augmentation*

**`pip install niftiai`** to **simplify your code** and skip complex frameworks like [`MONAI`](https://github.com/Project-MONAI/MONAI) and [`torchio`](https://github.com/fepegar/torchio)!

## Quick Start 🚀

Study the [beginner tutorial](https://colab.research.google.com/drive/1LSI49-n94uAML8h0uBe8hoz9YUEIQ59z?usp=sharing) to **understand** how **15 lines of code** can train a neural net to **classify 🧠 MRIs**...

```python
import openneuro as on
from deepmriprep import run_preprocess
from fastai.basics import pd, accuracy, CategoryBlock
from niftiai import cnn_learner3d, Scale, ImageDataLoaders3d
DATA_DIR = 'data/ds000005'

on.download(dataset=DATA_DIR[-8:], target_dir=DATA_DIR, include='*/anat/*T1w.*')
df = run_preprocess(bids_dir=DATA_DIR, outputs=['brain'])
table = pd.read_table(f'{DATA_DIR}/participants.tsv')
df = pd.concat([table, df.reset_index()], axis=1)
dls = ImageDataLoaders3d.from_df(df, fn_col='brain', item_tfms=[Scale()],
                                 label_col='sex', y_block=CategoryBlock())
learner = cnn_learner3d(dls, metrics=[accuracy])
learner.fit(3)
learner.show_results()  # add plt.show() if not in Jupyter notebook
```

...and the [intermediate tutorial](https://colab.research.google.com/drive/1L1RX8OTzt5GCSatTNTH5aoHSd5hGHMOQ?usp=sharing) to see how **12 lines of code** train a UNet to do **MR image segmentation** 🤯

```python
import openneuro as on
from deepmriprep import run_preprocess
from niftiai import unet_learner3d, aug_transforms3d, Scale, SegmentationDataLoaders3d
DATA_DIR = 'data/ds000001'

on.download(dataset=DATA_DIR[-8:], target_dir=DATA_DIR, include='*/anat/*T1w.*')
df = run_preprocess(bids_dir=DATA_DIR, outputs=['mask']).reset_index()
aug_tfms = aug_transforms3d()
dls = SegmentationDataLoaders3d.from_df(df, fn_col='t1', item_tfms=Scale(),
                                        label_col='mask', batch_tfms=aug_tfms)
learner = unet_learner3d(dls, c_out=2, linknet=True)
learner.fit(3, lr=1e-2)
learner.show_results()  # add plt.show() if not in Jupyter notebook
```

👩‍💻 Also study the [fastai tutorials](https://docs.fast.ai/tutorial.html) to **understand the underlying framework** that is

*... approachable and rapidly productive, while also being deeply hackable...*

and discover its **wide range of features** like

- **mixed precision** training for **reduced VRAM usage** via [`Learner.to_fp16()`](https://docs.fast.ai/callback.fp16.html#learner.to_fp16) and [`Learner.to_bf16()`](https://docs.fast.ai/callback.fp16.html#learner.to_bf16)
```python
learner = learner.to_fp16()  # to enable FP16, use this line before training
learner = learner.to_bf16()  # to enable BF16, use this line before training
```
- 1cycle scheduled training for **faster convergence** via [`Learner.fit_one_cycle()`](https://docs.fast.ai/callback.schedule.html#learner.fit_one_cycle)
```python
from fastai.callback.all import *
...
learner.fit_one_cycle(...)  # instead of learner.fit(...) to enable 1cycle scheduled training
```
- [**distributed training**](https://docs.fast.ai/distributed.html) to use **multiple GPUs** via [`accelerate`](https://github.com/huggingface/accelerate)
```python
from fastai.distributed import *
...
with learner.distrib_ctx(): learn.fit(...) # and run with "accelerate launch ..."
```
- ...so much more...
