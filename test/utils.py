import nibabel as nib
from fastai.basics import np, pd, plt, Path
from niftiview import ATLASES, TEMPLATES


def files_are_equal(file1, file2):
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        return f1.read() == f2.read()


def get_plt_figure_array(fig=None):
    fig = plt.gcf() if fig is None else fig
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(height, width, 4)[..., 1:]


def setup_dataset(path):
    path.mkdir(exist_ok=True, parents=True)
    img_fns, mask_fns = [], []
    for template, atlas in zip(TEMPLATES, ATLASES):
        img_fn = path / f'{template}.nii'
        mask_fn = path / f'{atlas}.nii'
        if not Path(img_fn).is_file():
            nib.load(TEMPLATES[template]).to_filename(img_fn)
        if not Path(img_fn).is_file():
            nib.load(ATLASES[atlas]).to_filename(mask_fn)
        img_fns.append(img_fn)
        mask_fns.append(mask_fn)
    df = pd.DataFrame({'img_fn': 2 * img_fns, 'mask_fn': 2 * img_fns, 'label': len(img_fns) * [0] + len(img_fns) * [1]})
    csv_path = path / 'dataset.csv'
    df.to_csv(csv_path)
    return csv_path
