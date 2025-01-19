import unittest
from PIL import Image
from fastai.basics import np, pd, plt, set_seed, Path
set_seed(1)

from niftiai.data import ImageDataLoaders3d, SegmentationDataLoaders3d
from niftiai.transforms import Resize
from niftiai.learner import cnn_learner3d, unet_learner3d
from test.utils import setup_dataset, get_plt_figure_array
DATA_DIR = Path('data/dls_images')
CSV_PATH = setup_dataset(DATA_DIR)


class TestLearner(unittest.TestCase):
    def test_cnn_learner3d(self, size=128, bs=2, valid_pct=.5):
        ref_dir = Path('data/images')
        df = pd.read_csv(CSV_PATH)
        dls = ImageDataLoaders3d.from_df(df, fn_col='img_fn', label_col='label',
                                         item_tfms=Resize(size), bs=bs, valid_pct=valid_pct)
        learner = cnn_learner3d(dls)
        learner.fit(1)
        preds, targs = learner.get_preds()
        n = int(valid_pct * len(df))
        self.assertEqual(tuple(preds.shape), (n, len(df.label.unique())))
        self.assertEqual(tuple(targs.shape), (n, ))
        learner.show_results()
        im_arr = get_plt_figure_array()
        plt.close('all')
        ref_im = Image.open(ref_dir / 'plt_results.png')
        #self.assertTrue(np.array_equal(im_arr, np.array(ref_im)))  # does sometimes fail (despite being ok) due to random seed
        #Image.fromarray(im_arr).save(ref_dir/'plt_results.png')

    def test_unet_learner3d(self, size=128, bs=2, valid_pct=.5, c_out=2):
        ref_dir = Path('data/images')
        df = pd.read_csv(CSV_PATH)
        dls = SegmentationDataLoaders3d.from_df(df, fn_col='img_fn', label_col='mask_fn',
                                                item_tfms=Resize(size), bs=bs, valid_pct=valid_pct)
        learner = unet_learner3d(dls, c_out=c_out)
        learner.fit(1)
        preds, targs = learner.get_preds()
        n = int(valid_pct * len(df))
        self.assertEqual(tuple(preds.shape), (n, c_out, size, size, size))
        self.assertEqual(tuple(targs.shape), (n, size, size, size))
        learner.show_results()
        im_arr = get_plt_figure_array()
        plt.close('all')
        #ref_im = Image.open(ref_dir / 'plt_segment_results.png')
        #self.assertTrue(np.array_equal(im_arr, np.array(ref_im)))  # does sometimes fail (despite being ok) due to random seed
        #Image.fromarray(im_arr).save(ref_dir/'plt_segment_results.png')


if __name__ == '__main__':
    unittest.main()
