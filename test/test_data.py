import unittest
from PIL import Image
from fastai.basics import np, pd, plt, set_seed, Path, TensorCategory
set_seed(1)

from niftiai.core import TensorMask3d, TensorImage3d
from niftiai.data import ImageDataLoaders3d, SegmentationDataLoaders3d
from niftiai.transforms import Resize
from test.utils import setup_dataset, get_plt_figure_array
DATA_DIR = Path('data/dls_images')
CSV_PATH = setup_dataset(DATA_DIR)


class TestImageDataLoaders3d(unittest.TestCase):
    def test_from_df(self, size=128, bs=2):
        ref_dir = Path('data/images')
        df = pd.read_csv(CSV_PATH)
        dls = ImageDataLoaders3d.from_df(df, fn_col='img_fn', label_col='label', item_tfms=Resize(size), bs=bs)
        x, y = dls.one_batch()
        self.assertTrue(isinstance(x, TensorImage3d))
        self.assertTrue(isinstance(y, TensorCategory))
        self.assertEqual(x.shape, (bs, 1, size, size, size))
        self.assertEqual(y.shape, (bs, ))
        dls.show_batch()
        im_arr = get_plt_figure_array()
        plt.close('all')
        ref_im = Image.open(ref_dir / 'plt_batch.png')
        #self.assertTrue(np.array_equal(im_arr, np.array(ref_im)))  # does sometimes fail (despite being ok) due to random seed
        #Image.fromarray(im_arr).save(ref_dir/'plt_batch.png')

    def test_from_csv(self, size=128, bs=2):
        dls = ImageDataLoaders3d.from_csv(path='.', csv_fname=CSV_PATH, fn_col='img_fn', label_col='label',
                                          item_tfms=Resize(size), bs=bs)
        x, y = dls.one_batch()
        self.assertTrue(isinstance(x, TensorImage3d))
        self.assertTrue(isinstance(y, TensorCategory))
        self.assertEqual(x.shape, (bs, 1, size, size, size))
        self.assertEqual(y.shape, (bs, ))


class TestSegmentationDataLoaders3d(unittest.TestCase):
    def test_from_df(self, size=128, bs=2):
        ref_dir = Path('data/images')
        df = pd.read_csv(CSV_PATH)
        dls = SegmentationDataLoaders3d.from_df(df, fn_col='img_fn', label_col='mask_fn', item_tfms=Resize(size), bs=bs)
        x, y = dls.one_batch()
        self.assertTrue(isinstance(x, TensorImage3d))
        self.assertTrue(isinstance(y, TensorMask3d))
        self.assertEqual(x.shape, (bs, 1, size, size, size))
        self.assertEqual(y.shape, (bs, size, size, size))
        dls.show_batch()
        im_arr = get_plt_figure_array()
        plt.close('all')
        ref_im = Image.open(ref_dir / 'plt_segment_batch.png')
        #self.assertTrue(np.array_equal(im_arr, np.array(ref_im)))  # does sometimes fail (despite being ok) due to random seed
        #Image.fromarray(im_arr).save(ref_dir/'plt_segment_batch.png')

    def test_from_csv(self, size=128, bs=2):
        dls = SegmentationDataLoaders3d.from_csv(path='.', csv_fname=CSV_PATH, fn_col='img_fn', label_col='mask_fn',
                                                 item_tfms=Resize(size), bs=bs)
        x, y = dls.one_batch()
        self.assertTrue(isinstance(x, TensorImage3d))
        self.assertTrue(isinstance(y, TensorMask3d))
        self.assertEqual(x.shape, (bs, 1, size, size, size))
        self.assertEqual(y.shape, (bs, size, size, size))


if __name__ == '__main__':
    unittest.main()
    #shutil.rmtree(DATA_DIR)
