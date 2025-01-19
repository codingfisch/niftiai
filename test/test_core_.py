import unittest
import nibabel as nib
from PIL import Image
from niftiview import ATLASES, TEMPLATES
from fastai.basics import np, plt, torch, shutil, F, Path

from niftiai.core import get_blended_image, TensorMask3d, TensorImageBase3d
from test.utils import files_are_equal, get_plt_figure_array
ATLAS_FN = ATLASES['aal3']
TEMPLATE_FN = TEMPLATES['ch2']


class TestTensorImageBase3d(unittest.TestCase):
    def setUp(self):
        self.x = TensorImageBase3d.create(TEMPLATE_FN)
        self.xa = TensorImageBase3d.create(ATLAS_FN)

    def test_show(self):
        ref_dir = Path('data/images')
        self.x.show()
        im_arr = get_plt_figure_array()
        plt.close('all')
        ref_im = Image.open(ref_dir / 'plt_x.png')
        self.assertTrue(np.array_equal(im_arr, np.array(ref_im)))
        #Image.fromarray(im_arr).save(ref_dir/'plt_x.png')
        figsize = (im_arr.shape[1] / 100, im_arr.shape[0] / 100)
        self.x.show(figsize=figsize)
        im_arr = get_plt_figure_array()
        plt.close('all')
        ref_im = Image.open(ref_dir / 'plt_figsize_x.png')
        self.assertTrue(np.array_equal(im_arr, np.array(ref_im)))
        #Image.fromarray(im_arr).save(ref_dir/'plt_figsize_x.png')
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        self.x.show(ctx=ax)
        im_arr = get_plt_figure_array()
        plt.close('all')
        ref_im = Image.open(ref_dir / 'plt_ctx_x.png')
        self.assertTrue(np.array_equal(im_arr, np.array(ref_im)))
        #Image.fromarray(im_arr).save(ref_dir/'plt_ctx_x.png')

    def test_get_image(self):
        ref_dir = Path('data/images')
        im = self.x.get_image()
        ref_im = Image.open(ref_dir/'x.png')
        self.assertTrue(np.array_equal(np.array(im), np.array(ref_im)))
        #im.save(ref_dir/'x.png')
        im = self.x.get_image(use_affine=False)
        #ref_im = Image.open(ref_dir/'x_noaffine.png')
        #self.assertTrue(np.array_equal(np.array(im), np.array(ref_im)))
        im.save(ref_dir/'x_noaffine.png')
        im = self.x.get_image(crosshair=True)
        ref_im = Image.open(ref_dir/'x_crosshair.png')
        self.assertTrue(np.array_equal(np.array(im), np.array(ref_im)))
        #im.save(ref_dir/'x_crosshair.png')
        im = self.xa.get_image()
        ref_im = Image.open(ref_dir/'xa.png')
        self.assertTrue(np.array_equal(np.array(im), np.array(ref_im)))
        #im.save(ref_dir / 'xa.png')
        im = TensorMask3d(self.xa).get_image()
        ref_im = Image.open(ref_dir/'xa_mask.png')
        self.assertTrue(np.array_equal(np.array(im), np.array(ref_im)))
        #im.save(ref_dir / 'xa_mask.png')
        xa_intepolated = F.interpolate(self.xa[None], self.x.shape[-3:])[0]
        im = torch.stack([self.x, xa_intepolated]).get_image()
        ref_im = Image.open(ref_dir/'xs_stacked.png')
        self.assertTrue(np.array_equal(np.array(im), np.array(ref_im)))
        #im.save(ref_dir / 'xs_stacked.png')
        im = get_blended_image([self.x, TensorMask3d(self.xa)], use_affine=True)
        ref_im = Image.open(ref_dir/'xs_blended.png')
        self.assertTrue(np.array_equal(np.array(im), np.array(ref_im)))
        #im.save(ref_dir / 'xs_layered.png')
        im = get_blended_image([self.x, TensorMask3d(self.xa)], use_affine=False)
        ref_im = Image.open(ref_dir/'xs_blended_noaffine.png')
        self.assertTrue(np.array_equal(np.array(im), np.array(ref_im)))
        #im.save(ref_dir / 'xs_blended_noaffine.png')

    def test_save(self):
        ref_dir = Path('data/3d_images')
        tmp_dir = ref_dir/'tmp'
        tmp_dir.mkdir(exist_ok=True)
        fn_cls_dict = {'x.nii': nib.Nifti1Image, 'x.nii.gz': nib.Nifti1Image, 'x.npy': None,
                       'x.img': nib.Spm2AnalyzeImage, 'x.mgh': nib.MGHImage}
        for fn, cls in fn_cls_dict.items():
            filepath = tmp_dir/fn
            self.x.save(filepath, cls=cls)
            self.assertTrue(files_are_equal(filepath, ref_dir/fn))
            #self.x.save(ref_dir / fn, cls=cls)
        shutil.rmtree(tmp_dir)

    def test_create(self):
        img = nib.as_closest_canonical(nib.load(TEMPLATE_FN))
        arr = img.get_fdata()[None]  # add channel dim. to 3d nifti
        self.assertEqual(tuple(self.x.shape), arr.shape)
        x_from_arr = TensorImageBase3d.create(arr, affine=self.x.affine, header=img.header, filepath=TEMPLATE_FN)
        self.assertTrue(torch.equal(self.x, x_from_arr))
        self.assertTrue(np.array_equal(self.x.affine, x_from_arr.affine))
        self.assertEqual(self.x.header.binaryblock, x_from_arr.header.binaryblock)
        self.assertEqual(self.x.filepath, x_from_arr.filepath)
        t = torch.from_numpy(arr)
        x_from_tensor = TensorImageBase3d.create(t, affine=self.x.affine, header=img.header, filepath=TEMPLATE_FN)
        self.assertTrue(torch.equal(self.x, x_from_tensor))
        self.assertTrue(np.array_equal(self.x.affine, x_from_tensor.affine))
        self.assertEqual(self.x.header.binaryblock, x_from_tensor.header.binaryblock)
        self.assertEqual(self.x.filepath, x_from_tensor.filepath)
        x_from_path = TensorImageBase3d.create(Path(TEMPLATE_FN))
        self.assertTrue(torch.equal(self.x, x_from_path))
        self.assertTrue(np.array_equal(self.x.affine, x_from_path.affine))
        self.assertEqual(self.x.header.binaryblock, x_from_path.header.binaryblock)
        self.assertEqual(self.x.filepath, x_from_path.filepath)


if __name__ == '__main__':
    unittest.main()
