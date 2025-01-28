import unittest
from torch import rand, equal, tensor, allclose, linspace

from niftiai import augment, TensorMask3d, TensorImage3d
SHAPES = ((1, 1, 7, 8, 6), (1, 3, 7, 8, 6), (2, 1, 7, 8, 6), (3, 2, 7, 8, 6))


class TestFlip3d(unittest.TestCase):
    def test_encodes(self):
        flip = augment.Flip3d()
        for cls in [TensorImage3d, TensorMask3d]:
            for shape in SHAPES:
                shape = shape[1:] if cls.__name__ == 'TensorMask3d' else shape
                x = cls(rand(*shape))
                x_flipped = flip.encodes(x)
                self.assertEqual(x_flipped.shape, x.shape)
                x_flipped = flip.encodes(x[0])
                self.assertEqual(x_flipped.shape, x[0].shape)


class TestDihedral3d(unittest.TestCase):
    def test_encodes(self):
        for cls in [TensorImage3d, TensorMask3d]:
            for shape in SHAPES:
                shape = shape[1:] if cls.__name__ == 'TensorMask3d' else shape
                dihedral = augment.Dihedral3d()
                x = cls(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
                x_out = dihedral.encodes(x)
                self.assertTrue(tensor(x_out.shape).prod() == tensor(x.shape).prod())
                self.assertTrue(x_out.shape[:-3] == x.shape[:-3])
                x_out = dihedral.encodes(x[0])
                self.assertTrue(tensor(x_out.shape).prod() == tensor(x[0].shape).prod())
                self.assertTrue(x_out.shape[:-3] == x[0].shape[:-3])


class TestCrop3d(unittest.TestCase):
    def test_encodes(self):
        for cls in [TensorImage3d, TensorMask3d]:
            for shape in SHAPES:
                shape = shape[1:] if cls.__name__ == 'TensorMask3d' else shape
                x = cls(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
                size = tuple(s - 2 for s in shape[-3:])
                crop = augment.Crop3d(size=size, p=1, batch=False)
                crop.before_call([x, ], split_idx=0)
                x_cropped = crop.encodes(x)
                self.assertEqual(x_cropped.shape, (*x.shape[:-3], *size))
                crop = augment.Crop3d(size=size, p=1, batch=True, item=True)
                crop.before_call([x[0], ], split_idx=0)
                x_cropped = crop.encodes(x[0])
                self.assertEqual(x_cropped.shape, (*x[0].shape[:-3], *size))
                crop = augment.Crop3d(size=size, p=1, batch=True)
                crop.before_call([x, ], split_idx=0)
                x_cropped = crop.encodes(x)
                self.assertEqual(x_cropped.shape, (*x.shape[:-3], *size))
                crop = augment.Crop3d(size=size, p=1, batch=True, item=True)
                crop.before_call([x[0], ], split_idx=0)
                x_cropped = crop.encodes(x[0])
                self.assertEqual(x_cropped.shape, (*x[0].shape[:-3], *size))


class TestZoom3d(unittest.TestCase):
    def test_encodes(self):
        for cls in [TensorImage3d, TensorMask3d]:
            for shape in SHAPES:
                shape = shape[1:] if cls.__name__ == 'TensorMask3d' else shape
                x = cls(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
                size = tuple(s - 2 for s in shape[-3:])
                zoom = augment.Zoom3d(size=size, p=1, batch=False, upsample=1)
                zoom.before_call([x, ], split_idx=0)
                x_zoomed = zoom.encodes(x)
                self.assertEqual(x_zoomed.shape[-3:], size)
                self.assertEqual(x_zoomed.shape[:-3], x.shape[:-3])
                zoom = augment.Zoom3d(max_zoom=0, p=1, batch=False, upsample=1, item=True)
                zoom.before_call([x[0], ], split_idx=0)
                x_zoomed = zoom.encodes(x[0])
                self.assertEqual(x_zoomed.shape[-3:], x.shape[-3:])
                self.assertEqual(x_zoomed.shape[:-3], x[0].shape[:-3])
                zoom = augment.Zoom3d(max_zoom=0, p=1, batch=True, upsample=1)
                zoom.before_call([x, ], split_idx=0)
                x_zoomed = zoom.encodes(x)
                self.assertTrue(allclose(x, x_zoomed))
                zoom = augment.Zoom3d(max_zoom=0, p=1, batch=True, upsample=2)
                zoom.before_call([x, ], split_idx=0)
                x_zoomed = zoom.encodes(x)
                self.assertTrue(allclose(x, x_zoomed, rtol=5e-2, atol=2e-1))


class TestRotate3d(unittest.TestCase):
    def test_encodes(self):
        for cls in [TensorImage3d, TensorMask3d]:
            for shape in SHAPES:
                shape = shape[1:] if cls.__name__ == 'TensorMask3d' else shape
                x = cls(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
                size = tuple(s - 2 for s in shape[-3:])
                rotate = augment.Rotate3d(size=size, p=1, batch=False, upsample=1)
                rotate.before_call([x, ], split_idx=0)
                x_rotated = rotate.encodes(x)
                self.assertEqual(x_rotated.shape[-3:], size)
                self.assertEqual(x_rotated.shape[:-3], x.shape[:-3])
                rotate = augment.Rotate3d(max_rotate=0, p=1, batch=False, upsample=1, item=True)
                rotate.before_call([x[0], ], split_idx=0)
                x_rotated = rotate.encodes(x[0])
                self.assertEqual(x_rotated.shape[-3:], x.shape[-3:])
                self.assertEqual(x_rotated.shape[:-3], x[0].shape[:-3])
                rotate = augment.Rotate3d(max_rotate=0, p=1, batch=True, upsample=1)
                rotate.before_call([x, ], split_idx=0)
                x_rotated = rotate.encodes(x)
                self.assertTrue(allclose(x, x_rotated))
                rotate = augment.Rotate3d(max_rotate=0, p=1, batch=True, upsample=2)
                rotate.before_call([x, ], split_idx=0)
                x_rotated = rotate.encodes(x)
                self.assertTrue(allclose(x, x_rotated, rtol=5e-2, atol=2e-1))


class TestTranslate3d(unittest.TestCase):
    def test_encodes(self):
        for cls in [TensorImage3d, TensorMask3d]:
            for shape in SHAPES:
                shape = shape[1:] if cls.__name__ == 'TensorMask3d' else shape
                x = cls(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
                size = tuple(s - 2 for s in shape[-3:])
                translate = augment.Translate3d(size=size, p=1, batch=False, upsample=1)
                translate.before_call([x, ], split_idx=0)
                x_translated = translate.encodes(x)
                self.assertEqual(x_translated.shape[-3:], size)
                self.assertEqual(x_translated.shape[:-3], x.shape[:-3])
                translate = augment.Translate3d(max_translate=0, p=1, batch=False, upsample=1, item=True)
                translate.before_call([x[0], ], split_idx=0)
                x_translated = translate.encodes(x[0])
                self.assertEqual(x_translated.shape[-3:], x.shape[-3:])
                self.assertEqual(x_translated.shape[:-3], x[0].shape[:-3])
                translate = augment.Translate3d(max_translate=0, p=1, batch=True, upsample=1)
                translate.before_call([x, ], split_idx=0)
                x_translated = translate.encodes(x)
                self.assertTrue(allclose(x, x_translated))
                translate = augment.Translate3d(max_translate=0, p=1, batch=True, upsample=2)
                translate.before_call([x, ], split_idx=0)
                x_translated = translate.encodes(x)
                self.assertTrue(allclose(x, x_translated, rtol=5e-2, atol=2e-1))


class TestShear3d(unittest.TestCase):
    def test_encodes(self):
        for cls in [TensorImage3d, TensorMask3d]:
            for shape in SHAPES:
                shape = shape[1:] if cls.__name__ == 'TensorMask3d' else shape
                x = cls(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
                size = tuple(s - 2 for s in shape[-3:])
                shear = augment.Shear3d(size=size, p=1, batch=False, upsample=1)
                shear.before_call([x, ], split_idx=0)
                x_sheared = shear.encodes(x)
                self.assertEqual(x_sheared.shape[-3:], size)
                self.assertEqual(x_sheared.shape[:-3], x.shape[:-3])
                shear = augment.Shear3d(max_shear=0, p=1, batch=False, upsample=1, item=True)
                shear.before_call([x[0], ], split_idx=0)
                x_sheared = shear.encodes(x[0])
                self.assertEqual(x_sheared.shape[-3:], x.shape[-3:])
                self.assertEqual(x_sheared.shape[:-3], x[0].shape[:-3])
                shear = augment.Shear3d(max_shear=0, p=1, batch=True, upsample=1)
                shear.before_call([x, ], split_idx=0)
                x_sheared = shear.encodes(x)
                self.assertTrue(allclose(x, x_sheared))
                shear = augment.Shear3d(max_shear=0, p=1, batch=True, upsample=2)
                shear.before_call([x, ], split_idx=0)
                x_sheared = shear.encodes(x)
                self.assertTrue(allclose(x, x_sheared, rtol=5e-2, atol=2e-1))


class TestAffine3d(unittest.TestCase):
    def test_encodes(self):
        for cls in [TensorImage3d, TensorMask3d]:
            for shape in SHAPES:
                shape = shape[1:] if cls.__name__ == 'TensorMask3d' else shape
                x = cls(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
                size = tuple(s - 2 for s in shape[-3:])
                affine = augment.Affine3d(size=size, p=1, batch=False, upsample=1)
                affine.before_call([x, ], split_idx=0)
                x_out = affine.encodes(x)
                self.assertEqual(x_out.shape[-3:], size)
                self.assertEqual(x_out.shape[:-3], x.shape[:-3])
                affine = augment.Affine3d(max_zoom=0, max_rotate=0, max_translate=0, max_shear=0,
                                          p=1, batch=False, upsample=1, item=True)
                affine.before_call([x[0], ], split_idx=0)
                x_out = affine.encodes(x[0])
                self.assertEqual(x_out.shape[-3:], x.shape[-3:])
                self.assertEqual(x_out.shape[:-3], x[0].shape[:-3])
                affine = augment.Affine3d(max_zoom=0, max_rotate=0, max_translate=0, max_shear=0,
                                          p=1, batch=True, upsample=1)
                affine.before_call([x, ], split_idx=0)
                x_out = affine.encodes(x)
                self.assertTrue(allclose(x, x_out))
                affine = augment.Affine3d(max_zoom=0, max_rotate=0, max_translate=0, max_shear=0,
                                          p=1, batch=True, upsample=2)
                affine.before_call([x, ], split_idx=0)
                x_out = affine.encodes(x)
                self.assertTrue(allclose(x, x_out, rtol=5e-2, atol=2e-1))


class TestWarp3d(unittest.TestCase):
    def test_encodes(self):
        for cls in [TensorImage3d, TensorMask3d]:
            for shape in SHAPES:
                shape = shape[1:] if cls.__name__ == 'TensorMask3d' else shape
                x = cls(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
                size = tuple(s - 2 for s in shape[-3:])
                warp = augment.Warp3d(size=size, p=1, batch=False, upsample=1)
                warp.before_call([x, ], split_idx=0)
                x_out = warp.encodes(x)
                self.assertEqual(x_out.shape[-3:], size)
                self.assertEqual(x_out.shape[:-3], x.shape[:-3])
                warp = augment.Warp3d(max_magnitude=0, p=1, batch=False, upsample=1, item=True)
                warp.before_call([x[0], ], split_idx=0)
                x_out = warp.encodes(x[0])
                self.assertEqual(x_out.shape[-3:], x.shape[-3:])
                self.assertEqual(x_out.shape[:-3], x[0].shape[:-3])
                warp = augment.Warp3d(max_magnitude=0, p=1, batch=True, upsample=1)
                warp.before_call([x, ], split_idx=0)
                x_out = warp.encodes(x)
                self.assertTrue(allclose(x, x_out))
                warp = augment.Warp3d(max_magnitude=0, p=1, batch=True, upsample=2)
                warp.before_call([x, ], split_idx=0)
                x_out = warp.encodes(x)
                self.assertTrue(allclose(x, x_out, rtol=5e-2, atol=2e-1))


class TestAffineWarp3d(unittest.TestCase):
    def test_encodes(self):
        for cls in [TensorImage3d, TensorMask3d]:
            for shape in SHAPES:
                shape = shape[1:] if cls.__name__ == 'TensorMask3d' else shape
                x = cls(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
                size = tuple(s - 2 for s in shape[-3:])
                affinewarp = augment.AffineWarp3d(size=size, p_affine=1, p_warp=1, batch=False, upsample=1)
                affinewarp.before_call([x, ], split_idx=0)
                x_out = affinewarp.encodes(x)
                self.assertEqual(x_out.shape[-3:], size)
                self.assertEqual(x_out.shape[:-3], x.shape[:-3])
                affinewarp = augment.AffineWarp3d(max_zoom=0, max_rotate=0, max_translate=0, max_shear=0, max_warp=0,
                                                  p_affine=1, p_warp=1, batch=False, upsample=1, item=True)
                affinewarp.before_call([x[0], ], split_idx=0)
                x_out = affinewarp.encodes(x[0])
                self.assertEqual(x_out.shape[-3:], x.shape[-3:])
                self.assertEqual(x_out.shape[:-3], x[0].shape[:-3])
                affinewarp = augment.AffineWarp3d(max_zoom=0, max_rotate=0, max_translate=0, max_shear=0, max_warp=0,
                                                  p_affine=1, p_warp=1, batch=True, upsample=1)
                affinewarp.before_call([x, ], split_idx=0)
                x_out = affinewarp.encodes(x)
                self.assertTrue(allclose(x, x_out))
                affinewarp = augment.AffineWarp3d(max_zoom=0, max_rotate=0, max_translate=0, max_shear=0, max_warp=0,
                                                  p_affine=1, p_warp=1, batch=True, upsample=2)
                affinewarp.before_call([x, ], split_idx=0)
                x_out = affinewarp.encodes(x)
                self.assertTrue(allclose(x, x_out, rtol=5e-2, atol=2e-1))


class TestBiasField3d(unittest.TestCase):
    def test_encodes(self):
        for shape in SHAPES:
            x = TensorImage3d(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
            bias_field = augment.BiasField3d(max_intensity=.0, p=1)
            bias_field.before_call([x, ], split_idx=0)
            x_out = bias_field.encodes(x)
            self.assertTrue(equal(x, x_out))
            bias_field.before_call([x[0], ], split_idx=0)
            x_out = bias_field.encodes(x[0])
            self.assertTrue(equal(x[0], x_out))
            bias_field = augment.BiasField3d(max_intensity=.0, p=1, batch=True)
            bias_field.before_call([x, ], split_idx=0)
            x_out = bias_field.encodes(x)
            self.assertTrue(equal(x, x_out))
            bias_field = augment.BiasField3d(p=1, k_size=(2, 2, 2))
            bias_field.before_call([x, ], split_idx=0)
            x_out = bias_field.encodes(x)
            self.assertFalse(equal(x, x_out))


class TestContrast3d(unittest.TestCase):
    def test_encodes(self):
        for shape in SHAPES:
            x = TensorImage3d(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
            contrast = augment.Contrast3d(max_lighting=.0, p=1)
            contrast.before_call([x, ], split_idx=0)
            x_out = contrast.encodes(x)
            self.assertTrue(equal(x, x_out))
            contrast.before_call([x[0], ], split_idx=0)
            x_out = contrast.encodes(x[0])
            self.assertTrue(equal(x[0], x_out))
            contrast = augment.Contrast3d(max_lighting=.0, p=1, batch=True)
            contrast.before_call([x, ], split_idx=0)
            x_out = contrast.encodes(x)
            self.assertTrue(equal(x, x_out))
            contrast = augment.Contrast3d(p=1)
            contrast.before_call([x, ], split_idx=0)
            x_out = contrast.encodes(x)
            self.assertFalse(equal(x, x_out))


class TestChiNoise3d(unittest.TestCase):
    def test_encodes(self):
        for shape in SHAPES:
            x = TensorImage3d(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
            noise = augment.ChiNoise3d(max_intensity=.0, p=1)
            noise.before_call([x, ], split_idx=0)
            x_out = noise.encodes(x)
            self.assertTrue(allclose(x, x_out))
            noise.before_call([x[0], ], split_idx=0)
            x_out = noise.encodes(x[0])
            self.assertTrue(allclose(x[0], x_out))
            noise = augment.ChiNoise3d(max_intensity=.0, p=1, batch=True)
            noise.before_call([x, ], split_idx=0)
            x_out = noise.encodes(x)
            self.assertTrue(allclose(x, x_out))
            noise = augment.ChiNoise3d(p=1)
            noise.before_call([x, ], split_idx=0)
            x_out = noise.encodes(x)
            self.assertFalse(allclose(x, x_out))


class TestDownsample3d(unittest.TestCase):
    def test_encodes(self):
        for shape in SHAPES:
            x = TensorImage3d(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
            downsample = augment.Downsample3d(max_downsample=.0, p=1)
            downsample.before_call([x, ], split_idx=0)
            x_out = downsample.encodes(x)
            self.assertTrue(equal(x, x_out))
            downsample = augment.Downsample3d(max_downsample=.0, p=1, item=True)
            downsample.before_call([x[0], ], split_idx=0)
            x_out = downsample.encodes(x[0])
            self.assertTrue(equal(x[0], x_out))
            downsample = augment.Downsample3d(max_downsample=.0, p=1, batch=True)
            downsample.before_call([x, ], split_idx=0)
            x_out = downsample.encodes(x)
            self.assertTrue(equal(x, x_out))
            downsample = augment.Downsample3d(p=1)
            downsample.before_call([x, ], split_idx=0)
            x_out = downsample.encodes(x)
            self.assertFalse(equal(x, x_out))


class TestGhosting3d(unittest.TestCase):
    def test_encodes(self):
        for shape in SHAPES:
            x = TensorImage3d(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
            ghosting = augment.Ghosting3d(max_intensity=.0, p=1)
            ghosting.before_call([x, ], split_idx=0)
            x_out = ghosting.encodes(x)
            self.assertTrue(allclose(x, x_out, atol=1e-7))
            ghosting.before_call([x[0], ], split_idx=0)
            x_out = ghosting.encodes(x[0])
            self.assertTrue(allclose(x[0], x_out, atol=1e-7))
            ghosting = augment.Ghosting3d(max_intensity=.0, p=1, batch=True)
            ghosting.before_call([x, ], split_idx=0)
            x_out = ghosting.encodes(x)
            self.assertTrue(allclose(x, x_out, atol=1e-7))
            ghosting = augment.Ghosting3d(p=1)
            ghosting.before_call([x, ], split_idx=0)
            x_out = ghosting.encodes(x)
            self.assertFalse(allclose(x, x_out, atol=1e-7))


class TestSpike3d(unittest.TestCase):
    def test_encodes(self):
        for shape in SHAPES:
            x = TensorImage3d(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
            spike = augment.Spike3d(max_intensity=.0, p=1)
            spike.before_call([x, ], split_idx=0)
            x_out = spike.encodes(x)
            self.assertTrue(allclose(x, x_out, atol=5e-7))
            spike.before_call([x[0], ], split_idx=0)
            x_out = spike.encodes(x[0])
            self.assertTrue(allclose(x[0], x_out, atol=5e-7))
            spike = augment.Spike3d(max_intensity=.0, p=1, batch=True)
            spike.before_call([x, ], split_idx=0)
            x_out = spike.encodes(x)
            self.assertTrue(allclose(x, x_out, atol=5e-7))


class TestRinging3d(unittest.TestCase):
    def test_encodes(self):
        for shape in SHAPES:
            x = TensorImage3d(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
            ringing = augment.Ringing3d(max_intensity=.0, p=1)
            ringing.before_call([x, ], split_idx=0)
            x_out = ringing.encodes(x)
            self.assertTrue(allclose(x, x_out, atol=1e-7))
            ringing.before_call([x[0], ], split_idx=0)
            x_out = ringing.encodes(x[0])
            self.assertTrue(allclose(x[0], x_out, atol=1e-7))
            ringing = augment.Ringing3d(max_intensity=.0, p=1, batch=True)
            ringing.before_call([x, ], split_idx=0)
            x_out = ringing.encodes(x)
            self.assertTrue(allclose(x, x_out, atol=1e-7))
            ringing = augment.Ringing3d(p=1)
            ringing.before_call([x, ], split_idx=0)
            x_out = ringing.encodes(x)
            self.assertFalse(allclose(x, x_out, atol=1e-7))


class TestMotion3d(unittest.TestCase):
    def test_encodes(self):
        for shape in SHAPES:
            x = TensorImage3d(linspace(0, 1, tensor(shape).prod().item()).view(*shape))
            motion = augment.Motion3d(max_intensity=.0, p=1)
            motion.before_call([x, ], split_idx=0)
            x_out = motion.encodes(x)
            self.assertTrue(allclose(x, x_out, atol=1e-7))
            motion = augment.Motion3d(max_intensity=.0, p=1, item=True)
            motion.before_call([x[0], ], split_idx=0)
            x_out = motion.encodes(x[0])
            self.assertTrue(allclose(x[0], x_out, atol=1e-7))
            motion = augment.Motion3d(max_intensity=.0, p=1, batch=True)
            motion.before_call([x, ], split_idx=0)
            x_out = motion.encodes(x)
            self.assertTrue(allclose(x, x_out, atol=1e-7))


if __name__ == '__main__':
    unittest.main()
