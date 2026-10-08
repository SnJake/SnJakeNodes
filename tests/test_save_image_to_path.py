import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

NODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NODE_ROOT))
sys.path.insert(0, str(NODE_ROOT.parents[1]))
from utils_nodes import SaveImageToPath


class SaveImageTests(unittest.TestCase):
    def save(self, image, **kwargs):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "nested" / "image.png"
        with contextlib.redirect_stdout(io.StringIO()):
            result = SaveImageToPath().save_image(image, str(path), kwargs.pop("save_workflow", True), **kwargs)
        self.assertEqual(result, ())
        self.assertTrue(path.exists(), "SaveImageToPath did not create the PNG")
        return Image.open(path)

    def test_rounding_and_clipping_preserve_color(self):
        pixels = torch.tensor([-0.1, 0.1, 0.5, 0.75, 1.1]).view(1, 1, 5, 1).expand(1, 2, 5, 3)
        with self.save(pixels) as image:
            expected = np.broadcast_to(np.array([0, 26, 128, 191, 255], dtype=np.uint8)[None, :, None], (2, 5, 3))
            np.testing.assert_array_equal(np.asarray(image), expected)

    def test_small_hwc_image_preserves_geometry_and_channels(self):
        pixels = torch.zeros(1, 3, 7, 3)
        pixels[..., 0] = 1
        pixels[:, 1, :, 1] = 1
        with self.save(pixels) as image:
            self.assertEqual(image.size, (7, 3))
            np.testing.assert_array_equal(np.asarray(image), (pixels[0].numpy() * 255).astype(np.uint8))

    def test_bf16_and_requires_grad_can_be_saved(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                pixels = torch.full((1, 2, 5, 3), 0.5, dtype=dtype, requires_grad=True)
                with self.save(pixels) as image:
                    np.testing.assert_array_equal(np.asarray(image), np.full((2, 5, 3), 128, np.uint8))

    def test_rgba_grayscale_and_chw(self):
        for channels, mode in ((4, "RGBA"), (1, "L")):
            with self.subTest(mode=mode), self.save(torch.full((1, 2, 5, channels), 0.5)) as image:
                self.assertEqual(image.mode, mode)
                self.assertEqual(image.size, (5, 2))
                self.assertTrue(np.all(np.asarray(image) == 128))
        chw = torch.ones(1, 3, 6, 7)
        with self.save(chw) as image:
            self.assertEqual(image.size, (7, 6))
            self.assertTrue(np.all(np.asarray(image) == 255))

    def test_workflow_metadata_toggle(self):
        prompt = {"1": {"class_type": "Example", "inputs": {}}}
        workflow = {"nodes": [{"id": 1}]}
        pixels = torch.zeros(1, 2, 5, 3)
        with self.save(pixels, prompt=prompt, extra_pnginfo={"workflow": workflow}) as image:
            self.assertEqual(json.loads(image.info["prompt"]), prompt)
            self.assertEqual(json.loads(image.info["workflow"]), workflow)
        with self.save(pixels, save_workflow=False, prompt=prompt, extra_pnginfo={"workflow": workflow}) as image:
            self.assertNotIn("prompt", image.info)
            self.assertNotIn("workflow", image.info)


if __name__ == "__main__":
    unittest.main()
