import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from src.luna16_3dino_3d import train_frozen_probe as probe


class FrozenProbeCacheTests(unittest.TestCase):
    def test_whole_volume_extraction_builds_and_reuses_missing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volume = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
            sitk.WriteImage(sitk.GetImageFromArray(volume), str(root / "scan.nii.gz"))
            rows = pd.DataFrame([dict(seriesuid="scan", image_path="scan.nii.gz")])
            args = argparse.Namespace(device="cpu", output_dir=root / "features", data_root=root,
                                      windows_cache_dir=root / "windows", shard_index=0, num_shards=1,
                                      code_dir=root, checkpoint_path=root / "teacher.pth", micro_batch_windows=1)
            captured = []
            def encode(_backbone, windows, *_args):
                captured.append(windows.clone())
                return torch.zeros(2 * probe.EMBED_DIM)
            with patch.object(probe, "build_backbone", return_value=torch.nn.Identity()), \
                 patch.object(probe, "load_checkpoint"), patch.object(probe, "encode_chunks", side_effect=encode):
                probe.extract_features(args, "whole", rows, None)
                (root / "features/whole/features/scan.pt").unlink()
                (root / "scan.nii.gz").unlink()  # Second pass must work from cached windows alone.
                probe.extract_features(args, "whole", rows, None)
            self.assertEqual(captured[0].shape, (1, 1, 112, 112, 112))
            torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(captured[0]).all())


if __name__ == "__main__":
    unittest.main()
