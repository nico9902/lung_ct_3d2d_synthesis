import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.det.CPMNetv2.export_synthesis_predictions import export_predictions, normalize_predictions


class PredictionExportTests(unittest.TestCase):
    def rows(self, uid="scan-a"):
        return pd.DataFrame([dict(seriesuid=uid, coordX=12, coordY=23, coordZ=34,
                                  probability=0.4, w=4, h=10, d=6)])

    def test_coordinates_scores_and_radius(self):
        row = normalize_predictions(self.rows()).iloc[0]
        self.assertEqual((row.coordX, row.coordY, row.coordZ), (12, 23, 34))
        self.assertEqual(row.radius, 5)
        self.assertEqual(row.probability, 0.4)  # Export must not apply the 0.50 cutoff.

    def test_empty_predictions_keep_schema(self):
        self.assertEqual(list(normalize_predictions(self.rows().iloc[:0]).columns),
                         ["seriesuid", "coordZ", "coordY", "coordX", "radius", "probability"])

    def test_invalid_boxes_rejected(self):
        for column, value in [("w", -1), ("coordX", float("nan")), ("probability", 1.2)]:
            with self.subTest(column=column), self.assertRaises(ValueError):
                normalize_predictions(self.rows().assign(**{column: value}))

    def test_oof_validation_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "detector/run_fold0/predictions"
            source.mkdir(parents=True)
            self.rows().to_csv(source / "test_predictions_test.csv", index=False)
            splits = root / "splits"
            splits.mkdir()
            csv = splits / "luna16_fold0.csv"
            pd.DataFrame([dict(seriesuid="scan-a", split="train")]).to_csv(csv, index=False)
            output, pooled = root / "output", root / "pooled.csv"
            with self.assertRaises(ValueError):
                export_predictions(root / "detector", output, splits, [0], pooled)
            self.assertFalse(output.exists())
            pd.DataFrame([dict(seriesuid="scan-b", split="test")]).to_csv(csv, index=False)
            with self.assertRaisesRegex(ValueError, "Non-test"):
                export_predictions(root / "detector", output, splits, [0], pooled)
            self.assertFalse(pooled.exists())
            pd.DataFrame([dict(seriesuid="scan-a", split="test")]).to_csv(csv, index=False)
            export_predictions(root / "detector", output, splits, [0], pooled)
            self.assertTrue((output / "run_fold0/predictions/test_predictions.csv").exists())
            row = pd.read_csv(pooled).iloc[0]
            self.assertEqual((row.fold, row.x, row.y, row.z), (0, 12, 23, 34))


if __name__ == "__main__":
    unittest.main()
