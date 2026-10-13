"""Map the 888 LUNA16 MHD series to their LIDC-IDRI patient identifiers.

This is the minimal mapping required by generate_luna_malignancy_csv.py when
``--match-source pylidc_mhd`` is used. It does not assign malignancy labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .generate_luna_malignancy_csv import import_pylidc


def build_mapping(luna_mhd_root: Path, output_csv: Path) -> pd.DataFrame:
    scans_root = luna_mhd_root / "subsets" if (luna_mhd_root / "subsets").is_dir() else luna_mhd_root
    seriesuids = sorted(path.stem for path in scans_root.rglob("*.mhd"))
    if not seriesuids:
        raise ValueError(f"No LUNA16 .mhd files found under {luna_mhd_root}")
    if len(seriesuids) != len(set(seriesuids)):
        raise ValueError("Duplicate LUNA16 seriesuid among .mhd files")

    pl = import_pylidc()
    scans = pl.query(pl.Scan).all()
    lidc_by_series = {str(scan.series_instance_uid): str(scan.patient_id) for scan in scans}
    missing = sorted(set(seriesuids) - set(lidc_by_series))
    if missing:
        raise ValueError(f"{len(missing)} LUNA16 series have no pylidc scan; first: {missing[0]}")

    mapping = pd.DataFrame(
        {"seriesuid": seriesuids, "lidc_id": [lidc_by_series[seriesuid] for seriesuid in seriesuids]}
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_csv, index=False)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--luna-mhd-root", type=Path, default=Path("data/raw/LUNA16"))
    parser.add_argument(
        "--output-csv", type=Path, default=Path("data/raw/LUNA16/luna16_all_seriesuid_to_lidc_id.csv")
    )
    args = parser.parse_args()
    result = build_mapping(args.luna_mhd_root, args.output_csv)
    print(f"wrote {len(result)} LUNA16 series to {args.output_csv}")


if __name__ == "__main__":
    main()
