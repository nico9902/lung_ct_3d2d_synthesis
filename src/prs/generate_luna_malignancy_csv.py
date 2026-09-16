from __future__ import annotations

import argparse
import configparser
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser

XML_NS = {"nih": "http://www.nih.gov"}


@dataclass(frozen=True)
class ScanGeometry:
    image_position: np.ndarray
    row_direction: np.ndarray
    column_direction: np.ndarray
    normal_direction: np.ndarray
    row_spacing: float
    column_spacing: float
    slice_positions: np.ndarray


@dataclass(frozen=True)
class ClusterInfo:
    center_world: np.ndarray | None
    mean_malignancy: float
    num_reader_annotations: int


@dataclass
class XmlReaderNodule:
    seriesuid: str
    x_px: float
    y_px: float
    z_mm: float
    diameter_mm: float
    malignancy: float


@dataclass
class XmlCluster:
    seriesuid: str
    x_px: float
    y_px: float
    z_mm: float
    diameter_mm: float
    malignancies: list[float]

    @property
    def mean_malignancy(self) -> float:
        return float(np.mean(self.malignancies))

    @property
    def num_reader_annotations(self) -> int:
        return len(self.malignancies)


def text_or_empty(element: ET.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def import_pylidc():
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import pylidc as pl

    return pl


def import_pydicom():
    import pydicom

    return pydicom


def import_sitk():
    import SimpleITK as sitk

    return sitk


def xml_seriesuid(root: ET.Element) -> str:
    return text_or_empty(root.find(".//nih:ResponseHeader/nih:SeriesInstanceUid", XML_NS))


def xml_reader_nodules(xml_path: Path, pixel_spacing: float, slice_thickness: float) -> list[XmlReaderNodule]:
    root = ET.parse(xml_path).getroot()
    seriesuid = xml_seriesuid(root)
    if not seriesuid:
        return []

    nodules = []
    for nodule in root.findall(".//nih:unblindedReadNodule", XML_NS):
        malignancy_text = text_or_empty(nodule.find("./nih:characteristics/nih:malignancy", XML_NS))
        if not malignancy_text:
            continue
        try:
            malignancy = float(malignancy_text)
        except ValueError:
            continue

        points = []
        for roi in nodule.findall("./nih:roi", XML_NS):
            z_text = text_or_empty(roi.find("./nih:imageZposition", XML_NS))
            if not z_text:
                continue
            try:
                z = float(z_text)
            except ValueError:
                continue
            for edge in roi.findall("./nih:edgeMap", XML_NS):
                x_text = text_or_empty(edge.find("./nih:xCoord", XML_NS))
                y_text = text_or_empty(edge.find("./nih:yCoord", XML_NS))
                if not x_text or not y_text:
                    continue
                try:
                    points.append((float(x_text), float(y_text), z))
                except ValueError:
                    continue

        if not points:
            continue
        coords = np.asarray(points, dtype=np.float64)
        x_span_mm = (coords[:, 0].max() - coords[:, 0].min() + 1.0) * pixel_spacing
        y_span_mm = (coords[:, 1].max() - coords[:, 1].min() + 1.0) * pixel_spacing
        z_span_mm = (coords[:, 2].max() - coords[:, 2].min()) + slice_thickness
        nodules.append(
            XmlReaderNodule(
                seriesuid=seriesuid,
                x_px=float(coords[:, 0].mean()),
                y_px=float(coords[:, 1].mean()),
                z_mm=float(coords[:, 2].mean()),
                diameter_mm=float(max(x_span_mm, y_span_mm, z_span_mm)),
                malignancy=malignancy,
            )
        )
    return nodules


def update_xml_cluster(cluster: XmlCluster, nodule: XmlReaderNodule) -> None:
    n = len(cluster.malignancies)
    cluster.x_px = (cluster.x_px * n + nodule.x_px) / (n + 1)
    cluster.y_px = (cluster.y_px * n + nodule.y_px) / (n + 1)
    cluster.z_mm = (cluster.z_mm * n + nodule.z_mm) / (n + 1)
    cluster.diameter_mm = (cluster.diameter_mm * n + nodule.diameter_mm) / (n + 1)
    cluster.malignancies.append(nodule.malignancy)


def cluster_xml_reader_nodules(
    nodules: list[XmlReaderNodule],
    pixel_spacing: float,
    xy_tolerance_mm: float,
    z_tolerance_mm: float,
) -> list[XmlCluster]:
    clusters: list[XmlCluster] = []
    for nodule in sorted(nodules, key=lambda n: (n.z_mm, n.y_px, n.x_px)):
        best_idx = None
        best_distance = None
        for idx, cluster in enumerate(clusters):
            xy_distance_mm = float(
                np.hypot(cluster.x_px - nodule.x_px, cluster.y_px - nodule.y_px) * pixel_spacing
            )
            z_distance_mm = abs(cluster.z_mm - nodule.z_mm)
            diameter_gate = max(cluster.diameter_mm, nodule.diameter_mm, xy_tolerance_mm)
            if xy_distance_mm > diameter_gate or z_distance_mm > max(z_tolerance_mm, diameter_gate):
                continue
            distance = xy_distance_mm + z_distance_mm
            if best_distance is None or distance < best_distance:
                best_idx = idx
                best_distance = distance

        if best_idx is None:
            clusters.append(
                XmlCluster(
                    seriesuid=nodule.seriesuid,
                    x_px=nodule.x_px,
                    y_px=nodule.y_px,
                    z_mm=nodule.z_mm,
                    diameter_mm=nodule.diameter_mm,
                    malignancies=[nodule.malignancy],
                )
            )
        else:
            update_xml_cluster(clusters[best_idx], nodule)
    return clusters


def load_xml_clusters_by_series(
    xml_root: Path,
    mapping: pd.DataFrame,
    xy_tolerance_mm: float,
    z_tolerance_mm: float,
) -> dict[str, list[XmlCluster]]:
    spacing_by_series = {
        str(row["seriesuid"]): float(row["pixel_spacing"])
        for _, row in mapping.iterrows()
        if pd.notna(row.get("pixel_spacing"))
    }
    thickness_by_series = {
        str(row["seriesuid"]): float(row["slice_thickness"])
        for _, row in mapping.iterrows()
        if pd.notna(row.get("slice_thickness"))
    }

    reader_nodules_by_series: dict[str, list[XmlReaderNodule]] = {}
    for xml_path in sorted(xml_root.rglob("*.xml")):
        root = ET.parse(xml_path).getroot()
        seriesuid = xml_seriesuid(root)
        if seriesuid not in spacing_by_series or seriesuid not in thickness_by_series:
            continue
        for nodule in xml_reader_nodules(xml_path, spacing_by_series[seriesuid], thickness_by_series[seriesuid]):
            reader_nodules_by_series.setdefault(seriesuid, []).append(nodule)

    return {
        seriesuid: cluster_xml_reader_nodules(
            nodules,
            pixel_spacing=spacing_by_series[seriesuid],
            xy_tolerance_mm=xy_tolerance_mm,
            z_tolerance_mm=z_tolerance_mm,
        )
        for seriesuid, nodules in reader_nodules_by_series.items()
    }


def find_mhd_paths(luna_mhd_root: Path) -> dict[str, Path]:
    mhd_paths = {}
    for path in sorted(luna_mhd_root.rglob("*.mhd")):
        mhd_paths[path.stem] = path
    return mhd_paths


def xml_cluster_center_to_world(cluster: XmlCluster, image) -> np.ndarray:
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)

    z_axis_scale = direction[2, 2] * spacing[2]
    if abs(z_axis_scale) < 1e-8:
        raise ValueError("Cannot recover z index from MHD geometry with near-zero z direction scale.")

    z_index = (
        cluster.z_mm
        - origin[2]
        - direction[2, 0] * cluster.x_px * spacing[0]
        - direction[2, 1] * cluster.y_px * spacing[1]
    ) / z_axis_scale

    return np.asarray(
        image.TransformContinuousIndexToPhysicalPoint(
            (float(cluster.x_px), float(cluster.y_px), float(z_index))
        ),
        dtype=np.float64,
    )


def load_mhd_xml_clusters_by_series(
    xml_root: Path,
    luna_mhd_root: Path,
    mapping: pd.DataFrame,
    xy_tolerance_mm: float,
    z_tolerance_mm: float,
) -> dict[str, list[ClusterInfo]]:
    sitk = import_sitk()
    xml_clusters_by_series = load_xml_clusters_by_series(
        xml_root=xml_root,
        mapping=mapping,
        xy_tolerance_mm=xy_tolerance_mm,
        z_tolerance_mm=z_tolerance_mm,
    )
    mhd_paths = find_mhd_paths(luna_mhd_root)

    clusters_by_series: dict[str, list[ClusterInfo]] = {}
    for seriesuid, xml_clusters in xml_clusters_by_series.items():
        mhd_path = mhd_paths.get(seriesuid)
        if mhd_path is None:
            continue
        image = sitk.ReadImage(str(mhd_path))
        clusters = []
        for xml_cluster in xml_clusters:
            clusters.append(
                ClusterInfo(
                    center_world=xml_cluster_center_to_world(xml_cluster, image),
                    mean_malignancy=xml_cluster.mean_malignancy,
                    num_reader_annotations=xml_cluster.num_reader_annotations,
                )
            )
        clusters_by_series[seriesuid] = clusters
    return clusters_by_series


def pylidc_cluster_center_to_world(annotation_cluster, image) -> np.ndarray:
    centroids = np.asarray([ann.centroid for ann in annotation_cluster], dtype=np.float64)
    row, col, slice_index = centroids.mean(axis=0)
    return np.asarray(
        image.TransformContinuousIndexToPhysicalPoint((float(col), float(row), float(slice_index))),
        dtype=np.float64,
    )


def load_pylidc_mhd_clusters_by_series(luna_mhd_root: Path) -> dict[str, list[ClusterInfo]]:
    pl = import_pylidc()
    sitk = import_sitk()
    mhd_paths = find_mhd_paths(luna_mhd_root)

    clusters_by_series: dict[str, list[ClusterInfo]] = {}
    for scan in pl.query(pl.Scan).all():
        seriesuid = str(scan.series_instance_uid)
        mhd_path = mhd_paths.get(seriesuid)
        if mhd_path is None:
            continue

        image = sitk.ReadImage(str(mhd_path))
        clusters = []
        for annotation_cluster in scan.cluster_annotations(verbose=False):
            malignancies = [float(ann.malignancy) for ann in annotation_cluster if ann.malignancy is not None]
            if not malignancies:
                continue
            clusters.append(
                ClusterInfo(
                    center_world=pylidc_cluster_center_to_world(annotation_cluster, image),
                    mean_malignancy=float(np.mean(malignancies)),
                    num_reader_annotations=len(malignancies),
                )
            )
        clusters_by_series[seriesuid] = clusters
    return clusters_by_series


def match_luna_nodules_to_xml_clusters(
    luna_nodules: pd.DataFrame,
    clusters: list[XmlCluster],
    z_tolerance_mm: float,
    diameter_tolerance_mm: float,
) -> tuple[list[XmlCluster], list[float], list[float]]:
    if luna_nodules.empty or not clusters:
        return [], [], []

    candidates = []
    for luna_idx, (_, nodule) in enumerate(luna_nodules.iterrows()):
        luna_z = float(nodule["coordZ"])
        luna_diameter = float(nodule["diameter_mm"])
        for cluster_idx, cluster in enumerate(clusters):
            z_diff = abs(cluster.z_mm - luna_z)
            diameter_diff = abs(cluster.diameter_mm - luna_diameter)
            diameter_gate = max(diameter_tolerance_mm, 0.75 * luna_diameter)
            if z_diff <= z_tolerance_mm and diameter_diff <= diameter_gate:
                candidates.append((z_diff + 0.25 * diameter_diff, z_diff, diameter_diff, luna_idx, cluster_idx))

    matched_luna = set()
    matched_clusters = set()
    matched = []
    z_diffs = []
    diameter_diffs = []
    for _, z_diff, diameter_diff, luna_idx, cluster_idx in sorted(candidates):
        if luna_idx in matched_luna or cluster_idx in matched_clusters:
            continue
        matched_luna.add(luna_idx)
        matched_clusters.add(cluster_idx)
        matched.append(clusters[cluster_idx])
        z_diffs.append(float(z_diff))
        diameter_diffs.append(float(diameter_diff))

    return matched, z_diffs, diameter_diffs


def scan_geometry(scan) -> ScanGeometry:
    pydicom = import_pydicom()
    dicom_dir = Path(scan.get_path_to_dicom_files())

    slices = []
    for dicom_path in sorted(dicom_dir.iterdir()):
        if not dicom_path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if not hasattr(ds, "ImagePositionPatient") or not hasattr(ds, "ImageOrientationPatient"):
            continue
        position = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
        pixel_spacing = np.asarray(ds.PixelSpacing, dtype=np.float64)
        slices.append((position, orientation, pixel_spacing))

    if not slices:
        raise ValueError(f"No readable DICOM slices found for {scan}.")

    first_position, first_orientation, first_spacing = slices[0]
    row_direction = first_orientation[:3]
    column_direction = first_orientation[3:]
    normal_direction = np.cross(row_direction, column_direction)
    slice_positions = np.asarray(
        sorted(float(np.dot(position, normal_direction)) for position, _, _ in slices),
        dtype=np.float64,
    )

    return ScanGeometry(
        image_position=first_position,
        row_direction=row_direction,
        column_direction=column_direction,
        normal_direction=normal_direction,
        row_spacing=float(first_spacing[0]),
        column_spacing=float(first_spacing[1]),
        slice_positions=slice_positions,
    )


def centroid_to_world(centroid: np.ndarray, geometry: ScanGeometry) -> np.ndarray:
    row, col, slice_index = np.asarray(centroid, dtype=np.float64)
    slice_position = float(np.interp(slice_index, np.arange(len(geometry.slice_positions)), geometry.slice_positions))
    base_position = geometry.image_position + (
        slice_position - float(np.dot(geometry.image_position, geometry.normal_direction))
    ) * geometry.normal_direction

    return (
        base_position
        + col * geometry.column_spacing * geometry.row_direction
        + row * geometry.row_spacing * geometry.column_direction
    )


def clusters_from_scan(scan, geometry: ScanGeometry | None = None) -> list[ClusterInfo]:
    clusters = []
    for annotation_cluster in scan.cluster_annotations(verbose=False):
        malignancies = [float(ann.malignancy) for ann in annotation_cluster if ann.malignancy is not None]
        if not malignancies:
            continue
        center_world = None
        if geometry is not None:
            centers = np.asarray(
                [centroid_to_world(ann.centroid, geometry) for ann in annotation_cluster],
                dtype=np.float64,
            )
            center_world = centers.mean(axis=0)
        clusters.append(
            ClusterInfo(
                center_world=center_world,
                mean_malignancy=float(np.mean(malignancies)),
                num_reader_annotations=len(malignancies),
            )
        )
    return clusters


def match_luna_nodules_to_clusters(
    luna_nodules: pd.DataFrame,
    clusters: list[ClusterInfo],
    min_match_radius_mm: float,
    match_radius_scale: float,
) -> tuple[list[ClusterInfo], list[float], set[int], set[int]]:
    if luna_nodules.empty or not clusters:
        return [], [], set(), set()

    candidates = []
    for luna_idx, (_, nodule) in enumerate(luna_nodules.iterrows()):
        center = np.asarray([nodule["coordX"], nodule["coordY"], nodule["coordZ"]], dtype=np.float64)
        radius = max(float(nodule["diameter_mm"]) * 0.5 * match_radius_scale, min_match_radius_mm)
        for cluster_idx, cluster in enumerate(clusters):
            if cluster.center_world is None:
                continue
            distance = float(np.linalg.norm(cluster.center_world - center))
            if distance <= radius:
                candidates.append((distance, luna_idx, cluster_idx))

    matched_luna = set()
    matched_clusters = set()
    matched = []
    distances = []
    for distance, luna_idx, cluster_idx in sorted(candidates):
        if luna_idx in matched_luna or cluster_idx in matched_clusters:
            continue
        matched_luna.add(luna_idx)
        matched_clusters.add(cluster_idx)
        matched.append(clusters[cluster_idx])
        distances.append(distance)

    return matched, distances, matched_luna, matched_clusters


def recover_unmatched_luna_nodules_to_nearest_clusters(
    luna_nodules: pd.DataFrame,
    clusters: list[ClusterInfo],
    matched_luna_indices: set[int],
    matched_cluster_indices: set[int],
    allow_reuse_nearest_cluster: bool,
) -> tuple[list[ClusterInfo], list[float], set[int], set[int]]:
    candidates = []
    for luna_idx, (_, nodule) in enumerate(luna_nodules.iterrows()):
        if luna_idx in matched_luna_indices:
            continue
        center = np.asarray([nodule["coordX"], nodule["coordY"], nodule["coordZ"]], dtype=np.float64)
        for cluster_idx, cluster in enumerate(clusters):
            if cluster.center_world is None:
                continue
            if not allow_reuse_nearest_cluster and cluster_idx in matched_cluster_indices:
                continue
            distance = float(np.linalg.norm(cluster.center_world - center))
            candidates.append((distance, luna_idx, cluster_idx))

    recovered = []
    distances = []
    recovered_luna = set()
    recovered_clusters = set()
    for distance, luna_idx, cluster_idx in sorted(candidates):
        if luna_idx in recovered_luna:
            continue
        if not allow_reuse_nearest_cluster and cluster_idx in recovered_clusters:
            continue
        recovered_luna.add(luna_idx)
        recovered_clusters.add(cluster_idx)
        recovered.append(clusters[cluster_idx])
        distances.append(distance)

    return recovered, distances, recovered_luna, recovered_clusters


def unmatched_spatial_diagnostics(
    seriesuid: str,
    lidc_id: str,
    luna_nodules: pd.DataFrame,
    clusters: list[ClusterInfo],
    matched_luna_indices: set[int],
    min_match_radius_mm: float,
    match_radius_scale: float,
) -> list[dict]:
    rows = []
    for luna_idx, (_, nodule) in enumerate(luna_nodules.iterrows()):
        if luna_idx in matched_luna_indices:
            continue

        center = np.asarray([nodule["coordX"], nodule["coordY"], nodule["coordZ"]], dtype=np.float64)
        diameter = float(nodule["diameter_mm"])
        current_radius = max(diameter * 0.5 * match_radius_scale, min_match_radius_mm)

        nearest_cluster = None
        nearest_distance = np.nan
        for cluster in clusters:
            if cluster.center_world is None:
                continue
            distance = float(np.linalg.norm(cluster.center_world - center))
            if nearest_cluster is None or distance < nearest_distance:
                nearest_cluster = cluster
                nearest_distance = distance

        rows.append(
            {
                "seriesuid": seriesuid,
                "lidc_id": lidc_id,
                "luna_nodule_index": luna_idx,
                "coordX": float(nodule["coordX"]),
                "coordY": float(nodule["coordY"]),
                "coordZ": float(nodule["coordZ"]),
                "diameter_mm": diameter,
                "current_match_radius_mm": current_radius,
                "nearest_cluster_distance_mm": nearest_distance,
                "needed_min_match_radius_mm": nearest_distance,
                "needed_match_radius_scale": float(nearest_distance / (diameter * 0.5))
                if diameter > 0 and not np.isnan(nearest_distance)
                else np.nan,
                "nearest_cluster_mean_malignancy": nearest_cluster.mean_malignancy
                if nearest_cluster is not None
                else np.nan,
                "nearest_cluster_reader_annotations": nearest_cluster.num_reader_annotations
                if nearest_cluster is not None
                else 0,
            }
        )
    return rows


def build_luna_malignancy_csv(
    mapping_csv: Path,
    annotations_csv: Path,
    output_csv: Path,
    unmatched_nodules_csv: Path | None,
    match_source: str,
    xml_root: Path,
    luna_mhd_root: Path,
    min_match_radius_mm: float,
    match_radius_scale: float,
    xml_xy_tolerance_mm: float,
    xml_z_tolerance_mm: float,
    xml_diameter_tolerance_mm: float,
    recover_unmatched_nearest: bool,
    allow_reuse_nearest_cluster: bool,
    require_all_matches: bool,
) -> pd.DataFrame:
    luna = pd.read_csv(mapping_csv)
    annotations = pd.read_csv(annotations_csv)
    if "seriesuid" not in luna.columns:
        raise ValueError(f"{mapping_csv} must contain a seriesuid column.")
    if "seriesuid" not in annotations.columns:
        raise ValueError(f"{annotations_csv} must contain a seriesuid column.")

    annotations_by_series = {
        str(seriesuid): rows for seriesuid, rows in annotations.groupby(annotations["seriesuid"].astype(str))
    }

    scans_by_series = {}
    xml_clusters_by_series = {}
    cluster_cache = {}
    if match_source == "dicom":
        pl = import_pylidc()
        scans_by_series = {scan.series_instance_uid: scan for scan in pl.query(pl.Scan).all()}
    elif match_source == "xml":
        xml_clusters_by_series = load_xml_clusters_by_series(
            xml_root=xml_root,
            mapping=luna,
            xy_tolerance_mm=xml_xy_tolerance_mm,
            z_tolerance_mm=xml_z_tolerance_mm,
        )
    elif match_source == "pylidc_mhd":
        cluster_cache = load_pylidc_mhd_clusters_by_series(luna_mhd_root=luna_mhd_root)
    elif match_source == "mhd":
        cluster_cache = load_mhd_xml_clusters_by_series(
            xml_root=xml_root,
            luna_mhd_root=luna_mhd_root,
            mapping=luna,
            xy_tolerance_mm=xml_xy_tolerance_mm,
            z_tolerance_mm=xml_z_tolerance_mm,
        )
    else:
        raise ValueError(f"Unknown match_source: {match_source}")

    geometry_cache = {}
    dicom_error_cache = {}
    rows = []
    unmatched_diagnostic_rows = []
    for _, row in luna.iterrows():
        seriesuid = str(row["seriesuid"])
        lidc_id = row.get("lidc_id", "")
        luna_nodules = annotations_by_series.get(seriesuid, pd.DataFrame())
        nodule_count = int(len(luna_nodules))

        matched_clusters: list[ClusterInfo] = []
        match_distances: list[float] = []
        matched_luna_indices: set[int] = set()
        matched_cluster_indices: set[int] = set()
        recovered_distances: list[float] = []
        recovered_reused_cluster = False
        z_diffs: list[float] = []
        diameter_diffs: list[float] = []
        if nodule_count > 0:
            if match_source == "xml":
                matched_clusters, z_diffs, diameter_diffs = match_luna_nodules_to_xml_clusters(
                    luna_nodules=luna_nodules,
                    clusters=xml_clusters_by_series.get(seriesuid, []),
                    z_tolerance_mm=xml_z_tolerance_mm,
                    diameter_tolerance_mm=xml_diameter_tolerance_mm,
                )
            elif match_source in ("mhd", "pylidc_mhd"):
                matched_clusters, match_distances, matched_luna_indices, matched_cluster_indices = match_luna_nodules_to_clusters(
                    luna_nodules=luna_nodules,
                    clusters=cluster_cache.get(seriesuid, []),
                    min_match_radius_mm=min_match_radius_mm,
                    match_radius_scale=match_radius_scale,
                )
                if recover_unmatched_nearest:
                    recovered, recovered_distances, recovered_luna, recovered_clusters = (
                        recover_unmatched_luna_nodules_to_nearest_clusters(
                            luna_nodules=luna_nodules,
                            clusters=cluster_cache.get(seriesuid, []),
                            matched_luna_indices=matched_luna_indices,
                            matched_cluster_indices=matched_cluster_indices,
                            allow_reuse_nearest_cluster=allow_reuse_nearest_cluster,
                        )
                    )
                    recovered_reused_cluster = bool(
                        allow_reuse_nearest_cluster
                        and any(cluster_idx in matched_cluster_indices for cluster_idx in recovered_clusters)
                    )
                    matched_clusters.extend(recovered)
                    match_distances.extend(recovered_distances)
                    matched_luna_indices.update(recovered_luna)
                    matched_cluster_indices.update(recovered_clusters)
                unmatched_diagnostic_rows.extend(
                    unmatched_spatial_diagnostics(
                        seriesuid=seriesuid,
                        lidc_id=str(lidc_id),
                        luna_nodules=luna_nodules,
                        clusters=cluster_cache.get(seriesuid, []),
                        matched_luna_indices=matched_luna_indices,
                        min_match_radius_mm=min_match_radius_mm,
                        match_radius_scale=match_radius_scale,
                    )
                )
            else:
                scan = scans_by_series.get(seriesuid)
                if scan is None:
                    raise ValueError(f"No pylidc scan found for LUNA seriesuid {seriesuid}.")
                if seriesuid not in geometry_cache:
                    try:
                        geometry_cache[seriesuid] = scan_geometry(scan)
                    except Exception as exc:
                        geometry_cache[seriesuid] = None
                        dicom_error_cache[seriesuid] = str(exc)
                if seriesuid not in cluster_cache:
                    cluster_cache[seriesuid] = clusters_from_scan(scan, geometry_cache[seriesuid])
                if geometry_cache[seriesuid] is None:
                    matched_clusters = []
                else:
                    matched_clusters, match_distances, matched_luna_indices, matched_cluster_indices = match_luna_nodules_to_clusters(
                        luna_nodules=luna_nodules,
                        clusters=cluster_cache[seriesuid],
                        min_match_radius_mm=min_match_radius_mm,
                        match_radius_scale=match_radius_scale,
                    )
                    if recover_unmatched_nearest:
                        recovered, recovered_distances, recovered_luna, recovered_clusters = (
                            recover_unmatched_luna_nodules_to_nearest_clusters(
                                luna_nodules=luna_nodules,
                                clusters=cluster_cache[seriesuid],
                                matched_luna_indices=matched_luna_indices,
                                matched_cluster_indices=matched_cluster_indices,
                                allow_reuse_nearest_cluster=allow_reuse_nearest_cluster,
                            )
                        )
                        recovered_reused_cluster = bool(
                            allow_reuse_nearest_cluster
                            and any(cluster_idx in matched_cluster_indices for cluster_idx in recovered_clusters)
                        )
                        matched_clusters.extend(recovered)
                        match_distances.extend(recovered_distances)
                        matched_luna_indices.update(recovered_luna)
                        matched_cluster_indices.update(recovered_clusters)
                unmatched_diagnostic_rows.extend(
                    unmatched_spatial_diagnostics(
                        seriesuid=seriesuid,
                        lidc_id=str(lidc_id),
                        luna_nodules=luna_nodules,
                        clusters=cluster_cache.get(seriesuid, []),
                        matched_luna_indices=matched_luna_indices,
                        min_match_radius_mm=min_match_radius_mm,
                        match_radius_scale=match_radius_scale,
                    )
                )

        matched_count = len(matched_clusters)
        unmatched_count = max(nodule_count - matched_count, 0)
        if require_all_matches and unmatched_count:
            dicom_error = dicom_error_cache.get(seriesuid)
            if match_source == "xml":
                detail = " XML-only matching uses z-position and approximate diameter; inspect this case."
            elif match_source == "mhd":
                detail = (
                    f" Missing {seriesuid}.mhd under {luna_mhd_root}, or no XML contour cluster "
                    "fell within the 3D match radius."
                )
            elif match_source == "pylidc_mhd":
                detail = (
                    f" Missing {seriesuid}.mhd under {luna_mhd_root}, no pylidc scan, or no pylidc cluster "
                    "fell within the 3D match radius."
                )
            else:
                detail = f" DICOM geometry error: {dicom_error}" if dicom_error else (
                    " Increase --min-match-radius-mm only after inspecting this case."
                )
            raise ValueError(
                f"Matched {matched_count}/{nodule_count} LUNA nodules for {seriesuid}. "
                f"{detail}"
            )

        nodule_malignancy_means = [float(cluster.mean_malignancy) for cluster in matched_clusters]
        mean_between_nodules = float(np.mean(nodule_malignancy_means)) if nodule_malignancy_means else 0.0
        max_between_nodules = float(np.max(nodule_malignancy_means)) if nodule_malignancy_means else 0.0
        output_row = {
            "seriesuid": seriesuid,
            "lidc_id": lidc_id,
            "nodule_count": nodule_count,
            "nodule_annotation_mean_malignancies": json.dumps(nodule_malignancy_means),
            "max_nodule_mean_malignancy": max_between_nodules,
            "mean_nodule_malignancy": mean_between_nodules,
            "mean_between_nodule_mean_malignancies": mean_between_nodules,
            "num_luna_nodules_matched": matched_count,
            "num_luna_nodules_unmatched": unmatched_count,
            "num_luna_nodules_recovered_nearest": len(recovered_distances),
            "recovered_nearest_reused_cluster": recovered_reused_cluster,
            "num_reader_malignancy_annotations": int(
                sum(cluster.num_reader_annotations for cluster in matched_clusters)
            ),
            "max_match_distance_mm": float(max(match_distances)) if match_distances else 0.0,
            "max_recovered_match_distance_mm": float(max(recovered_distances)) if recovered_distances else 0.0,
            "max_xml_z_diff_mm": float(max(z_diffs)) if z_diffs else 0.0,
            "max_xml_diameter_diff_mm": float(max(diameter_diffs)) if diameter_diffs else 0.0,
            "match_method": "none"
            if nodule_count == 0
            else (
                "xml_z_diameter"
                if z_diffs
                else (
                    "mhd_spatial_nearest_reused"
                    if match_source == "mhd" and recovered_distances and recovered_reused_cluster
                    else "pylidc_mhd_spatial_nearest_reused"
                    if match_source == "pylidc_mhd" and recovered_distances and recovered_reused_cluster
                    else "pylidc_mhd_spatial_nearest_recovered"
                    if match_source == "pylidc_mhd" and recovered_distances
                    else "mhd_spatial_nearest_recovered"
                    if match_source == "mhd" and recovered_distances
                    else (
                        "pylidc_mhd_spatial"
                        if match_source == "pylidc_mhd" and match_distances
                        else
                        "mhd_spatial"
                        if match_source == "mhd" and match_distances
                        else (
                            "spatial_nearest_reused"
                            if recovered_distances and recovered_reused_cluster
                            else "spatial_nearest_recovered"
                            if recovered_distances
                            else ("spatial" if match_distances else "unmatched")
                        )
                    )
                )
            ),
            "dicom_error": dicom_error_cache.get(seriesuid, ""),
        }
        for col in ("target", "split"):
            if col in luna.columns:
                output_row[col] = row[col]
        rows.append(output_row)

    result = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    if unmatched_nodules_csv is not None:
        unmatched_nodules_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(unmatched_diagnostic_rows).to_csv(unmatched_nodules_csv, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write per-LUNA-sample mean malignancy by matching LUNA annotations to LIDC nodule annotations."
    )
    parser.add_argument(
        "--match-source",
        choices=("xml", "mhd", "pylidc_mhd", "dicom"),
        default="xml",
        help=(
            "Use XML-only z/diameter, XML+MHD 3D spatial, pylidc clusters+MHD 3D spatial, "
            "or DICOM-backed 3D spatial matching."
        ),
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=Path("data/LUNA16/luna16_all_seriesuid_to_lidc_id.csv"),
        help="CSV with one row per LUNA seriesuid and a nodule_count column.",
    )
    parser.add_argument(
        "--annotations-csv",
        type=Path,
        default=Path("data/LUNA16/annotations.csv"),
        help="Official LUNA16 annotations.csv.",
    )
    parser.add_argument(
        "--xml-root",
        type=Path,
        default=Path("data/LIDC-IDRI files/annotations/LIDC-XML-only/tcia-lidc-xml"),
        help="Root directory containing LIDC XML annotation files.",
    )
    parser.add_argument(
        "--luna-mhd-root",
        type=Path,
        default=Path("data/LUNA16"),
        help="Root directory containing original LUNA16 .mhd/.raw files, usually with subset* folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/LUNA16/luna16_mean_nodule_malignancy.csv"),
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--unmatched-nodules-csv",
        type=Path,
        default=None,
        help="Optional per-unmatched-LUNA-nodule diagnostics CSV for spatial matching modes.",
    )
    parser.add_argument(
        "--min-match-radius-mm",
        type=float,
        default=5.0,
        help="Minimum center-distance threshold for matching a LUNA nodule to a pylidc cluster.",
    )
    parser.add_argument(
        "--match-radius-scale",
        type=float,
        default=1.0,
        help="Also allow matches within diameter_mm / 2 * this scale.",
    )
    parser.add_argument(
        "--xml-xy-tolerance-mm",
        type=float,
        default=12.0,
        help="XML reader annotations within this in-plane distance may be clustered.",
    )
    parser.add_argument(
        "--xml-z-tolerance-mm",
        type=float,
        default=8.0,
        help="Maximum z distance for XML clustering and LUNA-to-XML matching.",
    )
    parser.add_argument(
        "--xml-diameter-tolerance-mm",
        type=float,
        default=10.0,
        help="Maximum diameter difference for LUNA-to-XML matching.",
    )
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Write the CSV even if some positive LUNA nodules cannot be matched.",
    )
    parser.add_argument(
        "--recover-unmatched-nearest",
        action="store_true",
        help="Assign unmatched LUNA nodules to nearest unused spatial cluster, even outside the match radius.",
    )
    parser.add_argument(
        "--allow-reuse-nearest-cluster",
        action="store_true",
        help="Allow nearest-cluster recovery to reuse an already assigned cluster for another unmatched LUNA nodule.",
    )
    args = parser.parse_args()

    result = build_luna_malignancy_csv(
        mapping_csv=args.mapping_csv,
        annotations_csv=args.annotations_csv,
        output_csv=args.output_csv,
        unmatched_nodules_csv=args.unmatched_nodules_csv,
        match_source=args.match_source,
        xml_root=args.xml_root,
        luna_mhd_root=args.luna_mhd_root,
        min_match_radius_mm=args.min_match_radius_mm,
        match_radius_scale=args.match_radius_scale,
        xml_xy_tolerance_mm=args.xml_xy_tolerance_mm,
        xml_z_tolerance_mm=args.xml_z_tolerance_mm,
        xml_diameter_tolerance_mm=args.xml_diameter_tolerance_mm,
        recover_unmatched_nearest=args.recover_unmatched_nearest,
        allow_reuse_nearest_cluster=args.allow_reuse_nearest_cluster,
        require_all_matches=not args.allow_unmatched,
    )
    positives = result[result["nodule_count"] > 0]
    print(f"wrote {args.output_csv}")
    print(f"samples: {len(result)}")
    print(f"samples with nodules: {len(positives)}")
    print(f"samples without nodules: {int((result['nodule_count'] == 0).sum())}")
    print(f"matched LUNA nodules: {int(result['num_luna_nodules_matched'].sum())}")
    print(f"unmatched LUNA nodules: {int(result['num_luna_nodules_unmatched'].sum())}")
    if "num_luna_nodules_recovered_nearest" in result.columns:
        print(f"nearest-recovered LUNA nodules: {int(result['num_luna_nodules_recovered_nearest'].sum())}")


if __name__ == "__main__":
    main()
