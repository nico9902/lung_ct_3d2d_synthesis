# LUNA16 Annotation Selection

This note summarizes how the LUNA16 challenge selected its reference nodules
from LIDC-IDRI, based on the LUNA16 paper:

Setio et al., "Validation, comparison, and combination of algorithms for
automatic detection of pulmonary nodules in computed tomography images: The
LUNA16 challenge", arXiv:1612.08012v4.

## Dataset Selection

LIDC-IDRI contains 1018 CT scans. LUNA16 did not use all of them.

The LUNA16 authors filtered the scans as follows:

1. Start from the LIDC-IDRI CT scans.
2. Keep only thin-slice CT scans suitable for pulmonary nodule management.
3. Exclude scans with slice thickness greater than 3 mm.
4. Exclude scans with inconsistent slice spacing.
5. Exclude scans with missing slices.

After this filtering, LUNA16 used 888 CT scans.

This is why a local LIDC-IDRI installation can contain more scans than LUNA16.
For example, having about 1010 local LIDC-IDRI cases is not the same as using
the official LUNA16 subset.

## Original LIDC-IDRI Annotation Categories

Each LIDC-IDRI scan was reviewed by four experienced thoracic radiologists in a
two-phase process:

1. A blinded phase, where radiologists independently marked lesions.
2. An unblinded phase, where each radiologist reviewed the anonymized marks from
   the other radiologists.

The radiologists classified findings into three categories:

- `nodule >= 3 mm`
- `nodule < 3 mm`
- `non-nodule`

Only findings marked as `nodule >= 3 mm` were used as relevant nodule
candidates for the LUNA16 reference standard.

Nodules smaller than 3 mm and non-nodule findings were not used as target
positive nodules in LUNA16.

## Merging Reader Annotations

The same physical nodule could be annotated by more than one radiologist.
LUNA16 merged annotations from different readers when they referred to the same
lesion.

The merging rule described in the paper is:

```text
merge annotations when the distance between their centers is smaller than the
sum of their radii
```

When annotations were merged:

- the nodule position was averaged;
- the nodule diameter was averaged.

So the `diameter_mm` in the official LUNA16 annotation file is an averaged
diameter for merged reader annotations.

## Majority-Agreement Positive Nodules

After merging annotations, LUNA16 counted how many radiologists annotated each
nodule.

The paper reports:

| Reader agreement | Number of nodules |
| --- | ---: |
| At least 1 of 4 radiologists | 2290 |
| At least 2 of 4 radiologists | 1602 |
| At least 3 of 4 radiologists | 1186 |
| 4 of 4 radiologists | 777 |

LUNA16 selected the nodules annotated by at least 3 of 4 radiologists as the
positive reference standard.

Therefore, the official LUNA16 positive nodules are:

```text
nodule >= 3 mm
AND annotated by at least 3 of 4 radiologists
```

This produces 1186 positive nodules.

## Irrelevant Findings

LUNA16 did not simply treat every non-reference finding as a false positive.

The following findings were considered "irrelevant findings":

- nodules annotated by fewer than 3 of 4 radiologists;
- `nodule < 3 mm` annotations;
- `non-nodule` annotations.

During LUNA16 evaluation, CAD marks on irrelevant findings were ignored. They
were not counted as true positives and were not counted as false positives.

This is important: LUNA16 positive annotations are strict, but the evaluation
also avoids penalizing detections on ambiguous or clinically non-target
findings.

## Diameter and Radius

The official LUNA16 annotation CSV uses:

```text
seriesuid,coordX,coordY,coordZ,diameter_mm
```

The coordinates are physical world coordinates in millimeters.

The diameter is the averaged diameter after merging reader annotations for the
same nodule.

For LUNA-style matching, the nodule radius is:

```text
radius_mm = diameter_mm / 2
```

A detection is considered a true positive when its center falls within the
radius of a reference nodule, and that nodule has not already been matched by a
higher-confidence detection.

## Difference From This Repository's LIDC Preprocessing

The current SCPMNet LIDC preprocessing does not implement the LUNA16 annotation
selection rule by default.

In `src/det/SCPMNet/lidc_preprocessing.py`, nodules are generated from pylidc
clusters:

```python
for nodule_cluster in scan.cluster_annotations():
    consensus = pylidc.utils.consensus(nodule_cluster, clevel=consensus_threshold)
```

There is no default check like:

```python
if len(nodule_cluster) < 3:
    continue
```

Therefore, a nodule cluster annotated by only one radiologist can be included in
the generated LIDC labels, as long as it produces a non-empty consensus mask.

The generated LIDC labels are then converted from connected mask components into
box-like rows:

```text
seriesuid,x,y,z,w,h,d,label
```

For SCPMNet, these boxes are converted to spheres using:

```text
radius = max(w, h, d) / 2
```

This differs from official LUNA16, where:

```text
radius = diameter_mm / 2
```

and `diameter_mm` comes from averaged radiologist diameter measurements.

## To Approximate LUNA16 Selection In LIDC Preprocessing

To make the local LIDC preprocessing closer to LUNA16, the preprocessing should
filter nodule clusters before generating masks:

```python
for nodule_cluster in scan.cluster_annotations():
    if len(nodule_cluster) < 3:
        continue
    ...
```

A stricter reproduction should also ensure the nodule is at least 3 mm. The
exact measurement should be chosen deliberately:

- use pylidc/radiologist diameter metadata if available;
- or compute a mask-derived diameter consistently;
- or use an equivalent-sphere diameter from the consensus mask.

The closest LUNA16-compatible target is:

```text
include only nodules >= 3 mm with annotations from at least 3 radiologists
```

For exact LUNA16 experiments, prefer the official LUNA16 subset and official
`annotations.csv`.
