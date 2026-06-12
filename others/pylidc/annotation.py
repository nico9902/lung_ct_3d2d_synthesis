import numpy as np
np.int = int
np.float = float
import configparser
# Monkey-patch for Python 3.12+
if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser
import pylidc as pl
import matplotlib.pyplot as plt

patient_ids = (
    pl.query(pl.Annotation.scan_id)
    .distinct()
    .all()
)

scan_id = patient_ids[13][0]

scan = pl.query(pl.Scan).get(scan_id)
print("patient_id: ", scan.patient_id)
# => LIDC-IDRI-0001

anns = pl.query(pl.Annotation).filter(pl.Annotation.scan_id == scan.id)
print("anns count: ", anns.count())

nodules = scan.cluster_annotations()
print("nodules count: ", len(nodules))

nodule = nodules[0]

print(len(nodule))

ann = anns.first()
contours = ann.contours
print("contours: ", contours)

print("diameter: %.2f mm, surface_area: %.2f mm^2, volume: %.2f mm^3" % (ann.diameter,
                                         ann.surface_area,
                                         ann.volume))

mask = ann.boolean_mask()
print("mask shape: ", mask.shape, mask.dtype)

bbox = ann.bbox()
print("bbox: ", bbox)

vol = ann.scan.to_volume()
print("vol[bbox].shape: ", vol[bbox].shape, vol[bbox].dtype)

print("bbox_dims: ", ann.bbox_dims())


# visualization
vol = ann.scan.to_volume()

padding = [(30,10), (10,25), (0,0)]

mask = ann.boolean_mask(pad=padding)
bbox = ann.bbox(pad=padding)

fig,ax = plt.subplots(1,2,figsize=(5,3))

ax[0].imshow(vol[bbox][:,:,2], cmap=plt.cm.gray)
ax[0].axis('off')

ax[1].imshow(mask[:,:,2], cmap=plt.cm.gray)
ax[1].axis('off')

plt.tight_layout()
#plt.savefig("../images/mask_bbox.png", bbox_inches="tight")
plt.show()


# visualization in scan
ann.visualize_in_scan()