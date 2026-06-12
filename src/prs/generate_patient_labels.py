import numpy as np
# Monkey-patch older numpy types for pylidc compatibility
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float

import configparser
# Monkey-patch for Python 3.12+
if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser

import pylidc as pl
import pandas as pd
import sys

def generate_patient_labels():
    print("Querying all scans from pylidc...")
    scans = pl.query(pl.Scan).all()
    print(f"Found {len(scans)} scans.")

    patient_data = {}

    for i, scan in enumerate(scans):
        pid = scan.patient_id
        if i % 10 == 0:
            print(f"Processing scan {i+1}/{len(scans)} (Patient: {pid})...")

        if pid not in patient_data:
            patient_data[pid] = {
                "nodules": [], # List of max malignancies per nodule (or just all malignancies?)
                               # The logic requires checking if ANY nodule is >= 4.
                               # But also "nodule counts". Only clusters count as nodules.
                "all_malignancies": [],
                "nodule_means": [] # List of mean malignancy per nodule
            }

        # Cluster annotations to identify distinct nodules
        # verbose=False suppresses progress bars
        try:
            nods = scan.cluster_annotations(verbose=False)
        except Exception as e:
            print(f"Error clustering annotations for patient {pid}: {e}")
            continue
        
        # nods is a list of clusters. Each cluster is a list of annotations.
        
        # We need to count distinct nodules
        # Note: If a patient has multiple scans, we sum the nodules found in each scan.
        # This assumes scans don't duplicate the same nodules (or if they do, we count them as observed instances).
        # Given LIDC structure, this is the standard approach unless doing spatial registration.
        
        for nod_cluster in nods:
            # For this nodule, collect all malignancy scores from annotations
            nodule_malignancies = [ann.malignancy for ann in nod_cluster]
            
            if nodule_malignancies:
                # We store the list of malignancies for this nodule
                patient_data[pid]["nodules"].append(nodule_malignancies)
                patient_data[pid]["all_malignancies"].extend(nodule_malignancies)
                # Store the mean malignancy for this nodule
                patient_data[pid]["nodule_means"].append(np.mean(nodule_malignancies))

    print("Processing completed. Generating labels...")

    results = []
    for pid, data in patient_data.items():
        nodule_count = len(data["nodules"])
        all_mals = data["all_malignancies"]
        
        # Label logic:
        # 1. If at least one nodule has malignancy >= 4 -> False
        # 2. If always < 3 -> True
        # 3. If there is a 3 (and no >= 4) -> Indeterminate

        # Note: "at least one nodule with malignancy >= 4"
        # Since we collected all malignancy scores, we can check the max of all scores.
        # If any single annotation is >= 4, does that trigger it?
        # User said "malignancy score of the nodules".
        # Usually checking the max of all readings is the safest interpretation of "if ... has ... >= 4" 
        # given the xml2patient.py reference.
        
        if not all_mals:
            # No nodules/annotations? 
            max_mal_orig = 0
            max_mal_new = 0
        else:
            max_mal_orig = max(all_mals)
            max_mal_new = max(data["nodule_means"]) if data["nodule_means"] else 0
            
        # Logic 1: Original (Max of all annotations)
        if max_mal_orig >= 4:
            label_orig = "True"
        elif max_mal_orig < 3:
            label_orig = "False"
        else:
            label_orig = "Indeterminate"

        # Logic 2: New (Max of nodule means)
        if max_mal_new > 3:
            label_new = "True"
        elif max_mal_new < 3:
            label_new = "False"
        else:
            label_new = "Indeterminate"
                    
        results.append({
            "patient_id": pid,
            "nodule_count": nodule_count,
            "label": label_orig,
            "label_nodule_mean": label_new
        })
        
    df = pd.DataFrame(results)
    output_file = "patient_level_labels.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved results for {len(df)} patients to {output_file}")
    # Show distribution
    # Show distribution
    print("\nLabel Distribution (Original - Max of Annotations):")
    print(df["label"].value_counts())
    print("\nLabel Distribution (New - Max of Nodule Means):")
    print(df["label_nodule_mean"].value_counts())

if __name__ == "__main__":
    generate_patient_labels()
