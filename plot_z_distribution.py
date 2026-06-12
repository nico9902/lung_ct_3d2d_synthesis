import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    preprocessed_dir = "data/preprocessed_z_only"
    z_sizes = []
    patient_ids = []

    # Iterate through all patient directories
    for patient_id in sorted(os.listdir(preprocessed_dir)):
        patient_path = os.path.join(preprocessed_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue
            
        metadata_path = os.path.join(patient_path, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                    # resampled_size is [X, Y, Z]
                    z_size = meta['slices']['kept']
                    z_sizes.append(z_size)
                    patient_ids.append(patient_id)
            except Exception as e:
                print(f"Error reading {metadata_path}: {e}")

    if not z_sizes:
        print("No metadata found.")
        return

    plt.figure(figsize=(12, 6))

    # Plot 1: Histogram/Distribution
    plt.subplot(1, 2, 1)
    plt.hist(z_sizes, color="skyblue", bins=max(5, len(z_sizes)//2), edgecolor='black')
    plt.title("Distribution of Resampled Z-Size (Number of Slices)", fontsize=14)
    plt.xlabel("Number of Slices (Z)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 2: Scatter plot (Patient vs Z-size) to see variability
    plt.subplot(1, 2, 2)
    indices = np.arange(len(z_sizes))
    plt.scatter(indices, z_sizes, alpha=0.6, color="coral", edgecolors='black', s=100)
    plt.title("Resampled Z-Size per Patient", fontsize=14)
    plt.xlabel("Patient Index", fontsize=12)
    plt.ylabel("Number of Slices (Z)", fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    output_plot = "data/z_distribution_plot.png"
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    print(f"Plot saved to {output_plot}")

if __name__ == "__main__":
    main()
