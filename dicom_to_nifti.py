import os
import argparse
import dicom2nifti
import logging
import nibabel as nib
import numpy as np

def convert_npy_to_nifti(input_npy, output_file, compression=True):
    """
    Converts a .npy file into a NIfTI file.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # If output_file doesn't have correct extension, add it
    ext = ".nii.gz" if compression else ".nii"
    if not output_file.lower().endswith(ext):
        output_file += ext

    print(f"Loading NPY from: {input_npy}")
    try:
        data = np.load(input_npy)
        # Assuming input shape is (D, H, W), convert to (H, W, D) for NIfTI
        if data.ndim == 3:
            data = np.transpose(data, (1, 2, 0))
        
        # Use identity affine if not provided
        ni_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(ni_img, output_file)
        print(f"Saving NIfTI to: {output_file}")
        print("Conversion successful!")
    except Exception as e:
        print(f"Error during NPY conversion: {e}")

def convert_dicom_to_nifti(input_dir, output_file, compression=True):
    """
    Converts a directory of DICOM files into a single NIfTI file.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # If output_file doesn't have correct extension, add it
    ext = ".nii.gz" if compression else ".nii"
    if not output_file.lower().endswith(ext):
        output_file += ext

    print(f"Converting DICOMs from: {input_dir}")
    print(f"Saving NIfTI to: {output_file}")

    try:
        # dicom2nifti.dicom_series_to_nifti returns a dictionary with information 
        # about the conversion.
        dicom2nifti.dicom_series_to_nifti(input_dir, output_file, reorient_nifti=True)
        print("Conversion successful!")
    except Exception as e:
        print(f"Error during conversion: {e}")

def batch_convert(root_dir, output_root, compression=True, is_npy=False):
    """
    Search for DICOM series or NPY volumes in subdirectories and convert each to NIfTI.
    """
    patients = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for patient_id in patients:
        patient_dir = os.path.join(root_dir, patient_id)
        patient_output_dir = os.path.join(output_root, patient_id)
        
        if is_npy:
            # Handle volume
            volume_npy = os.path.join(patient_dir, "volume.npy")
            if os.path.exists(volume_npy):
                output_file = os.path.join(patient_output_dir, f"{patient_id}_volume")
                convert_npy_to_nifti(volume_npy, output_file, compression)
            
            # Handle nodule mask
            nodule_mask_npy = os.path.join(patient_dir, "nodule_mask.npy")
            if os.path.exists(nodule_mask_npy):
                output_mask_file = os.path.join(patient_output_dir, f"{patient_id}_nodule_mask")
                convert_npy_to_nifti(nodule_mask_npy, output_mask_file, compression)
            
            # # Handle lung mask
            # lung_mask_npy = os.path.join(patient_dir, "lung_mask.npy")
            # if os.path.exists(lung_mask_npy):
            #     output_lung_file = os.path.join(patient_output_dir, f"{patient_id}_lung_mask")
            #     convert_npy_to_nifti(lung_mask_npy, output_lung_file, compression)
        else:
            output_file = os.path.join(patient_output_dir, f"{patient_id}")
            convert_dicom_to_nifti(patient_dir, output_file, compression)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DICOM series to NIfTI format.")
    parser.add_argument("input", help="Directory containing DICOM files or root directory for batch conversion.")
    parser.add_argument("output", help="Output NIfTI file path or output directory for batch conversion.")
    parser.add_argument("--batch", action="store_true", help="Perform batch conversion on subdirectories of input.")
    parser.add_argument("--npy", action="store_true", help="Input is a directory of .npy preprocessed files (volume.npy/nodule_mask.npy).")
    parser.add_argument("--no-compression", action="store_false", dest="compression", help="Save as .nii instead of .nii.gz")
    parser.set_defaults(compression=True)

    args = parser.parse_args()

    # Disable dicom2nifti internal logging if it's too noisy
    logging.getLogger('dicom2nifti').setLevel(logging.ERROR)

    if args.batch:
        batch_convert(args.input, args.output, args.compression, args.npy)
    else:
        if args.npy:
            convert_npy_to_nifti(args.input, args.output, args.compression)
        else:
            convert_dicom_to_nifti(args.input, args.output, args.compression)
