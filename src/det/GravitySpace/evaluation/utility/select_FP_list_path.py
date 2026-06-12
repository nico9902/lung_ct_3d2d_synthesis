import sys

from src.det.GravitySpace.utility.msg.msg_error import msg_error


def select_FP_list_path(dataset_name: str,
                        FP_images: str,
                        view: str,
                        path: dict) -> str:
    """
    Select list of images which calculate False Positive (FP)

    :param dataset_name: dataset name
    :param FP_images: images where calculate FP
    :param view: view CTA type
    :param path: path dictionary
    :return: path images list
    """

    # TruetaHospital
    if dataset_name == 'TruetaHospital':

        # FP calculated on all images
        if FP_images == 'all':
            if view in ['axial', 'coronal', 'sagittal']:
                FP_list_path = path[dataset_name]['lists']['all'][view]
            elif view in ['axial-cropped-vessel', 'coronal-cropped-vessel', 'sagittal-cropped-vessel']:
                FP_list_path = path[dataset_name]['lists']['all_cropped_vessel'][view.split('-')[0]]
            elif view in ['axial-cropped-volume', 'coronal-cropped-volume', 'sagittal-cropped-volume']:
                FP_list_path = path[dataset_name]['lists']['all_cropped_volume'][view.split('-')[0]]
            else:
                sys.exit("WRONG VIEW in select_FP_list_path.py")

        # FP calculated only on images normals (with no lesions)
        elif FP_images == 'normals':
            if view in ['axial', 'coronal', 'sagittal']:
                FP_list_path = path[dataset_name]['lists']['normals'][view]
            elif view in ['axial-cropped-vessel', 'coronal-cropped-vessel', 'sagittal-cropped-vessel']:
                FP_list_path = path[dataset_name]['lists']['normals_cropped_vessel'][view.split('-')[0]]
            elif view in ['axial-cropped-volume', 'coronal-cropped-volume', 'sagittal-cropped-volume']:
                FP_list_path = path[dataset_name]['lists']['normals_cropped_volume'][view.split('-')[0]]
            else:
                sys.exit("WRONG VIEW in select_FP_list_path.py")

        else:
            str_err = msg_error(file=__file__,
                                variable=FP_images,
                                type_variable="FP images",
                                choices="[all, normals]")
            sys.exit(str_err)

    else:
        str_err = msg_error(file=__file__,
                            variable=dataset_name,
                            type_variable="dataset name",
                            choices="[TruetaHospital]")
        sys.exit(str_err)

    return FP_list_path
