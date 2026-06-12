import sys

from src.det.GravitySpace.utility.msg.msg_error import msg_error


def select_TotalNumOfImages(FP_images: str,
                            num_images: int,
                            num_normal_images: int) -> int:
    """
    Select TotalNumOfImages for FROC computation based on where False Positive (FP) where calculated

    :param FP_images: images where calculate FP
    :param num_images: num images
    :param num_normal_images: num normal images
    :return: TotalNumOfImages based on where calculate FP
    """

    # split with images no-healthy
    if FP_images == 'all':
        TotalNumOfImages = num_images

    elif FP_images == 'normals':
        TotalNumOfImages = num_normal_images

    else:
        str_err = msg_error(file=__file__,
                            variable=FP_images,
                            type_variable="FP images",
                            choices="[all, normals]")
        sys.exit(str_err)

    return TotalNumOfImages
