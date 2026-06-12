import sys
from typing import List

from src.det.GravitySpace.utility.msg.msg_error import msg_error
from src.det.GravitySpace.utility.read_file import read_file


def select_list(view: str,
                path_lists: dict) -> List[str]:
    """
    Select filename list according to view

    :param view: view CTA type [axial, coronal, sagittal]
    :param: path_lists: path dataset lists
    :return: filename list
    """

    # axial view
    if view == 'axial':
        filename_list = read_file(file_path=path_lists[view])

    # coronal view
    elif view == 'coronal':
        filename_list = read_file(file_path=path_lists[view])

    # sagittal
    elif view == 'sagittal':
        filename_list = read_file(file_path=path_lists[view])

    else:
        str_err = msg_error(file=__file__,
                            variable=view,
                            type_variable="view",
                            choices="[axial, coronal, sagittal]")
        sys.exit(str_err)

    return filename_list

