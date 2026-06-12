from pandas import read_csv


def read_split_case(path_split_case: str) -> dict:
    """
    Read data split per case

    :param path_split_case: path split case
    :return: split case dictionary
    """

    # read csv
    dtype_mapping = {"CASE": str}  # define data type for cols
    data_split_case = read_csv(filepath_or_buffer=path_split_case, usecols=["INDEX", "CASE", "SPLIT"], dtype=dtype_mapping).values

    index = data_split_case[:, 0]
    case = data_split_case[:, 1]
    split = data_split_case[:, 2]

    split_case_dict = {
        'index': index.tolist(),
        'case': case.tolist(),
        'split': split.tolist()
    }

    return split_case_dict
