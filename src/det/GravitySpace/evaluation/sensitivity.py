from typing import Tuple

import numpy as np


def sensitivity(FPS: np.ndarray,
                sens: np.ndarray,
                work_point: int) -> Tuple[float, float]:
    """
    Compute sensitivity at work point and sensitivity max

    :param FPS: false positive per scan (FPS)
    :param sens: sensitivity
    :param work_point: work point (FPS)
    :return: sensitivity work point,
             sensitivity max
    """

    x1 = x2 = y1 = y2 = 0
    for n in range(len(FPS)):
        if FPS[n] >= work_point:
            x1 = FPS[n - 1]
            x2 = FPS[n]
            y1 = sens[n - 1]
            y2 = sens[n]
            break

    # sens work point (TPR wp)
    try:
        sens_work_point = y1 + (work_point - x1) * (y2 - y1) / (x2 - x1)
    except ZeroDivisionError:
        sens_work_point = np.nan
        print("\nWARNING: ZeroDivisionError at sensitivity.py with work point {}".format(work_point))

    # sens max (TPR max)
    sens_max = sens[-1]

    return sens_work_point, sens_max
