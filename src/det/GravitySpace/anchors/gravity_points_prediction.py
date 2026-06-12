import torch


def gravity_points_prediction(gravity_points: torch.Tensor,
                              hook: int,
                              regressions: torch.Tensor) -> torch.Tensor:
    """
    Compute gravity points prediction:

    :param gravity_points: gravity points configuration of A x 2
    :param hook: hook distance
    :param regressions: output regressions subnet stack of B x S x A x 2
    :return: predictions stack of B x S x A x 2
    """

    # coords gravity points: A x 2
    coord_gravity_points_x = gravity_points[:, 0]
    coord_gravity_points_y = gravity_points[:, 1]

    # delta regression subnet (de-normalized on hook dist in the Loss)
    delta_x = regressions[:, :, :, 0] * hook
    delta_y = regressions[:, :, :, 1] * hook

    # compute predictions
    predictions_x = coord_gravity_points_x + delta_x
    predictions_y = coord_gravity_points_y + delta_y

    # coords prediction
    predictions = torch.stack([predictions_x, predictions_y], dim=3)

    return predictions
