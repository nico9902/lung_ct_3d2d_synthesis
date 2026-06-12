from typing import Union, List

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts

from src.det.GravitySpace.utility.msg.msg_save_model_complete import msg_save_best_model_complete
from src.det.GravitySpace.utility.msg.msg_save_model_complete import msg_save_resume_model_complete


def save_best_model(epoch: int,
                    net: torch.nn.Module,
                    metrics: List[float],
                    metrics_type: str,
                    optimizer: Union[Adam, SGD],
                    scheduler: Union[ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts],
                    path: str):
    """
    Save best model

    :param epoch: num epoch
    :param net: net
    :param metrics: metrics
    :param metrics_type: metrics type
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param path: path to save model
    """

    # save model
    torch.save({
        'epoch': epoch,
        'net_state_dict': net.state_dict(),
        metrics_type: max(metrics),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng_state': torch.get_rng_state()
    }, path)

    # msg save best-model
    msg_save_best_model_complete(metrics_type=metrics_type)


def save_resume_model(epoch: int,
                      net: torch.nn.Module,
                      sensitivity_1_FPS: float,
                      sensitivity_10_FPS: float,
                      AUFROC_0_1: float,
                      AUFROC_0_10: float,
                      optimizer: Union[Adam, SGD],
                      scheduler: Union[ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts],
                      path: str):
    """
    Save resume model

    :param epoch: num epoch
    :param net: net
    :param sensitivity_1_FPS: sensitivity at 1 FPS
    :param sensitivity_10_FPS: sensitivity at 10 FPS
    :param AUFROC_0_1: AUFROC [0, 1]
    :param AUFROC_0_10: AUFROC [0, 10]
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param path: path to save resume model
    """

    # save model
    torch.save({
        'epoch': epoch,
        'net_state_dict': net.state_dict(),
        'sensitivity 1 FPS': sensitivity_1_FPS,
        'sensitivity 10 FPS': sensitivity_10_FPS,
        'AUFROC [0, 1]': AUFROC_0_1,
        'AUFROC [0, 10]': AUFROC_0_10,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng_state': torch.get_rng_state()
    }, path)

    # msg save resume-model
    msg_save_resume_model_complete(epoch=epoch)
