import sys
from typing import Tuple

import torch
import torch.nn as nn

# from src.det.GravitySpace.debug.debug_hooking import debug_hooking


class GravityLoss(nn.Module):
    """
    Gravity Loss
    """

    def __init__(self,
                 alpha: float,
                 gamma: float,
                 config: str,
                 hook: int,
                 hook_gap: int,
                 num_gravity_points_feature_map: int,
                 device: torch.device,
                 debug: bool):
        """
        __init__ method: run one when instantiating the object

        :param alpha: alpha parameter
        :param gamma: gamma parameters
        :param config: configuration
        :param hook: hook distance
        :param hook_gap: hook distance gap
        :param num_gravity_points_feature_map: num gravity points for feature map
        :param device: device
        :param debug: debug option
        """
        super(GravityLoss, self).__init__()

        # alpha parameter
        self.alpha = alpha

        # gamma parameter
        self.gamma = gamma

        # gravity points config
        self.config = config

        # hook distance (to assign positive gravity points)
        self.hook = hook

        # gap hook distance (to assign negative gravity points and rejected gravity points)
        self.gap = hook_gap

        # num gravity points in feature map (reference window)
        self.num_gravity_points_feature_map = num_gravity_points_feature_map

        # device (CPU or GPU)
        self.device = device

        # debug
        self.debug = debug

        # epsilon to prevent NaN in log operations
        self.eps = 1e-7

    def forward(self,
                images_batch: torch.Tensor,
                classifications_batch: torch.Tensor,
                regressions_batch: torch.Tensor,
                gravity_points: torch.Tensor,
                annotations_batch: torch.Tensor,
                slice_lengths: torch.Tensor = None,
                eval_slice_start: torch.Tensor = None,
                eval_slice_end: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward method: directly call a method in the class when an instance name is called

        :param images_batch: CTA batch
        :param classifications_batch: classifications score of the CTA batch
        :param regressions_batch: regressions of the CTA batch
        :param gravity_points: gravity points
        :param annotations_batch: annotations of the CTA batch
        :return: classification loss,
                 regression loss
        """

        # batch_size
        batch_size = classifications_batch.shape[0]

        # init losses list
        classification_losses = []
        regression_losses = []

        # num gravity points
        num_gravity_points = gravity_points.shape[0]

        # split the coord of each gravity point
        gravity_point_coord_x = gravity_points[:, 0]  # gravity points coord x (A)
        gravity_point_coord_y = gravity_points[:, 1]  # gravity points coord y (A)

        # for each batch
        for i in range(batch_size):

            # slices (for each image in batch)
            slices = images_batch[i, :, :, :, :]  # S x C x H x W

            valid_slices = slices.shape[0]
            if slice_lengths is not None:
                valid_slices = int(slice_lengths[i].item())
            first_eval_slice = int(eval_slice_start[i].item()) if eval_slice_start is not None else 0
            last_eval_slice = int(eval_slice_end[i].item()) if eval_slice_end is not None else valid_slices
            last_eval_slice = min(last_eval_slice, valid_slices)

            # annotations (for each image in batch)
            annotations = annotations_batch[i]  # S x MAX_NODULES x 4

            # classifications (for each image in batch)
            classifications = classifications_batch[i, :, :, :]  # S x A x 2
            classifications = torch.clamp(classifications, 1e-4, 1.0 - 1e-4)  # to avoid log(0) and Loss NaN

            # regression (for each image in batch)
            regressions = regressions_batch[i, :, :, :]  # S x A x 2

            # for each slice
            for j in range(first_eval_slice, last_eval_slice):

                slice = slices[j, :, :, :]  # C x H x W
                annotation = annotations[j]  # MAX_NODULES x 4

                classification = classifications[j, :, :]
                regression = regressions[j, :, :]

                annotation = annotation[annotation[:, 0] != -1]  # delete padding -1
                annotation = annotation[:, :2]
                num_annotations = annotation.shape[0]

                # CASE NO TARGET
                if num_annotations == 0:
                    alpha_factor = torch.ones(classification.shape).to(classification.device) * self.alpha

                    alpha_factor = 1. - alpha_factor  # - alpha
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)  # (1-p)^gamma

                    bce = -(torch.log(1.0 - classification + self.eps))  # log(p)

                    cls_loss = focal_weight * bce  # LOSS
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().to(classification.device))

                    continue

                # -------- #
                # DISTANCE #
                # -------- #
                # (CASE TARGET)
                # compute the euclidean distance between Gravity Points (Ax2) and Annotation (Tx2)
                # each column is the dist of each annotation respect to all gravity points
                dist = torch.cdist(gravity_points.float(), annotation.float(), p=2)  # num_gravity_points (A) x num_annotations (T)

                # dist
                #     T1    T2    T3    T4    T5
                # A1  .     .     .     .     .
                # A2  .     .     .     .     .
                # A3  .     .     .     .     .

                # ------------ #
                # MIN DISTANCE #
                # ------------ #
                # compute the min dist for each gravity points row (dist_min) to of all the gravity points closer to the annotation
                # and get the index (index_min) of that specific annotation
                # in other words: dist_min is the distance between annotation and closer gravity point
                #                 index_min is the index of the annotation
                dist_min, index_min = torch.min(dist, dim=1)  # A x 1

                # init labels
                labels = torch.ones(classification.shape).to(classification.device) * -1  # A x 2

                # POSITIVE INDICES: gravity points with dist min <= hook dist
                positive_indices = torch.le(dist_min, self.hook)

                # NEGATIVE INDICES: gravity points with dist min > hook dist + hook gap
                negative_indices = torch.gt(dist_min, self.hook + self.gap)

                # REJECTED INDICES: gravity points with dist min < hook dist and > hook gap
                rejected_indices = torch.logical_and(torch.gt(dist_min, self.hook), torch.le(dist_min, self.hook + self.gap))

                # num positive gravity points
                num_positive_gravity_points = positive_indices.sum()

                # num negative gravity points
                num_negative_gravity_points = negative_indices.sum()

                # num rejected gravity points
                num_rejected_gravity_points = rejected_indices.sum()

                # annotation closest to each gravity points (with index min)
                assigned_annotations = annotation[index_min]

                # marks the positive indices with '1'
                labels[positive_indices, :] = 1

                # marks the negative indices with '0'
                labels[negative_indices, :] = 0

                # marks the rejected indices with '-1'
                labels[rejected_indices, :] = -1

                # ----- #
                # DEBUG #
                # ----- #
                if self.debug:
                    print("\nDEBUG HOOKING"
                          "\nConfig: {}".format(self.config),
                          "\nGravity Points: {}".format(num_gravity_points),
                          "\nHook: {}".format(self.hook),
                          "\n",
                          "\nPositive Gravity Points: {} / {}".format(num_positive_gravity_points, num_gravity_points),
                          "\nNegative Gravity Points: {} / {}".format(num_negative_gravity_points, num_gravity_points),
                          "\nRejected Gravity Points: {} / {}".format(num_rejected_gravity_points, num_gravity_points),
                          "\n",
                          "\nAnnotations: {}".format(num_annotations),
                          "\nHooked Annotations: {} / {}".format(len(torch.unique(assigned_annotations[positive_indices, 0])), num_annotations))

                    # debug hooking
                    # debug_hooking(gravity_points=gravity_points,
                    #               annotation=annotation,
                    #               assigned_annotations=assigned_annotations,
                    #               positive_indices=positive_indices,
                    #               negative_indices=negative_indices,
                    #               rejected_indices=rejected_indices,
                    #               image=slice,
                    #               save=True,
                    #               path="./debug/GravityPoints-Hooking|config={}|gravity-points={}|hook={}|gap={}.png".format(self.config,
                    #                                                                                                          num_gravity_points,
                    #                                                                                                          self.hook,
                    #                                                                                                          self.gap))

                    sys.exit('\nDEBUG HOOKING: COMPLETE')

                # ------------------- #
                # CLASSIFICATION LOSS #
                # ------------------- #
                # alpha factor
                alpha_factor = torch.ones(labels.shape).to(classification.device) * self.alpha
                alpha_factor = torch.where(torch.eq(labels, 1.), alpha_factor, 1. - alpha_factor)

                # focal weight
                focal_weight = torch.where(torch.eq(labels, 1.), 1. - classification, classification)
                focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

                # bce
                bce = -(labels * torch.log(classification + self.eps) + (1.0 - labels) * torch.log(1.0 - classification + self.eps))

                # classification loss
                cls_loss = focal_weight * bce
                cls_loss = torch.where(torch.ne(labels, -1.0), cls_loss, torch.zeros(cls_loss.shape).to(classification.device))

                classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_gravity_points.float(), min=1.0))

                # --------------- #
                # REGRESSION LOSS #
                # --------------- #
                if positive_indices.sum() > 0:

                    # annotation of positive indices
                    assigned_annotations = assigned_annotations[positive_indices, :]
                    annotation_coord_x = assigned_annotations[:, 0]  # annotation coord x
                    annotation_coord_y = assigned_annotations[:, 1]  # annotation coord y

                    # regression of positive indices
                    regression = regression[positive_indices, :]

                    # gravity points of positive indices
                    gravity_point_coord_x_pi = gravity_point_coord_x[positive_indices]  # gravity points (positive) coord x
                    gravity_point_coord_y_pi = gravity_point_coord_y[positive_indices]  # gravity points (positive) coord y

                    # delta
                    annotations_delta_x = (annotation_coord_x - gravity_point_coord_x_pi)  # annotation coord x - gravity point coord pi x
                    annotations_delta_y = (annotation_coord_y - gravity_point_coord_y_pi)  # annotation coord y - gravity point coord pi y

                    annotations_delta = torch.stack((annotations_delta_x, annotations_delta_y))
                    annotations_delta = annotations_delta.t()

                    # normalization (to hook dist)
                    annotations_delta = annotations_delta / torch.Tensor([[self.hook, self.hook]]).to(classification.device)

                    # regression diff
                    regression_diff = torch.abs(annotations_delta - regression)

                    # -------------- #
                    # SMOOTH L1 LOSS #
                    # -------------- #
                    # condition: regression_diff <= 1.0 / num_gravity_points_feature_map
                    # true: 0.5 * num_gravity_points_feature_map * regression_diff^2
                    # false: regression_diff - 0.5 / num_gravity_points_feature_map
                    threshold = 1.0 / (self.num_gravity_points_feature_map + self.eps)
                    regression_loss = torch.where(
                        torch.le(regression_diff, threshold),
                        0.5 * self.num_gravity_points_feature_map * torch.pow(regression_diff + self.eps, 2),
                        regression_diff - 0.5 / (self.num_gravity_points_feature_map + self.eps)
                    )

                    regression_losses.append(regression_loss.mean())

                else:
                    regression_losses.append(torch.tensor(0).float().to(classification.device))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)
