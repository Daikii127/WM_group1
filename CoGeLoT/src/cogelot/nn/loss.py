import torch
from einops import rearrange


def compute_fine_grained_loss_with_loops(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    *,
    ignore_target_index: int = -100,
) -> torch.Tensor:
    """Compute a fine-grained loss across all the poses and the axes per pose.

    Since a trajectory can be made of more than one action, we need to average the loss across the
    trajectory _and_ the batch.

    Additionally, the loss for each action is constructed from predictions across each axis for
    each head (e.g. the rotation pose has 4 heads: one for each of X, Y, Z, W), meaning there are
    multiple separate losses to compute and compare.

    On top of calculating the loss, we need a way to ensure the mask from the targets are kept as
    we will need these when reducing the loss. If we don't, the loss will be incorrect. To get
    around this, any masked values are replaces with NaN's. This is because torch's reduction
    operators support NaN's, so we can just use those to reduce the loss. (See
    `reduce_fine_grained_loss()` for how that's done.)

    Note: Since this is using for-loops, it is much slower than using vectorised operations. Since
    we have that (see `compute_fine_grained_loss`), we are using that in the code. However, this
    exists for debugging and testing purposes, so we can be sure that it is correct.
    """
    predicted_logits_per_axis = predicted_actions.unbind(0)
    targets_per_axis = target_actions.unbind(0)

    loss_per_action_per_axis: list[torch.Tensor] = []

    for predicted_logits, targets in zip(predicted_logits_per_axis, targets_per_axis, strict=True):
        # When calculating the loss itself, we need to reshape we to be 2-dimensional (not sure why
        # but it's needed). Afterwards, we can reshape it to the target shape, as we are now in the
        # shape of (batch, num_timesteps, 1). This seems excessive but it's the only way I know how
        # right now because torch docs say that providing class indices to the targets is more
        # performative, and performance is good.
        loss = torch.nn.functional.cross_entropy(
            predicted_logits.reshape(-1, predicted_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=ignore_target_index,
            reduction="none",
        ).reshape(targets.shape)

        # To make sure we mask out the ignored target indices when reducing the loss, we turn every
        # masked loss into a nan. This is because torch's reduction operators support nan's.
        loss[targets == ignore_target_index] = float("nan")
        loss_per_action_per_axis.append(loss)

    return torch.stack(loss_per_action_per_axis, dim=-1)


# def compute_fine_grained_loss(
#     predicted_actions_logits_per_axis: torch.Tensor,
#     target_actions_per_axis: torch.Tensor,
#     *,
#     ignore_target_index: int = -100,
# ) -> torch.Tensor:
#     """Compute a fine-grained loss across all the poses and the axes per pose.

#     Since a trajectory can be made of more than one action, we need to average the loss across the
#     trajectory _and_ the batch.

#     Additionally, the loss for each action is constructed from predictions across each axis for
#     each head (e.g. the rotation pose has 4 heads: one for each of X, Y, Z, W), meaning there are
#     multiple separate losses to compute and compare.

#     On top of calculating the loss, we need a way to ensure the mask from the targets are kept as
#     we will need these when reducing the loss. If we don't, the loss will be incorrect. To get
#     around this, any masked values are replaces with NaN's. This is because torch's reduction
#     operators support NaN's, so we can just use those to reduce the loss. (See
#     `reduce_fine_grained_loss()` for how that's done.)
#     """
#     # Rearrange them into a shape that the cross-entropy function likes when doing
#     # multidimensional loss.
#     logits_tensor = rearrange(
#         predicted_actions_logits_per_axis, "pose bsz toks dim -> pose dim bsz toks"
#     )

#     # Calculate the loss. We need to use the ignore_index argument to make sure that if we are not
#     # letting masked targets from contributing. We also disable reduction since we are going to
#     # reduce it ourselves later (see `reduce_fine_grained_loss()`)
#     loss = torch.nn.functional.cross_entropy(
#         logits_tensor, target_actions_per_axis, ignore_index=ignore_target_index, reduction="none"
#     )
#     loss[target_actions_per_axis == ignore_target_index] = float("nan")

#     # Make the shape the same as expected by the slow loss. This is so that we can easily reduce
#     # the losses across the correct axes
#     loss = rearrange(loss, "pose bsz toks -> bsz toks pose")

#     # Shape: (batch size, num obs, axis)
#     return loss


# def reduce_fine_grained_loss(fine_grained_loss: torch.Tensor) -> torch.Tensor:
#     """Reduce the fine-grained loss into a single number for backprop.

#     The fine-grained loss is a 3D tensor of an un-reduced loss across all the axes and actions,
#     with the shape (batch size, observations, axes).

#     To reduce the loss, we need to sum across the axes for a given timestep, and then we average
#     across all the timesteps within a batch, and then across all batches.

#     Since the fine-grained loss will likely have NaN's in it (due to the use of it in the mask
#     calculation before), we need to use the nan versions of the reduction operators.

#     The mean-ing must be performed in two steps since it is not identical to doing it in a single
#     step. For example, I ran the following during debugging and they are not identical:

#     ```python
#     > loss_per_batch_per_timestep_per_axis.sum(dim=-1).mean().item()
#     48.338687896728516

#     > loss_per_batch_per_timestep_per_axis.sum(dim=-1).mean(dim=-1).mean(dim=-1).item()
#     48.33868408203125
#     ```
#     """
#     # We average the loss across all of the axes for a given timestep.
#     loss_per_batch_per_timestep = fine_grained_loss.mean(dim=-1)

#     # Since every timestep is essentially a new example for the model to view, we can just mean
#     # across the the batch and timesteps in one go.
#     loss = loss_per_batch_per_timestep.nanmean()

#     assert loss.ndim == 0
#     return loss


def compute_fine_grained_loss_with_pose_axis_dependencies(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    *,
    ignore_target_index: int = -100,
    dependency_factor: float = 0.1,
) -> torch.Tensor:
    """
    Compute a fine-grained loss across all the poses and the axes per pose,
    incorporating dependencies between corresponding axes of Pose 0 and Pose 1.
    """
    predicted_logits_per_axis = rearrange(
        predicted_actions, "pose bsz toks dim -> pose dim bsz toks"
    )

    # Split targets
    target_pose0_position = target_actions[:3, :, :]
    target_pose0_rotation = target_actions[3:7, :, :]
    target_pose1_position = target_actions[7:10, :, :]
    target_pose1_rotation = target_actions[10:14, :, :]

    # Split predictions
    predicted_pose0_position = predicted_logits_per_axis[:3, :, :, :]
    predicted_pose0_rotation = predicted_logits_per_axis[3:7, :, :, :]
    predicted_pose1_position = predicted_logits_per_axis[7:10, :, :, :]
    predicted_pose1_rotation = predicted_logits_per_axis[10:14, :, :, :]

    # Initialize loss
    loss = torch.zeros(
        predicted_actions.shape[:3],
        dtype=torch.float32,
        device=predicted_actions.device,
    )

    # Pose 0 position loss
    pose0_position_loss = []
    for t in range(3):
        logits = rearrange(predicted_pose0_position[t, :, :, :], "dim bsz toks -> bsz dim toks")
        targets = target_pose0_position[t, :, :]

        loss_t = torch.nn.functional.cross_entropy(
            logits,
            targets,
            ignore_index=ignore_target_index,
            reduction="none",
        )
        pose0_position_loss.append(torch.nanmean(loss_t))
        loss[t, :, :] = loss_t

    # Pose 0 rotation loss
    pose0_rotation_loss = []
    for t in range(4):
        logits = rearrange(predicted_pose0_rotation[t, :, :, :], "dim bsz toks -> bsz dim toks")
        targets = target_pose0_rotation[t, :, :]

        loss_t = torch.nn.functional.cross_entropy(
            logits,
            targets,
            ignore_index=ignore_target_index,
            reduction="none",
        )
        pose0_rotation_loss.append(torch.nanmean(loss_t))
        loss[3 + t, :, :] = loss_t

    # Calculate loss for Pose 1 position with dependency on Pose 0 position (axis-specific)
    for t in range(3):
        logits = rearrange(predicted_pose1_position[t, :, :, :], "dim bsz toks -> bsz dim toks")
        targets = target_pose1_position[t, :, :]

        loss_t = torch.nn.functional.cross_entropy(
            logits,
            targets,
            ignore_index=ignore_target_index,
            reduction="none",
        )
        # Adjust by corresponding Pose 0 axis loss
        adjusted_loss_t = loss_t * (1 + dependency_factor * pose0_position_loss[t])
        loss[7 + t, :, :] = adjusted_loss_t

    # Calculate loss for Pose 1 rotation with dependency on Pose 0 rotation (axis-specific)
    for t in range(4):
        logits = rearrange(predicted_pose1_rotation[t, :, :, :], "dim bsz toks -> bsz dim toks")
        targets = target_pose1_rotation[t, :, :]

        loss_t = torch.nn.functional.cross_entropy(
            logits,
            targets,
            ignore_index=ignore_target_index,
            reduction="none",
        )
        # Adjust by corresponding Pose 0 axis loss
        adjusted_loss_t = loss_t * (1 + dependency_factor * pose0_rotation_loss[t])
        loss[10 + t, :, :] = adjusted_loss_t

    loss[target_actions == ignore_target_index] = float("nan")

    # Reshape and return the result
    loss = rearrange(loss, "pose bsz toks -> bsz toks pose")
    return loss


# def compute_fine_grained_loss_with_pose_dependencies(
#     predicted_actions_logits_per_axis: torch.Tensor,
#     target_actions: torch.Tensor,
#     *,
#     ignore_target_index: int = -100,
#     dependency_factor: float = 0.1,
# ) -> torch.Tensor:
#     """
#     Compute a fine-grained loss considering dependencies between Pose 0 and Pose 1.
#     """
#     logits_tensor = rearrange(
#         predicted_actions_logits_per_axis, "pose bsz toks dim -> pose dim bsz toks"
#     )

#     # 各ターゲットを分割
#     target_pose0_position = target_actions[:3, :, :]
#     target_pose0_rotation = target_actions[3:7, :, :]
#     target_pose1_position = target_actions[7:10, :, :]
#     target_pose1_rotation = target_actions[10:14, :, :]

#     # 各予測値を分割
#     predicted_pose0_position = logits_tensor[:3, :, :, :]
#     predicted_pose0_rotation = logits_tensor[3:7, :, :, :]
#     predicted_pose1_position = logits_tensor[7:10, :, :, :]
#     predicted_pose1_rotation = logits_tensor[10:14, :, :, :]

#     # 各損失を計算して蓄積
#     loss = torch.zeros(
#         predicted_actions_logits_per_axis.shape[:3],
#         dtype=torch.float32,
#         device=predicted_actions_logits_per_axis.device,
#     )

#     # Pose 0 の位置損失計算
#     pose0_position_loss = []
#     for t in range(3):
#         logits = rearrange(predicted_pose0_position[t, :, :, :], "dim bsz toks -> bsz dim toks")
#         targets = target_pose0_position[t, :, :]

#         loss_t = torch.nn.functional.cross_entropy(
#             logits,
#             targets,
#             ignore_index=ignore_target_index,
#             reduction="none",
#         )
#         pose0_position_loss.append(torch.nanmean(loss_t))
#         loss[t, :, :] = loss_t

#     # Pose 0 の回転損失計算
#     pose0_rotation_loss = []
#     for t in range(4):
#         logits = rearrange(predicted_pose0_rotation[t, :, :, :], "dim bsz toks -> bsz dim toks")
#         targets = target_pose0_rotation[t, :, :]

#         loss_t = torch.nn.functional.cross_entropy(
#             logits,
#             targets,
#             ignore_index=ignore_target_index,
#             reduction="none",
#         )
#         pose0_rotation_loss.append(torch.nanmean(loss_t))
#         loss[3 + t, :, :] = loss_t

#     # Pose 1 の位置損失計算 (Pose 0 の位置損失を考慮)
#     for t in range(3):
#         logits = rearrange(predicted_pose1_position[t, :, :, :], "dim bsz toks -> bsz dim toks")
#         targets = target_pose1_position[t, :, :]

#         loss_t = torch.nn.functional.cross_entropy(
#             logits,
#             targets,
#             ignore_index=ignore_target_index,
#             reduction="none",
#         )
#         adjusted_loss_t = loss_t * (1 + dependency_factor * torch.sum(torch.stack(pose0_position_loss)))
#         loss[7 + t, :, :] = adjusted_loss_t

#     # Pose 1 の回転損失計算 (Pose 0 の回転損失を考慮)
#     for t in range(4):
#         logits = rearrange(predicted_pose1_rotation[t, :, :, :], "dim bsz toks -> bsz dim toks")
#         targets = target_pose1_rotation[t, :, :]

#         loss_t = torch.nn.functional.cross_entropy(
#             logits,
#             targets,
#             ignore_index=ignore_target_index,
#             reduction="none",
#         )
#         adjusted_loss_t = loss_t * (1 + dependency_factor * torch.sum(torch.stack(pose0_rotation_loss)))
#         loss[10 + t, :, :] = adjusted_loss_t

#     loss[target_actions == ignore_target_index] = float("nan")

#     # 結果を整形して返す
#     loss = rearrange(loss, "pose bsz toks -> bsz toks pose")
#     return loss
