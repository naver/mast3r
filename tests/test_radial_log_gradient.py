"""The radial logarithmic map extends differentiably to the origin."""

import pytest
import torch

from mast3r.losses import Regr3D, apply_log_to_norm
from dust3r.losses import L21Loss


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_radial_log_origin_jacobian_is_identity(dtype):
    value = torch.zeros(3, dtype=dtype, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(apply_log_to_norm, value)
    torch.testing.assert_close(jacobian, torch.eye(3, dtype=dtype))


def test_radial_log_gradcheck_includes_zero():
    value = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(apply_log_to_norm, (value,))


def test_small_vectors_keep_the_identity_limit():
    value = torch.tensor(
        [[1e-12, -2e-12, 3e-12]], dtype=torch.float64, requires_grad=True
    )
    result = apply_log_to_norm(value)
    torch.testing.assert_close(result, value, atol=0, rtol=1e-10)
    result.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value), atol=1e-10, rtol=0)


def test_nonzero_radial_value_and_jacobian_follow_closed_form():
    value = torch.tensor([2.0, -3.0, 4.0], dtype=torch.float64, requires_grad=True)
    radius = value.detach().norm()
    scale = radius.log1p() / radius
    expected = value.detach() * scale
    coefficient = (radius / (1 + radius) - radius.log1p()) / radius**3
    expected_jacobian = scale * torch.eye(
        3, dtype=value.dtype
    ) + coefficient * torch.outer(value.detach(), value.detach())
    torch.testing.assert_close(apply_log_to_norm(value), expected)
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(apply_log_to_norm, value), expected_jacobian
    )


def test_log_regression_can_update_zero_point_predictions():
    first = torch.nn.Parameter(torch.zeros((1, 2, 2, 3), dtype=torch.float64))
    second = torch.nn.Parameter(torch.zeros_like(first))
    target = torch.tensor([1.0, -2.0, 3.0], dtype=first.dtype).expand_as(first)
    gt = {
        "pts3d": target,
        "camera_pose": torch.eye(4, dtype=first.dtype)[None],
        "valid_mask": torch.ones((1, 2, 2), dtype=torch.bool),
        "sky_mask": torch.zeros((1, 2, 2), dtype=torch.bool),
        "is_metric_scale": torch.ones(1, dtype=torch.bool),
    }
    criterion = Regr3D(L21Loss(), norm_mode="", sky_loss_value=0, loss_in_log=True)
    optimizer = torch.optim.SGD([first, second], lr=0.05)
    initial_loss, _ = criterion(
        gt, gt, {"pts3d": first}, {"pts3d_in_other_view": second}
    )
    initial_loss.backward()
    expected = -(target / target.norm(dim=-1, keepdim=True)) / 4
    torch.testing.assert_close(first.grad, expected)
    torch.testing.assert_close(second.grad, expected)
    optimizer.step()
    updated_loss, _ = criterion(
        gt, gt, {"pts3d": first}, {"pts3d_in_other_view": second}
    )
    assert updated_loss < initial_loss
