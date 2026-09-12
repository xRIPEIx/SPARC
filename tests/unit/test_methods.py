"""Tests for the SSL method interface.

The point of the interface is that the trainer knows nothing method-specific.
These tests treat the three built-ins uniformly, which is the property that
makes adding a fourth a config change rather than a code change.
"""

from __future__ import annotations

import pytest
import torch

from sparc.methods import SSLBatch, get_method, list_methods
from sparc.models.backbones import build_backbone

SMALL = dict(proj_dim=16, memory_bank_size=32)


@pytest.fixture(scope="module")
def spec():
    return build_backbone("resnet18")


def _batch(method, size=2, hw=32):
    batch = SSLBatch(torch.randn(size, 3, hw, hw), torch.randn(size, 3, hw, hw))
    if method.requires_region_masks:
        mask = torch.zeros(size, hw, hw, dtype=torch.long)
        mask[:, hw // 2 :, :] = 1
        mask[:, :, hw // 2 :] += 2
        batch = SSLBatch(batch.view_q, batch.view_k, mask, mask.clone())
    return batch


@pytest.mark.parametrize("name", sorted(list_methods()))
class TestUniformInterface:
    def test_training_step_returns_a_finite_scalar_loss(self, name, spec):
        method = get_method(name)(spec, **SMALL)
        out = method.training_step(_batch(method))
        assert out.loss.ndim == 0
        assert torch.isfinite(out.loss)
        assert "loss" in out.metrics

    def test_loss_is_differentiable(self, name, spec):
        method = get_method(name)(spec, **SMALL)
        method.training_step(_batch(method)).loss.backward()
        grads = [p.grad for p in method.parameters() if p.requires_grad and p.grad is not None]
        assert grads, "no parameter received a gradient"
        assert all(torch.isfinite(g).all() for g in grads)

    def test_encoder_state_dict_is_plain_backbone_keys(self, name, spec):
        state = get_method(name)(spec, **SMALL).encoder_state_dict()
        assert "conv1.weight" in state
        assert not any("projection_head" in k or "region_projector" in k for k in state)

    def test_momentum_encoder_is_frozen_and_tracks(self, name, spec):
        """The key encoder must not receive gradients, but must follow the query
        encoder through the EMA update."""
        method = get_method(name)(spec, **SMALL)
        momentum_params = [p for n, p in method.named_parameters() if "momentum" in n]
        assert momentum_params, "no momentum parameters found"
        assert not any(p.requires_grad for p in momentum_params)

        before = method.backbone_momentum.state_dict()["conv1.weight"].clone()
        with torch.no_grad():
            for p in method.backbone.parameters():
                p.add_(1.0)
        method.update_momentum(0.5)
        after = method.backbone_momentum.state_dict()["conv1.weight"]
        assert not torch.equal(before, after), "momentum encoder did not track the online one"


class TestSPARCSpecifics:
    def test_requires_region_masks_is_declared(self, spec):
        assert get_method("sparc")(spec, **SMALL).requires_region_masks is True
        assert get_method("moco")(spec, **SMALL).requires_region_masks is False
        assert get_method("densecl")(spec, **SMALL).requires_region_masks is False

    def test_missing_masks_raise_a_directive_error(self, spec):
        method = get_method("sparc")(spec, **SMALL)
        batch = SSLBatch(torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32))
        with pytest.raises(ValueError, match="sparc-masks"):
            method.training_step(batch)

    def test_lambda_zero_ignores_the_region_term(self, spec):
        """At lambda_region=0 the objective is exactly MoCo-global. This is what
        makes the SPARC and DenseCL lambda sweeps meet the MoCo baseline at a
        single shared point."""
        method = get_method("sparc")(spec, lambda_region=0.0, **SMALL)
        out = method.training_step(_batch(method))
        assert out.loss.item() == pytest.approx(out.metrics["loss_global"], abs=1e-5)

    def test_lambda_one_ignores_the_global_term(self, spec):
        method = get_method("sparc")(spec, lambda_region=1.0, **SMALL)
        out = method.training_step(_batch(method))
        assert out.loss.item() == pytest.approx(out.metrics["loss_region"], abs=1e-5)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_rejects_lambda_outside_the_unit_interval(self, spec, bad):
        with pytest.raises(ValueError, match="lambda_region"):
            get_method("sparc")(spec, lambda_region=bad, **SMALL)


class TestDenseCLSpecifics:
    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_rejects_dense_lambda_outside_the_unit_interval(self, spec, bad):
        with pytest.raises(ValueError, match="dense_lambda"):
            get_method("densecl")(spec, dense_lambda=bad, **SMALL)


class TestRegistry:
    def test_unknown_method_names_what_is_available(self):
        with pytest.raises(KeyError) as exc:
            get_method("simclr")
        assert "sparc" in str(exc.value)

    def test_built_ins_are_registered(self):
        assert set(list_methods()) == {"sparc", "moco", "densecl"}
