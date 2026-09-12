"""Tests for the backbone registry.

The equivalence test here is what licenses deleting the previous
`ResNetFeatureMap`, which hand-rolled the ResNet forward pass as
conv1 -> bn1 -> relu -> maxpool -> layer1..layer4. That hardcoding is exactly
what stopped any non-ResNet architecture from working, but replacing it is only
safe if the replacement is numerically identical -- otherwise every published
number shifts by an amount too small to notice.
"""

from __future__ import annotations

import pytest
import torch

from sparc.models.backbones import STAGE_NAMES, build_backbone, list_backbones

RESNETS = ["resnet18", "resnet34", "resnet50"]


def _legacy_feature_map(resnet, x):
    """The previous ResNetFeatureMap.forward, verbatim."""
    x = resnet.conv1(x)
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x)
    x = resnet.layer1(x)
    x = resnet.layer2(x)
    x = resnet.layer3(x)
    x = resnet.layer4(x)
    return x


class TestLegacyEquivalence:
    @pytest.mark.parametrize("name", RESNETS)
    def test_final_stage_matches_the_hand_written_forward_exactly(self, name):
        """Bit-identical, not merely close: torch.equal, no tolerance."""
        spec = build_backbone(name)
        spec.net.eval()
        extractor = spec.final_extractor().eval()
        x = torch.randn(2, 3, 64, 64)

        with torch.no_grad():
            new = extractor(x)["s4"]
            old = _legacy_feature_map(spec.net, x)

        assert new.shape == old.shape
        assert torch.equal(new, old), "extractor diverged from the legacy forward pass"

    def test_equivalence_holds_in_train_mode_too(self):
        """BatchNorm behaves differently in train mode; the paths must still agree."""
        spec = build_backbone("resnet18")
        spec.net.train()
        extractor = spec.final_extractor().train()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            assert torch.equal(extractor(x)["s4"], _legacy_feature_map(spec.net, x))


class TestSpecMetadata:
    @pytest.mark.parametrize("name", RESNETS)
    def test_stage_channels_match_a_real_forward_pass(self, name):
        """The FPN is built from stage_channels. If it were wrong, detection
        would fail at construction -- or worse, silently mis-wire."""
        spec = build_backbone(name)
        spec.net.eval()
        extractor = spec.make_extractor(STAGE_NAMES).eval()
        with torch.no_grad():
            outs = extractor(torch.randn(1, 3, 128, 128))
        observed = tuple(outs[s].shape[1] for s in STAGE_NAMES)
        assert observed == spec.stage_channels

    @pytest.mark.parametrize("name", RESNETS)
    def test_stage_strides_match_observed_downsampling(self, name):
        spec = build_backbone(name)
        spec.net.eval()
        extractor = spec.make_extractor(STAGE_NAMES).eval()
        size = 128
        with torch.no_grad():
            outs = extractor(torch.randn(1, 3, size, size))
        observed = tuple(size // outs[s].shape[-1] for s in STAGE_NAMES)
        assert observed == spec.stage_strides

    @pytest.mark.parametrize("name", RESNETS)
    def test_feature_dim_is_the_final_stage(self, name):
        spec = build_backbone(name)
        assert spec.feature_dim == spec.stage_channels[-1]

    def test_expansion_is_derived_not_hardcoded(self):
        """BasicBlock (expansion 1) vs Bottleneck (expansion 4)."""
        assert build_backbone("resnet18").stage_channels == (64, 128, 256, 512)
        assert build_backbone("resnet50").stage_channels == (256, 512, 1024, 2048)


class TestParameterSharing:
    def test_extractor_shares_parameters_with_the_trunk(self):
        """Momentum updates are applied to spec.net. If the extractor held
        copies, the momentum encoder would silently stop tracking."""
        spec = build_backbone("resnet18")
        extractor = spec.final_extractor()
        with torch.no_grad():
            spec.net.conv1.weight.fill_(0.123)
        found = dict(extractor.named_parameters())["conv1.weight"]
        assert torch.allclose(found, torch.full_like(found, 0.123))

    def test_state_dict_keys_are_plain_resnet_keys(self):
        """Downstream transfer strips prefixes and loads into a torchvision
        ResNet. Mangled key names (as torch.fx would produce) would break it."""
        spec = build_backbone("resnet18")
        keys = set(spec.net.state_dict())
        assert "conv1.weight" in keys
        assert "layer4.1.bn2.weight" in keys
        assert not any(k.startswith("fc.") for k in keys), "classification head not removed"


class TestRegistry:
    def test_lists_the_built_ins(self):
        assert {"resnet18", "resnet34", "resnet50", "resnet101"} <= set(list_backbones())

    def test_unknown_backbone_names_what_is_available(self):
        """This message replaces the argparse choices= list."""
        with pytest.raises(KeyError) as exc:
            build_backbone("resnet19")
        message = str(exc.value)
        assert "resnet19" in message
        assert "resnet18" in message, "error should list registered backbones"
        assert "timm:" in message, "error should mention the timm escape hatch"

    def test_partial_stage_selection(self):
        spec = build_backbone("resnet18")
        spec.net.eval()
        with torch.no_grad():
            outs = spec.make_extractor(("s2", "s4")).eval()(torch.randn(1, 3, 64, 64))
        assert set(outs) == {"s2", "s4"}

    def test_rejects_unknown_stage_names(self):
        spec = build_backbone("resnet18")
        with pytest.raises(ValueError, match="Unknown stages"):
            spec.make_extractor(("s9",))


class TestSpecValidation:
    def test_rejects_inconsistent_channel_counts(self):
        from sparc.models.backbones.base import BackboneSpec

        with pytest.raises(ValueError, match="stage_channels"):
            BackboneSpec(
                name="broken",
                net=torch.nn.Identity(),
                feature_dim=512,
                stage_channels=(64, 128),
                stage_strides=(4, 8, 16, 32),
                make_extractor=lambda stages: torch.nn.Identity(),
            )

    def test_rejects_feature_dim_that_is_not_the_last_stage(self):
        from sparc.models.backbones.base import BackboneSpec

        with pytest.raises(ValueError, match="feature_dim"):
            BackboneSpec(
                name="broken",
                net=torch.nn.Identity(),
                feature_dim=999,
                stage_channels=(64, 128, 256, 512),
                stage_strides=(4, 8, 16, 32),
                make_extractor=lambda stages: torch.nn.Identity(),
            )
