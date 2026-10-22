"""Model distribution/integrity regressions; no new performance evaluation."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from component_inspector.vision.ksdd2_ensemble import DEFAULT_MANIFEST, read_manifest, verify_bundle


def sign(value):
    value = copy.deepcopy(value)
    value.pop("identity", None)
    value["identity"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value


def test_registry_heads_are_exactly_the_selected_archived_artifacts():
    manifest = read_manifest()
    assert manifest["calibration"]["alpha"] == .08
    assert manifest["calibration"]["n"] == 200 and manifest["calibration"]["k"] == 185
    assert manifest["evidence"]["original_primary_qualification_passed"] is False
    assert manifest["evidence"]["independent_factory_confirmation"] is False
    for member in manifest["members"]:
        path = DEFAULT_MANIFEST.parent / member["filename"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == member["sha256"]
        assert path.stat().st_size == manifest["files"][path.name]["bytes"]


@pytest.mark.parametrize("change", ["threshold", "seed", "method"])
def test_even_a_resigned_manifest_cannot_switch_the_selected_recipe(tmp_path, change):
    value = read_manifest()
    if change == "threshold":
        value["calibration"]["threshold"] = .5
    elif change == "seed":
        value["members"][0]["seed"] = 4217
    else:
        value["method"] = "tiles108"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(sign(value)), encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized"):
        read_manifest(path)


def test_accidental_manifest_edit_breaks_its_identity(tmp_path):
    value = read_manifest()
    value["research_status"] = "production_validated"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        read_manifest(path)


def test_missing_or_corrupt_tensor_is_rejected_before_loading(tmp_path):
    data = b"fixture, not model weights"
    manifest = {"files": {"head.pt": {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}}}
    with pytest.raises(ValueError, match="missing or changed"):
        verify_bundle(tmp_path, manifest, check_versions=False)
    (tmp_path / "head.pt").write_bytes(data)
    verify_bundle(tmp_path, manifest, check_versions=False)
    (tmp_path / "head.pt").write_bytes(b"x" * len(data))
    with pytest.raises(ValueError, match="missing or changed"):
        verify_bundle(tmp_path, manifest, check_versions=False)


def test_artifact_paths_cannot_escape_the_bundle(tmp_path):
    manifest = {"files": {"../outside.pt": {"sha256": "0" * 64, "bytes": 0}}}
    with pytest.raises(ValueError, match="Unsafe"):
        verify_bundle(tmp_path, manifest, check_versions=False)


def test_wrong_numerical_environment_is_rejected(tmp_path):
    manifest = {"files": {}, "numerical_versions": {"numpy": "0.0.0"}}
    with pytest.raises(ValueError, match="environment"):
        verify_bundle(tmp_path, manifest)
