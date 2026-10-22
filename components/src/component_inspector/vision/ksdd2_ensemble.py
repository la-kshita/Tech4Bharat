"""Portable, hash-checked inference for the one selected MVP ensemble.

No training, research manifests, image labels or dataset paths are used here.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from component_inspector.vision.geometry import prepare, restore_map
from component_inspector.vision.media import decode_media

PROJECT = Path(__file__).resolve().parents[3]
MODEL_ID = "ksdd2-ensemble-mvp-v1"
DEFAULT_MANIFEST = PROJECT / "model_registry" / MODEL_ID / "manifest.json"
DEFAULT_BUNDLE = PROJECT / "models" / MODEL_ID


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    identity = data.pop("identity", None)
    expected = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if identity != expected or data.get("model_id") != MODEL_ID:
        raise ValueError("Model manifest identity mismatch")
    data["identity"] = identity
    if (data["calibration"]["threshold"] != 0.33108379983608666
            or data["calibration"]["comparison"] != "strict_greater"
            or [r["seed"] for r in data["members"]] != [4220, 4221, 4222]
            or data["method"] != "longer108"
            or data["recipe_id"] != "ksdd2_surface"):
        raise ValueError("Unrecognized MVP recipe or calibration")
    return data


def verify_bundle(bundle: Path, manifest: dict, *, check_versions: bool = True) -> None:
    for name, record in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe model filename")
        path = bundle / name
        if (not path.is_file() or path.stat().st_size != record["bytes"]
                or sha256(path) != record["sha256"]):
            raise ValueError("Model artifact missing or changed")
    if check_versions:
        for name, expected in manifest["numerical_versions"].items():
            if version(name) != expected:
                raise ValueError("Model numerical environment mismatch")


class Ensemble:
    def __init__(self, bundle: Path = DEFAULT_BUNDLE, manifest_path: Path = DEFAULT_MANIFEST):
        import torch
        import timm
        from safetensors.torch import load_file

        self.manifest = read_manifest(manifest_path)
        verify_bundle(bundle, self.manifest)
        torch.set_num_threads(4)
        self.backbone = timm.create_model("resnet18", pretrained=False, features_only=True,
                                          out_indices=(2, 3)).cpu().eval()
        weights = load_file(str(bundle / "backbone.safetensors"), device="cpu")
        keys = set(self.backbone.state_dict())
        self.backbone.load_state_dict({key: weights[key] for key in keys}, strict=True)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.states = [torch.load(bundle / row["filename"], map_location="cpu", weights_only=True)
                       for row in self.manifest["members"]]
        for state in self.states:
            if (set(state["head"]) != {"segment.0.weight", "segment.0.bias", "segment.2.weight",
                                       "segment.2.bias", "classify.weight", "classify.bias"}
                    or state["mean"].shape != (1, 384, 1, 1)
                    or state["std"].shape != (1, 384, 1, 1)
                    or not torch.isfinite(state["mean"]).all()
                    or not torch.isfinite(state["std"]).all() or (state["std"] <= 0).any()
                    or not all(torch.isfinite(v).all() for v in state["head"].values())):
                raise ValueError("Invalid ensemble state")
        self.mean = torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1)

    def predict_image(self, image: Image.Image, *, with_attention: bool = True) -> dict:
        import torch
        from torch.nn import functional as F

        with torch.inference_mode():
            prepared = prepare(image, self.manifest["geometry"])
            x = torch.from_numpy(prepared.rgb_float32.transpose(2, 0, 1).copy())[None]
            layer2, layer3 = self.backbone((x - self.mean) / self.std)
            # Half-rounding is part of the original cache/native inference recipe.
            features = F.avg_pool2d(torch.cat([layer2, F.interpolate(layer3, size=layer2.shape[-2:],
                                                                    mode="nearest")], 1), 3, 1, 1).half().float()
            roi = F.interpolate(torch.from_numpy(prepared.valid_roi.astype(np.float32))[None, None],
                                size=features.shape[-2:], mode="nearest")[0, 0].bool()
            logits, maps = [], []
            for state in self.states:
                head = state["head"]
                raw = F.conv2d(F.relu(F.conv2d((features - state["mean"]) / state["std"],
                                              head["segment.0.weight"], head["segment.0.bias"])),
                               head["segment.2.weight"], head["segment.2.bias"])[0, 0]
                values = raw[roi]
                if not len(values) or not torch.isfinite(values).all():
                    raise ValueError("Nonfinite model output")
                logits.append(float(values.topk(max(1, math.ceil(.001 * len(values)))).values.mean()))
                if with_attention:
                    maps.append(raw.sigmoid())
            scores = [1 / (1 + math.exp(-value)) if value >= 0 else
                      math.exp(value) / (1 + math.exp(value)) for value in logits]
            score = math.fsum(scores) / 3
            threshold = self.manifest["calibration"]["threshold"]
            result = {"model_id": MODEL_ID, "model_identity": self.manifest["identity"],
                      "recipe_id": "ksdd2_surface", "score": score, "threshold": threshold,
                      "flagged": score > threshold, "member_logits": logits,
                      "status": "flagged" if score > threshold else "not_flagged",
                      "research_status": "experimental_human_review_required"}
            if with_attention:
                attention = torch.stack(maps).mean(0)[None, None]
                canvas = F.interpolate(attention, size=(256, 640), mode="bilinear", align_corners=False)[0, 0].numpy()
                original = restore_map(canvas, prepared.geometry)
                rgba = np.zeros((*original.shape, 4), np.uint8)
                rgba[:, :, :3] = [239, 100, 55]
                rgba[:, :, 3] = np.rint(np.clip(original, 0, 1) * 180).astype(np.uint8)
                stream = io.BytesIO()
                Image.fromarray(rgba).save(stream, format="PNG")
                result["attention_png"] = base64.b64encode(stream.getvalue()).decode("ascii")
            return result

    def predict_bytes(self, data: bytes) -> dict:
        decoded = decode_media(data)
        result = self.predict_image(decoded.image)
        result["warnings"] = decoded.warnings
        result["width"], result["height"] = decoded.image.size
        width, height = decoded.image.size
        if not (184 <= width <= 242 and 602 <= height <= 660):
            result["warnings"].append("outside_ksdd2_capture_dimensions")
        return result
