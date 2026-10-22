"""Install the pinned ML bundle; no training or application dependencies."""
import argparse
import json
from pathlib import Path
import shutil
import urllib.request

from component_inspector.vision.ksdd2_ensemble import (
    DEFAULT_BUNDLE, DEFAULT_MANIFEST, PROJECT, read_manifest, sha256, verify_bundle,
)


def install(destination: Path, *, backbone: Path | None = None, download: bool = False) -> dict:
    manifest = read_manifest()
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    registry = DEFAULT_MANIFEST.parent
    # Never replace a different model or change a head while installing.
    for member in manifest["members"]:
        name = member["filename"]
        source = registry / name
        if sha256(source) != manifest["files"][name]["sha256"]:
            raise ValueError("Committed head checksum mismatch")
        target = destination / name
        if target.exists():
            if sha256(target) != manifest["files"][name]["sha256"]:
                raise ValueError("Destination contains a different model")
        else:
            shutil.copyfile(source, target)
    target = destination / "backbone.safetensors"
    expected = manifest["files"][target.name]
    if not target.exists():
        if backbone is None:
            cached = PROJECT.parent / ".cache/recall-downloads/model.safetensors"
            backbone = cached if cached.is_file() else None
        if backbone is not None:
            if backbone.stat().st_size != expected["bytes"] or sha256(backbone) != expected["sha256"]:
                raise ValueError("Local backbone checksum mismatch")
            shutil.copyfile(backbone, target)
        elif download:
            temporary = destination / "backbone.download"
            request = urllib.request.Request(manifest["backbone"]["url"], headers={"User-Agent": "Tech4Bharat-ML-handoff"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as output:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > expected["bytes"]:
                        raise ValueError("Backbone download exceeds registered size")
                    output.write(chunk)
            if temporary.stat().st_size != expected["bytes"] or sha256(temporary) != expected["sha256"]:
                raise ValueError("Backbone download checksum mismatch; preserve failed download")
            temporary.rename(target)
        else:
            raise ValueError("Backbone not available: pass --backbone PATH or --download-backbone")
    verify_bundle(destination, manifest)
    target_manifest = destination / "manifest.json"
    if target_manifest.exists() and target_manifest.read_bytes() != DEFAULT_MANIFEST.read_bytes():
        raise ValueError("Destination manifest differs from pinned selection")
    if not target_manifest.exists():
        shutil.copyfile(DEFAULT_MANIFEST, target_manifest)
    return {"status": "verified", "model_id": manifest["model_id"],
            "identity": manifest["identity"], "bundle": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--download-backbone", action="store_true")
    args = parser.parse_args()
    print(json.dumps(install(args.destination, backbone=args.backbone, download=args.download_backbone), indent=2))


if __name__ == "__main__":
    main()
