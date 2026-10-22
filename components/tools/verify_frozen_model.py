"""Portability regression against archived TRAIN scores, never a new accuracy study."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import psutil
from PIL import Image

from component_inspector.vision.ksdd2_ensemble import DEFAULT_MANIFEST, Ensemble, PROJECT, sha256
from component_inspector.vision.resources import run_bounded


def verify(output: Path):
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    source = PROJECT.parent / manifest["evidence"]["source_record"]
    assert sha256(source) == manifest["evidence"]["source_record_sha256"]
    reference = json.loads(source.read_text(encoding="utf-8"))
    # The archived original TRAIN manifest supplies only original paths and hashes.
    original_manifest = PROJECT / "outputs/recall-round1/manifest.json"
    originals = json.loads(original_manifest.read_text(encoding="utf-8"))
    by = {row["id"]: row for row in originals["records"]}
    model = Ensemble()
    maximum = 0.; mismatches = 0; rows = []; timings = []
    for n, expected in enumerate(reference["predictions"]):
        row = by[expected["id"]]
        assert row["official_split"] == "train" and sha256(Path(row["path"])) == row["hashes"]["sha256"]
        tick = time.perf_counter()
        with Image.open(row["path"]) as image:
            actual = model.predict_image(image, with_attention=False)
        timings.append(time.perf_counter() - tick)
        difference = abs(actual["score"] - expected["score"])
        maximum = max(maximum, difference)
        mismatches += int(actual["flagged"] != expected["flagged"])
        rows.append({"id": expected["id"], "score": actual["score"], "flagged": actual["flagged"],
                     "reference_score": expected["score"], "absolute_difference": difference})
        if n % 100 == 0 or n + 1 == len(reference["predictions"]):
            print("PORTABLE ML REPLAY", n + 1, "/", len(reference["predictions"]), flush=True)
    assert len(rows) == 638 and maximum <= 1e-5 and mismatches == 0
    import numpy as np
    import component_inspector.vision.ksdd2_ensemble as runtime
    result = {"model_id": manifest["model_id"], "identity": manifest["identity"],
              "originals": len(rows), "maximum_score_difference": maximum, "flag_mismatches": mismatches,
              "source_record_sha256": sha256(source), "runtime_source_sha256": sha256(Path(runtime.__file__)),
              "verifier_sha256": sha256(Path(__file__)), "official_test_accessed": False,
              "feature_caches_or_masks_read": False, "accuracy_claim": "none; same original TRAIN scores replayed",
              "per_image_seconds_in_this_run": {"median": float(np.median(timings)), "p95": float(np.percentile(timings, 95))},
              "predictions": rows}
    with (output / "parity.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write("\n")
    print("PORTABLE ML ADAPTER MATCHED", maximum, mismatches, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        verify(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    cap = min(4 * 1024**3, int(.6 * psutil.virtual_memory().available))
    resource = run_bounded([sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(args.output)],
                           args.output / "worker.log", max_rss_bytes=cap, max_seconds=1800)
    with (args.output / "resources.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(resource, stream, indent=2, sort_keys=True); stream.write("\n")
    print(json.dumps(resource, indent=2))
    if resource["status"] != "completed":
        raise RuntimeError("Preserve failed portable adapter verification")


if __name__ == "__main__":
    main()
