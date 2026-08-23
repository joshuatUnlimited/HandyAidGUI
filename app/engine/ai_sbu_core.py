"""Minimal AI SBU stitching and speaker reconciliation."""

from __future__ import annotations

from copy import deepcopy
from itertools import permutations

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # optional; greedy/exhaustive fallback works without scipy
    linear_sum_assignment = None


MODEL_PRESETS = {
    "parakeet": {
        "label": "Parakeet — overlap matching",
        "min_overlap_s": 0.20,
    },
    "moss": {
        "label": "MOSS — overlap matching",
        "min_overlap_s": 0.15,
    },
    "generic": {
        "label": "Generic — overlap matching",
        "min_overlap_s": 0.20,
    },
}


def _speaker_key(value):
    if value is None or value == "":
        return None
    return str(value)


def _model_family(model_path=None):
    text = str(model_path or "").lower()
    if "parakeet" in text:
        return "parakeet"
    if "moss" in text:
        return "moss"
    return "generic"


def _overlap(a, b, start, end):
    left = max(float(a.get("start", 0.0)), float(b.get("start", 0.0)), start)
    right = min(float(a.get("end", 0.0)), float(b.get("end", 0.0)), end)
    return max(0.0, right - left)


def _pair_scores(prev_segments, curr_segments, overlap_start, overlap_end):
    prev_ids = sorted({
        _speaker_key(s.get("speaker"))
        for s in prev_segments
        if _speaker_key(s.get("speaker")) is not None
    })
    curr_ids = sorted({
        _speaker_key(s.get("speaker"))
        for s in curr_segments
        if _speaker_key(s.get("speaker")) is not None
    })

    scores = {}
    for prev_id in prev_ids:
        for curr_id in curr_ids:
            total = 0.0
            for a in prev_segments:
                if _speaker_key(a.get("speaker")) != prev_id:
                    continue
                for b in curr_segments:
                    if _speaker_key(b.get("speaker")) != curr_id:
                        continue
                    total += _overlap(a, b, overlap_start, overlap_end)
            scores[(prev_id, curr_id)] = total

    return prev_ids, curr_ids, scores


def _assign(prev_ids, curr_ids, scores):
    if not prev_ids or not curr_ids:
        return {}

    if linear_sum_assignment is not None:
        import numpy as np

        matrix = np.zeros((len(prev_ids), len(curr_ids)), dtype=float)
        for i, prev_id in enumerate(prev_ids):
            for j, curr_id in enumerate(curr_ids):
                matrix[i, j] = scores.get((prev_id, curr_id), 0.0)

        rows, cols = linear_sum_assignment(-matrix)

        return {
            curr_ids[j]: prev_ids[i]
            for i, j in zip(rows, cols)
            if matrix[i, j] > 0.0
        }

    # Tiny fallback for models with only a few speaker slots.
    best = {}
    best_score = -1.0
    width = min(len(prev_ids), len(curr_ids))

    for prev_perm in permutations(prev_ids, width):
        candidate = {}
        score = 0.0
        for curr_id, prev_id in zip(curr_ids[:width], prev_perm):
            value = scores.get((prev_id, curr_id), 0.0)
            score += value
            if value > 0:
                candidate[curr_id] = prev_id

        if score > best_score:
            best_score = score
            best = candidate

    return best


def stitch_segment_shards(results):
    """Stitch chunk results while dropping duplicated overlap segments."""
    stitched = []

    for chunk_index, result in enumerate(results):
        nominal_start = float(
            result.get("nominal_start", result.get("offset_s", 0.0))
        )

        for raw in result.get("segments", []) or []:
            seg = deepcopy(raw)
            seg["_chunk_index"] = chunk_index

            midpoint = (
                float(seg.get("start", 0.0))
                + float(seg.get("end", 0.0))
            ) / 2.0

            # Previous chunk owns the overlap region.
            if chunk_index and midpoint < nominal_start:
                continue

            stitched.append(seg)

    stitched.sort(
        key=lambda s: (
            float(s.get("start", 0.0)),
            float(s.get("end", 0.0)),
        )
    )
    return stitched


def reconcile_speaker_labels(results, overlap_s=4.0, model_path=None):
    """Map local chunk speaker slots onto stable global Speaker N labels."""
    preset = MODEL_PRESETS[_model_family(model_path)]
    speaker_map = {}
    next_global_id = 1

    for chunk_index, result in enumerate(results):
        current_segments = result.get("segments", []) or []
        current_ids = sorted({
            _speaker_key(s.get("speaker"))
            for s in current_segments
            if _speaker_key(s.get("speaker")) is not None
        })

        if chunk_index == 0:
            for local_id in current_ids:
                speaker_map[(chunk_index, local_id)] = (
                    f"Speaker {next_global_id}"
                )
                next_global_id += 1
            continue

        previous = results[chunk_index - 1]
        previous_segments = previous.get("segments", []) or []

        previous_start = float(previous.get("offset_s", 0.0))
        current_start = float(result.get("offset_s", 0.0))

        previous_end = max(
            (float(s.get("end", previous_start)) for s in previous_segments),
            default=previous_start,
        )

        overlap_start = max(previous_start, current_start)
        overlap_end = min(
            previous_end,
            current_start + overlap_s,
        )

        assignments = {}

        if overlap_end - overlap_start >= preset["min_overlap_s"]:
            prev_ids, curr_ids, scores = _pair_scores(
                previous_segments,
                current_segments,
                overlap_start,
                overlap_end,
            )
            assignments = _assign(prev_ids, curr_ids, scores)

        for local_id in current_ids:
            previous_local_id = assignments.get(local_id)

            global_id = None
            if previous_local_id is not None:
                global_id = speaker_map.get(
                    (chunk_index - 1, previous_local_id)
                )

            if global_id is None:
                global_id = f"Speaker {next_global_id}"
                next_global_id += 1

            speaker_map[(chunk_index, local_id)] = global_id

    return speaker_map


def apply_speaker_map(segments, speaker_map):
    output = []

    for seg in segments:
        item = dict(seg)
        chunk_index = item.pop("_chunk_index", None)
        local_id = _speaker_key(item.get("speaker"))

        if local_id is not None and chunk_index is not None:
            item["speaker"] = speaker_map.get(
                (chunk_index, local_id),
                item.get("speaker"),
            )

        output.append(item)

    return output


def load_united_file(path):
    import json

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("segments", "results", "chunks"):
            if isinstance(data.get(key), list):
                return data[key]

    return []


def save_united_file(path, segments):
    import json
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {"segments": segments},
            fh,
            ensure_ascii=False,
            indent=2,
        )


__all__ = [
    "MODEL_PRESETS",
    "stitch_segment_shards",
    "reconcile_speaker_labels",
    "apply_speaker_map",
    "load_united_file",
    "save_united_file",
]
