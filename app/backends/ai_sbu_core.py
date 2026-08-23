"""
AI SBU core — Small overlapping windows, Big stitched-back transcript,
United speaker labels.

Renamed/rebuilt from the old balanced-multi-GPU "chunk" path. Works
directly on the segment-dict shape run_ai_sbu_transcription already
produces (dicts with 'start', 'end', 'text', optional 'speaker') —
nothing here assumes a specific model. Any model whose parser fills
those keys works: a model that never sets 'speaker' just gets an
empty speaker map and passes through unchanged, and a model that
diarizes gets its per-window labels reconciled into one consistent
set of global speaker IDs.

No Tk, no subprocess — pure post-processing over what the per-window
workers already returned.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional


def stitch_segment_shards(shard_results: list[dict]) -> list[dict]:
    """
    shard_results: ordered by nominal_start (as produced by
    run_ai_sbu_transcription — one dict per window, each with
    'nominal_start' and 'segments' already offset to absolute time).

    Ownership rule: window 0 owns everything it produced. Every later
    window only contributes segments starting at/after its own
    nominal_start — anything earlier fell inside its LEADING overlap
    padding, which the previous window already owns from its TRAILING
    overlap padding. This avoids duplicate/garbled text at cut points
    without trying to fuzzy-match two independently-run transcriptions
    against each other (they may not even agree on wording).

    Each returned segment is tagged with '_shard_index' so
    apply_speaker_map can look up the right global label without a
    second time-based pass.
    """
    merged = []
    for i, shard in enumerate(shard_results):
        boundary = shard["nominal_start"]
        for seg in shard["segments"]:
            if i == 0 or seg["start"] >= boundary:
                tagged = dict(seg)
                tagged["_shard_index"] = i
                merged.append(tagged)
    merged.sort(key=lambda s: s["start"])
    return merged


def _overlap_evidence(prev: dict, cur: dict, overlap_s: float):
    boundary = cur["nominal_start"]
    prev_tail = [s for s in prev["segments"] if s["start"] >= boundary - overlap_s]
    cur_head = [s for s in cur["segments"] if s["start"] < boundary + overlap_s]
    return prev_tail, cur_head


def _heuristic_reconcile(prev_tail: list[dict], cur_head: list[dict]) -> dict:
    """
    Default reconciler: votes on which raw speaker labels co-occur at
    close timestamps in the overlap window. No AI call. Good enough
    when the overlap is unambiguous (one speaker talking through the
    boundary); leaves ambiguous cases unmapped so the caller mints a
    fresh global label instead of guessing wrong.
    """
    votes = defaultdict(int)
    for p in prev_tail:
        p_speaker = p.get("speaker")
        if not p_speaker:
            continue
        for c in cur_head:
            c_speaker = c.get("speaker")
            if not c_speaker:
                continue
            if abs(p["start"] - c["start"]) < max(2.0, p["end"] - p["start"]):
                votes[(p_speaker, c_speaker)] += 1

    mapping: dict = {}
    used = set()
    for (p_lbl, c_lbl), _ in sorted(votes.items(), key=lambda kv: -kv[1]):
        if p_lbl in used or c_lbl in mapping:
            continue
        mapping[c_lbl] = p_lbl
        used.add(p_lbl)
    return mapping  # cur raw label -> prev raw label


# Optional callback for ambiguous overlaps — same (prev_tail, cur_head)
# evidence as the heuristic, same return shape (cur raw label -> prev
# raw label). Wire your own model-specific logic or an LLM call here;
# return {} (or raise) to fall back to the heuristic.
ReconcilerFn = Callable[[list, list], dict]


def reconcile_speaker_labels(
    shard_results: list[dict],
    overlap_s: float = 4.0,
    reconciler: Optional[ReconcilerFn] = None,
) -> dict:
    """
    Returns {shard_index_str: {raw_label: global_label}}, one entry
    per window in shard_results (already ordered by nominal_start).
    A window whose segments carry no 'speaker' key at all (a model
    that doesn't diarize) gets an empty map and is left untouched —
    this is what makes the function safe to call unconditionally
    regardless of which model produced the segments.
    """
    if not shard_results:
        return {}

    speaker_map: dict = {}
    first_labels = sorted({
        s.get("speaker") for s in shard_results[0]["segments"] if s.get("speaker")
    })
    speaker_map["0"] = {lbl: lbl for lbl in first_labels}

    for i in range(1, len(shard_results)):
        prev, cur = shard_results[i - 1], shard_results[i]
        cur_labels = sorted({s.get("speaker") for s in cur["segments"] if s.get("speaker")})
        if not cur_labels:
            speaker_map[str(i)] = {}
            continue

        prev_tail, cur_head = _overlap_evidence(prev, cur, overlap_s)

        raw_to_prev = {}
        if reconciler is not None:
            try:
                raw_to_prev = reconciler(prev_tail, cur_head) or {}
            except Exception:
                raw_to_prev = {}
        if not raw_to_prev:
            raw_to_prev = _heuristic_reconcile(prev_tail, cur_head)

        prev_map = speaker_map[str(i - 1)]
        cur_map = {}
        for raw in cur_labels:
            prev_raw = raw_to_prev.get(raw)
            if prev_raw and prev_raw in prev_map:
                cur_map[raw] = prev_map[prev_raw]
            else:
                seen = {v for m in speaker_map.values() for v in m.values()}
                cur_map[raw] = f"Speaker_{len(seen)}"
        speaker_map[str(i)] = cur_map

    return speaker_map


def apply_speaker_map(segments: list[dict], speaker_map: dict) -> list[dict]:
    """
    Rewrite each stitched segment's 'speaker' using the global map
    (matched via the '_shard_index' tag stitch_segment_shards left
    behind), and strip that internal tag from the output.
    """
    out = []
    for seg in segments:
        idx = seg.get("_shard_index")
        new_seg = {k: v for k, v in seg.items() if k != "_shard_index"}
        raw = new_seg.get("speaker")
        if raw:
            gmap = speaker_map.get(str(idx), {})
            new_seg["speaker"] = gmap.get(raw, raw)
        out.append(new_seg)
    return out
