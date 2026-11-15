"""Adapter: video-encoding configuration (project stage: run)."""

from src.core.schema import ROOT, DatasetSpec

SPEC = DatasetSpec(
    name="x264",
    label="video encoding",
    client="video-platform engineer",
    path=ROOT / "moot/optimize/config/X264_AllMeasurements.csv",
    objectives=["Performance-"],
    # One reference frame (ref_1=1) versus more: is the "fast" setting really faster?
    contrast=("ref_1", "1", "0"),
    requirements={
        1: "Summarize the encoder settings and quality measures in our historical runs.",
        2: "Predict encoding time before running a new configuration.",
        3: "Tell us whether a faster configuration is meaningfully faster or within normal noise.",
        4: "Evaluate configurations as benchmark results arrive one at a time.",
        5: "Explain which settings are responsible for slow encodes.",
        6: "Justify a recommended configuration to a platform administrator.",
        7: "Select a configuration that minimizes encoding time under a budget.",
        8: "Suggest the smallest setting change that improves a slow configuration.",
        9: "Detect an anomalous benchmark result or possible measurement error.",
        10: "Compare our tuner against a baseline search strategy.",
    },
)
