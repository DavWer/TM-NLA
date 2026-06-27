# Findings

## Main Result

TM-NLA (Temporal Multimodal Natural Language Autoencoder) demonstrates that frozen visual-temporal hidden activations can be projected into a language-manifold probe while preserving weak recoverable temporal semantic traces.

The strongest evidence is not the generated text. The strongest evidence is that activation trajectories preserve locality and event-like structure after projection. The language readout is fragile, but the system demonstrates that some evolving semantic structure can be probed and partially externalized.

## Text Readout

The text layer is best understood as a calibrated microscope:

- The activation verbalizer proposes a small fixed set of short candidate readouts.
- The text reconstructor maps each phrase back into activation space.
- The selected readout is chosen only by text-to-activation specificity, without lexical gating.
- The `specificity_status` annotation is numerical only: it compares target activation compatibility against a control activation, and is not a ground-truth correctness label.

This is more faithful than taking raw generated text at face value, but it is still imperfect.

## Negative Results

Several plausible improvements did not become the final stack:

- Deeper AV capacity did not improve the proof-of-concept checkpoint.
- Ground-up mixed-curriculum training did not outperform the accumulated general probe.
- Some caption/RL/joint refinement attempts introduced caption-prior drift and were not selected as the final stack.
- Self-training the AV against the improved reconstructor did not beat the older AV generator.

These negative results are part of the final project boundary. They suggest the remaining bottleneck is not simply more training or a larger adapter.

## Final Stack

- Temporal geometry probe: selected full-precision mixed-curriculum checkpoint.
- Text readout proposer: selected clean-caption AV SFT checkpoint.
- Text compatibility verifier: selected soft-caption-calibrated reconstructor.

## Interpretation

The project supports a modest claim:

> A frozen VLM's visual-temporal activations can be probed through a language-oriented manifold in a way that exposes weak recoverable temporal semantic traces.

It does not support the stronger claim:

> The system can reliably produce faithful open-ended captions for arbitrary video.

Open-ended AV verbalization remains fragile. Small candidate search is used as a practical readout mechanism, not as evidence of a solved decoder.
