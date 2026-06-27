---
license: mit
base_model: Qwen/Qwen3.5-0.8B
library_name: pytorch
tags:
- interpretability
- activation-probing
- vision-language-models
- temporal
- qwen
---

# TM-NLA Checkpoints

These checkpoints support the TM-NLA public runtime.

TM-NLA (Temporal Multimodal Natural Language Autoencoder) is a proof-of-concept temporal activation-verbalization probe for frozen visual-temporal hidden states. It projects video-window activations from `Qwen/Qwen3.5-0.8B` into a continuous language-oriented manifold, then selects semantic readout hypotheses over time with compatibility and specificity annotations.

## Files

| File | Role | SHA-256 |
| --- | --- | --- |
| `temporal_probe.pt` | Temporal visual activation -> continuous language-manifold probe | `48AABF662423EA529C985855EA54970A9CE3256EC3F14AFE0169C2BF1A55B217` |
| `activation_verbalizer.pt` | Activation -> small set of short candidate readout proposals | `88DDB0267AE535A287755DE5C9E638EB771FFD5C5377051ACFCADF96A98C08C1` |
| `text_reconstructor.pt` | Text readout -> activation compatibility verifier/reranker | `C5CEA721A2173DCA884617258BB3F671668CF13EBA07728E11C39C9695B5DE32` |

## Base Model

The runtime uses frozen `Qwen/Qwen3.5-0.8B` activations. The public demo path does not train or fine-tune Qwen.

## Intended Use

Use these checkpoints with the TM-NLA GitHub repository to run local terminal-based semantic readouts over video windows:

https://github.com/DavWer/TM-NLA

The checkpoints are intended for interpretability and research exploration. They are not intended as a production captioner, classifier, benchmark system, or broad video-analysis product.

## Limitations

The continuous activation geometry is the strongest result. TM-NLA shows weak recoverable temporal semantic traces from frozen visual-temporal activations. Natural-language readouts are useful but noisy externalizations, not ground-truth captions.

Readouts can be malformed, wrong, generic, or underspecified. The activation verbalizer remains fragile, and candidate generation plus AR reranking is a practical readout mechanism rather than a solved decoder. The text reconstructor is a compatibility-based verifier/reranker, not an oracle. `specificity_status` is a numerical activation-specificity annotation, not a ground-truth correctness label.

## Citation

If you publish work building on this proof of concept, cite or link the companion TM-NLA GitHub repository:

https://github.com/DavWer/TM-NLA
