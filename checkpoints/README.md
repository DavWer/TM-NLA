# Checkpoints

Checkpoint weights are not stored in this GitHub repository. Download them from the companion Hugging Face repository and place the files in this directory:

```text
https://huggingface.co/davwer/tm-nla-checkpoints
```

Expected filenames:

| Runtime name | Role |
| --- | --- |
| `temporal_probe.pt` | Projects frozen visual-temporal hidden states into the continuous language-manifold probe. |
| `activation_verbalizer.pt` | Proposes a small set of short candidate readouts from probed activations. |
| `text_reconstructor.pt` | Checks and reranks readout-to-activation compatibility by reconstructing the frozen target activation. |

Expected SHA-256 hashes:

```text
temporal_probe.pt        48AABF662423EA529C985855EA54970A9CE3256EC3F14AFE0169C2BF1A55B217
activation_verbalizer.pt 88DDB0267AE535A287755DE5C9E638EB771FFD5C5377051ACFCADF96A98C08C1
text_reconstructor.pt    C5CEA721A2173DCA884617258BB3F671668CF13EBA07728E11C39C9695B5DE32
```

These files are ignored by git by default.
