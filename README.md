# TM-NLA

**Temporal Multimodal Natural Language Autoencoder**

TM-NLA (Temporal Multimodal Natural Language Autoencoder) is a proof-of-concept temporal activation-verbalization probe for frozen visual-temporal hidden states. It maps video-window activations from `Qwen/Qwen3.5-0.8B` into a continuous language-oriented manifold, then externalizes semantic readout hypotheses over time with compatibility annotations.

The goal is not video classification, ground-truth caption generation, production video understanding, a full replication of Anthropic NLA, or a solved activation-to-language decoder. The goal is to test whether internal visual-temporal activations contain partially recoverable temporal semantic traces that can be externalized as readout hypotheses.

## What It Does

1. Samples temporal windows from a video.
2. Extracts frozen `Qwen/Qwen3.5-0.8B` visual hidden activations.
3. Adds a small temporal contextualizer around each window.
4. Projects the contextual activation through a learned language-manifold probe.
5. Uses an activation verbalizer to propose a small fixed set of short candidate readouts from each hidden activation.
6. Uses a text reconstructor to score how specifically each phrase reconstructs the original activation.
7. Prints the semantic readout timeline with compatibility and specificity annotations in the terminal.

## Final Runtime Stack

| Component | Role |
| --- | --- |
| Base VLM | Frozen `Qwen/Qwen3.5-0.8B` |
| `temporal_probe.pt` | Temporal visual activation -> continuous language-manifold probe |
| `activation_verbalizer.pt` | Activation -> small set of candidate readout proposals |
| `text_reconstructor.pt` | Text readout -> activation compatibility verifier/reranker |

The released demo path does not train Qwen or any project checkpoint.

## Setup

Use Python 3.11 or 3.12.

PyTorch is required, but it is intentionally installed separately from `requirements.txt` because the correct wheel depends on your OS, Python version, and compute platform (CPU, CUDA, ROCm, or MPS). Choose the right command from the official PyTorch selector, then install the remaining project dependencies:

```powershell
# Step 1: install PyTorch for your machine:
# https://pytorch.org/get-started/locally/

# Step 2: install TM-NLA dependencies:
python -m pip install -r requirements.txt
```

Do not treat any one CUDA wheel as the public default. Older NVIDIA GPUs, newer NVIDIA GPUs, CPU-only systems, ROCm systems, and Apple Silicon can need different PyTorch installs.

You can verify the selected PyTorch build with:

```powershell
python -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available())"
```

The default public commands use non-quantized `fp16` Qwen loading on CUDA.

Download checkpoints from the companion Hugging Face repository and place them in `checkpoints/`:

```text
https://huggingface.co/davwer/tm-nla-checkpoints
```

A model card for the checkpoint repository is available at [docs/HUGGINGFACE_MODEL_CARD.md](docs/HUGGINGFACE_MODEL_CARD.md).

Expected local filenames:

```text
checkpoints/temporal_probe.pt
checkpoints/activation_verbalizer.pt
checkpoints/text_reconstructor.pt
```

Expected SHA-256 hashes:

```text
temporal_probe.pt        48AABF662423EA529C985855EA54970A9CE3256EC3F14AFE0169C2BF1A55B217
activation_verbalizer.pt 88DDB0267AE535A287755DE5C9E638EB771FFD5C5377051ACFCADF96A98C08C1
text_reconstructor.pt    C5CEA721A2173DCA884617258BB3F671668CF13EBA07728E11C39C9695B5DE32
```

## Run On Your Own Video

After installing dependencies and downloading the checkpoints, run TM-NLA on a local video with one command:

```powershell
python scripts\run_video_readout.py --video path\to\your_video.mp4
```

The script:

1. samples temporal windows from the video,
2. extracts frozen Qwen visual activations,
3. applies the temporal language-manifold probe,
4. runs activation-conditioned candidate readout search,
5. selects one phrase per window using text-to-activation compatibility,
6. prints the semantic timeline with compatibility and specificity annotations in the terminal.

You can pass multiple videos by repeating `--video`:

```powershell
python scripts\run_video_readout.py --video path\to\first_video.mp4 --video path\to\second_video.mov
```

Useful options:

```powershell
python scripts\run_video_readout.py --video path\to\your_video.mp4 --num_windows 9 --candidates_per_window 16 --max_new_tokens 12
```

The wrapper writes temporary trajectory artifacts and readout tables under `outputs/inference/`, which is ignored by git. It does not create a web page.

Treat the printed text as a selected semantic hypothesis: an activation-conditioned readout checked by reconstruction compatibility, not a ground-truth caption or a claim of perfect model introspection.

## How The Readout Is Selected

TM-NLA is inspired by the Natural Language Autoencoder pattern:

1. Hidden activation -> activation verbalizer -> small fixed set of short candidate readouts.
2. Candidate readout -> text reconstructor -> reconstructed activation vector.
3. The phrase with the strongest specificity margin is selected for the window.
4. The selected phrase is annotated with compatibility and specificity metrics.

The default release setting is `--candidates_per_window 16` and `--max_new_tokens 12`. The sampled candidate path uses `temperature=0.6`, `top_p=0.9`, and includes one deterministic no-sampling candidate. Increasing the candidate count can make outputs look better by increasing search breadth, so any demo or result discussion should report the candidate count.

The public output shows only the selected phrase per window. Candidate-level tables are written only with `--write_candidates` or `--debug`.

## Reading The Output

Each timeline row contains:

- `timestamp`: timestamp in seconds.
- generated phrase: the selected candidate readout for that window.
- `compatibility`: cosine similarity between the activation reconstructed from the generated phrase and the original target activation. This asks: "Does this phrase map back near the activation it is supposed to describe?"
- `control_compatibility`: cosine similarity between the same phrase reconstruction and a control activation from another temporal window in the same run. This asks: "Would this phrase also fit some other window about equally well?"
- `specificity_margin`: `compatibility - control_compatibility`. This asks: "How much more specific is the phrase to its own window than to the control window?"
- `specificity_status`: numerical activation-specificity annotation derived only from compatibility metrics, never from phrase vocabulary.

These metrics annotate the readout. They are not ground truth, proof of correctness, or statistical significance tests. The control comparison is a lightweight specificity check. Raw compatibility can be high because the activation space contains shared/common directions, so `specificity_margin` is often more informative than compatibility alone. A low or negative `specificity_margin` does not prove the phrase is false; it means the phrase is not very specific under this control.

`specificity_status` asks whether the generated phrase reconstructs its target activation better than a control activation. The public thresholds are:

- `specific`: `compatibility >= 0.85` and `specificity_margin >= 0.03`.
- `weak`: `specificity_margin >= 0.00`, but below the `specific` threshold.
- `non_specific`: `specificity_margin < 0.00`.

Malformed, generic, symbolic, or nonsensical text may appear. TM-NLA does not hide those outputs with lexical filters; exposed verbalizer failures are part of the readout.

The summary values are averages over windows:

- `candidates/window`: number of candidate readouts considered per temporal window.
- `mean compatibility`: average text-to-activation reconstruction cosine.
- `mean control compatibility`: average control compatibility.
- `mean specificity margin`: average specificity margin.

## Lower-Level Commands

The one-command wrapper is the recommended public path. For debugging or custom workflows, the lower-level scripts can be run directly.

Create a manifest with one row per video:

```csv
video_name,tag
example_video.mp4,example_video
second_video.mov,second_video
```

Save that file anywhere locally, then run trajectory extraction. For example, if the videos live in `data/`:

```powershell
python scripts\extract_trajectories.py --video_dir data --metadata_csv path\to\manifest.csv --split . --checkpoint checkpoints\temporal_probe.pt --qwen_precision fp16 --lm_forward_precision fp32 --num_windows 9 --save_thumbnails --thumbnail_dir outputs\thumbnails --output outputs\trajectories.pt
```

Then print semantic readouts:

```powershell
python scripts\generate_readout.py --archive outputs\trajectories.pt --av_checkpoint checkpoints\activation_verbalizer.pt --text_reconstructor_checkpoint checkpoints\text_reconstructor.pt --target_key h_temporal --qwen_precision fp16 --lm_forward_precision fp32 --candidates_per_window 16 --max_new_tokens 12
```

The readout script prints a summary and semantic timeline directly in the terminal.

By default, `generate_readout.py` writes:

```text
outputs/readout/selected_readout_windows.csv
outputs/readout/selected_timeline.csv
```

## Findings

TM-NLA shows weak recoverable temporal semantic traces from frozen visual-temporal activations. The language readout is fragile, but the system demonstrates that some evolving semantic structure can be probed and partially externalized.

The strongest result is geometric: the probe preserves meaningful visual-temporal structure in a language-oriented activation space. Temporal contextualization improved locality and event-like structure compared with a frame-centered baseline.

The text readout is useful but imperfect. The activation verbalizer is the most fragile part of the system: single no-sampling AV decoding was tested and often collapsed into malformed text, symbols, whitespace, or generic fragments. TM-NLA therefore uses a small candidate search plus text-to-activation reconstruction as a practical readout mechanism.

## Limitations

- TM-NLA is a proof of concept, not a general video captioner.
- TM-NLA is not a classifier, production video-understanding tool, full Anthropic NLA replication, or solved activation-to-language decoder.
- Selected readouts are fragile and can be wrong, generic, malformed, or underspecified.
- The text reconstructor is a compatibility-based verifier/reranker, not a ground-truth oracle.
- Candidate search is a practical readout mechanism, not a solved decoder.
- Exact caption identity is a poor target because multiple descriptions can be semantically valid.
- The project was developed under heavy consumer-GPU constraints.

## Final Demo Assets

Final public demo assets live under `examples/final_demo/`.

- Primary demo: Basketball. This is the clearest current temporal-trace example because selected phrases repeatedly recover player, ball, court, catching, throwing, and jumping fragments across time.
- Secondary demo: HorseRiding. This is useful as a second temporal trace, but it has more object-label drift.

The demo should be described as weak recoverable temporal semantics from frozen activations, not robust video captioning.
