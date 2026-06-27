import argparse
import csv
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.temporal_context_dataset import (
    TemporalContextDataset,
    build_hidden_trajectory_cache,
)
from data.video_dataset import (
    select_diverse_video_paths,
    select_train_holdout_group_video_paths,
)
from extract.hidden_state_extractor import HiddenStateExtractor
from models.activation_verbalizer import build_activation_verbalizer
from models.temporal_contextualizer import TemporalContextualizer
from models.temporal_probe import TemporalProbe
from models.tm_nla import TMNLA
from utils.qwen_utils import cast_qwen_language_model, load_qwen


def infer_av_architecture(checkpoint: dict) -> str:
    metadata = checkpoint.get("metadata", {})
    if metadata.get("av_architecture"):
        return metadata["av_architecture"]
    state_dict = checkpoint["av_state_dict"]
    if "input_layer.weight" in state_dict:
        return "residual4"
    return "mlp2"


def load_metadata_labels(metadata_csv: str) -> dict[str, str]:
    labels = {}
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            video_name = row["video_name"]
            labels[video_name] = row["tag"]
            labels[Path(video_name).name] = row["tag"]
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="data/videos")
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["temporal", "center_only"], default="temporal")
    parser.add_argument("--target_layer", type=int, default=20)
    parser.add_argument("--prefix_len", type=int, default=8)
    parser.add_argument("--nla_len", type=int, default=8)
    parser.add_argument("--clips_per_label", type=int, default=1)
    parser.add_argument("--max_videos", type=int, default=5)
    parser.add_argument("--holdout_group", type=str, default=None)
    parser.add_argument("--holdout_clips_per_label", type=int, default=1)
    parser.add_argument("--num_windows", type=int, default=9)
    parser.add_argument("--output", type=str, default="outputs/trajectories.pt")
    parser.add_argument("--source_archive", type=str, default=None)
    parser.add_argument("--skip_bad_videos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_thumbnails", action="store_true")
    parser.add_argument("--thumbnail_dir", type=str, default="outputs/thumbnails")
    parser.add_argument(
        "--qwen_precision",
        type=str,
        choices=["4bit", "fp16", "fp32"],
        default="4bit",
    )
    parser.add_argument(
        "--lm_forward_precision",
        type=str,
        choices=["model", "fp32"],
        default="model",
        help=(
            "Precision for checkpoint projection after source activation "
            "extraction. Use fp32 for the released full-precision probe."
        ),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_qwen(device, precision=args.qwen_precision)
    trajectories = None
    source_records = None
    labels_by_name = {}
    if args.source_archive:
        source_archive = torch.load(args.source_archive, map_location="cpu", weights_only=False)
        source_records = source_archive["records"]
    else:
        metadata_csv = args.metadata_csv or os.path.join(args.video_dir, f"{args.split}.csv")
        labels_by_name = load_metadata_labels(metadata_csv)
        if args.holdout_group is not None:
            _, video_paths = select_train_holdout_group_video_paths(
                video_root=args.video_dir,
                metadata_csv=metadata_csv,
                split=args.split,
                train_clips_per_label=1,
                holdout_group=args.holdout_group,
                holdout_clips_per_label=args.holdout_clips_per_label,
            )
        else:
            video_paths = select_diverse_video_paths(
                video_root=args.video_dir,
                metadata_csv=metadata_csv,
                split=args.split,
                clips_per_label=args.clips_per_label,
                max_videos=args.max_videos,
            )

        extractor = HiddenStateExtractor(
            model,
            target_layer=args.target_layer,
            tokenizer=processor.tokenizer,
        )
        try:
            trajectories = build_hidden_trajectory_cache(
                video_paths=video_paths,
                processor=processor,
                extractor=extractor,
                device=device,
                num_windows=args.num_windows,
                skip_errors=args.skip_bad_videos,
            )
        finally:
            extractor.close()
        if not trajectories:
            raise RuntimeError("No trajectories were extracted; all selected videos failed or selection was empty")

    if args.lm_forward_precision == "fp32":
        cast_qwen_language_model(model, torch.float32)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    checkpoint_metadata = checkpoint.get("metadata", {})
    av = build_activation_verbalizer(
        architecture=infer_av_architecture(checkpoint),
        hidden_dim=1024,
        prefix_len=args.prefix_len,
    ).to(device)
    tm_nla = TMNLA(av, model, processor=processor, nla_len=args.nla_len).to(device)
    contextualizer = TemporalContextualizer(hidden_dim=1024, mode=args.mode).to(device)
    temporal_probe = TemporalProbe(contextualizer, tm_nla).to(device)
    temporal_probe.contextualizer.load_state_dict(checkpoint["contextualizer_state_dict"])
    temporal_probe.av.load_state_dict(checkpoint["av_state_dict"])
    temporal_probe.eval()

    dataset = source_records if source_records is not None else TemporalContextDataset(trajectories)
    records = []
    for sample in dataset:
        context = sample["context"].unsqueeze(0).to(device)
        with torch.no_grad():
            h_temporal = temporal_probe.contextualize(context)[0].detach().cpu()
            nla_emb = temporal_probe(context)[0].detach().cpu()

        thumbnail_path = sample.get("thumbnail_path")
        if args.save_thumbnails and trajectories is not None:
            trajectory = next(
                item for item in trajectories if item.video_id == sample["video_id"]
            )
            frame = trajectory.frames[sample["window_idx"]]
            output_dir = Path(args.thumbnail_dir) / Path(sample["video_id"]).stem
            output_dir.mkdir(parents=True, exist_ok=True)
            thumbnail_path = output_dir / f"{sample['window_idx']:03d}_{sample['timestamp']:.3f}.jpg"
            frame.save(thumbnail_path, quality=85)

        records.append(
            {
                "video_id": sample["video_id"],
                "video_path": sample["video_path"],
                "label": sample.get(
                    "label",
                    labels_by_name.get(sample["video_id"], "unknown"),
                ),
                "timestamp": sample["timestamp"],
                "window_idx": sample["window_idx"],
                "frame_sha1": sample["frame_sha1"],
                "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
                "center_h": sample["center_h"],
                "context": sample["context"],
                "h_temporal": h_temporal,
                "nla_embedding": nla_emb,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                "run_type": "trajectory_extraction",
                "mode": args.mode,
                "target_layer": args.target_layer,
                "num_windows": args.num_windows,
                "qwen_precision": args.qwen_precision,
                "lm_forward_precision": args.lm_forward_precision,
                "checkpoint": args.checkpoint,
                "checkpoint_metadata": checkpoint_metadata,
                "holdout_group": args.holdout_group,
                "source_archive": args.source_archive,
                "skip_bad_videos": args.skip_bad_videos,
            },
            "records": records,
        },
        output_path,
    )
    print(f"Saved {len(records)} trajectory records: {output_path}")


if __name__ == "__main__":
    main()
