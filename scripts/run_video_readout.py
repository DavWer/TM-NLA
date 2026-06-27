"""One-command wrapper: extract video trajectories, then print TM-NLA readouts."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def checkpoint_path(checkpoint_dir: Path, name: str) -> Path:
    """Resolve a release checkpoint used by the inference wrapper."""
    path = checkpoint_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}\n"
            "Download the TM-NLA checkpoints from Hugging Face and place them "
            "in the checkpoint directory."
        )
    return path


def write_manifest(path: Path, videos: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_name", "tag"])
        writer.writeheader()
        for video in videos:
            writer.writerow(
                {
                    "video_name": str(video),
                    "tag": video.stem,
                }
            )


def run_command(command: list[str], verbose: bool = False) -> None:
    if verbose:
        print("\n> " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run terminal TM-NLA semantic readouts on video files."
    )
    parser.add_argument(
        "--video",
        action="append",
        required=True,
        help="Path to a video file. Repeat this flag to process multiple videos.",
    )
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--work_dir", default="outputs/inference")
    parser.add_argument("--num_windows", type=int, default=9)
    parser.add_argument("--target_layer", type=int, default=20)
    parser.add_argument("--target_key", choices=["h_temporal", "center_h"], default="h_temporal")
    parser.add_argument("--candidates_per_window", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--qwen_precision", choices=["4bit", "fp16", "fp32"], default="fp16")
    parser.add_argument("--lm_forward_precision", choices=["model", "fp32"], default="fp32")
    parser.add_argument("--skip_bad_videos", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    verbose = args.verbose or args.debug

    videos = [Path(item).expanduser().resolve() for item in args.video]
    missing_videos = [str(path) for path in videos if not path.exists()]
    if missing_videos:
        raise FileNotFoundError("Missing video file(s): " + ", ".join(missing_videos))

    checkpoint_dir = (REPO_ROOT / args.checkpoint_dir).resolve()
    temporal_probe = checkpoint_path(checkpoint_dir, "temporal_probe.pt")
    activation_verbalizer = checkpoint_path(checkpoint_dir, "activation_verbalizer.pt")
    text_reconstructor = checkpoint_path(checkpoint_dir, "text_reconstructor.pt")

    work_dir = (REPO_ROOT / args.work_dir).resolve()
    manifest_path = work_dir / "videos_manifest.csv"
    trajectory_path = work_dir / "trajectories.pt"
    write_manifest(manifest_path, videos)

    extract_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "extract_trajectories.py"),
        "--video_dir",
        ".",
        "--metadata_csv",
        str(manifest_path),
        "--split",
        ".",
        "--checkpoint",
        str(temporal_probe),
        "--qwen_precision",
        args.qwen_precision,
        "--lm_forward_precision",
        args.lm_forward_precision,
        "--target_layer",
        str(args.target_layer),
        "--num_windows",
        str(args.num_windows),
        "--clips_per_label",
        str(len(videos)),
        "--max_videos",
        str(len(videos)),
        "--output",
        str(trajectory_path),
    ]
    if args.skip_bad_videos:
        extract_command.append("--skip_bad_videos")

    readout_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_readout.py"),
        "--archive",
        str(trajectory_path),
        "--av_checkpoint",
        str(activation_verbalizer),
        "--text_reconstructor_checkpoint",
        str(text_reconstructor),
        "--target_key",
        args.target_key,
        "--qwen_precision",
        args.qwen_precision,
        "--lm_forward_precision",
        args.lm_forward_precision,
        "--candidates_per_window",
        str(args.candidates_per_window),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--output_dir",
        str(work_dir / "readout"),
    ]
    if args.verbose:
        readout_command.append("--verbose")
    if args.debug:
        readout_command.append("--debug")

    run_command(extract_command, verbose=verbose)
    run_command(readout_command, verbose=verbose)


if __name__ == "__main__":
    main()
