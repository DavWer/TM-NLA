import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from data.temporal_sampler import sample_temporal_windows
from pathlib import Path
from typing import Callable, Iterable, List
import csv
import re


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
UCF_GROUP_PATTERN = re.compile(r"_g(\d+)_c(\d+)")


def discover_video_paths(video_dir: str, recursive: bool = True) -> List[str]:
    root = Path(video_dir)
    pattern = "**/*" if recursive else "*"
    return sorted(
        str(path)
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def select_diverse_video_paths(
    video_root: str,
    metadata_csv: str,
    split: str = "train",
    labels: Iterable[str] | None = None,
    clips_per_label: int = 1,
    max_videos: int | None = 5,
) -> List[str]:
    allowed_labels = set(labels) if labels else None
    root = Path(video_root)
    split_dir = root / split
    selected: list[str] = []
    counts: dict[str, int] = {}

    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"video_name", "tag"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata CSV missing columns: {sorted(missing)}")

        for row in reader:
            label = row["tag"]
            if allowed_labels is not None and label not in allowed_labels:
                continue
            if counts.get(label, 0) >= clips_per_label:
                continue

            path = split_dir / row["video_name"]
            if not path.exists():
                path = root / row["video_name"]
            if not path.exists():
                continue

            selected.append(str(path))
            counts[label] = counts.get(label, 0) + 1
            if max_videos is not None and len(selected) >= max_videos:
                break

    return selected


def select_train_holdout_video_paths(
    video_root: str,
    metadata_csv: str,
    split: str = "train",
    labels: Iterable[str] | None = None,
    train_clips_per_label: int = 4,
    holdout_clip_number: int = 5,
) -> tuple[List[str], List[str]]:
    if holdout_clip_number < 1:
        raise ValueError("holdout_clip_number is 1-based and must be positive")

    allowed_labels = set(labels) if labels else None
    root = Path(video_root)
    split_dir = root / split
    paths_by_label: dict[str, list[str]] = {}

    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"video_name", "tag"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata CSV missing columns: {sorted(missing)}")

        for row in reader:
            label = row["tag"]
            if allowed_labels is not None and label not in allowed_labels:
                continue
            path = split_dir / row["video_name"]
            if not path.exists():
                path = root / row["video_name"]
            if path.exists():
                paths_by_label.setdefault(label, []).append(str(path))

    train_paths = []
    holdout_paths = []
    holdout_index = holdout_clip_number - 1
    for label, paths in paths_by_label.items():
        if len(paths) <= holdout_index:
            raise ValueError(
                f"Label '{label}' has {len(paths)} clips, but holdout clip "
                f"number {holdout_clip_number} was requested"
            )
        holdout_paths.append(paths[holdout_index])
        train_candidates = [
            path for idx, path in enumerate(paths)
            if idx != holdout_index
        ]
        if len(train_candidates) < train_clips_per_label:
            raise ValueError(
                f"Label '{label}' has only {len(train_candidates)} non-held-out clips"
            )
        train_paths.extend(train_candidates[:train_clips_per_label])

    return train_paths, holdout_paths


def select_train_holdout_group_video_paths(
    video_root: str,
    metadata_csv: str,
    split: str = "train",
    labels: Iterable[str] | None = None,
    train_clips_per_label: int = 4,
    holdout_group: str = "09",
    holdout_clips_per_label: int = 1,
) -> tuple[List[str], List[str]]:
    if holdout_clips_per_label < 1:
        raise ValueError("holdout_clips_per_label must be positive")
    allowed_labels = set(labels) if labels else None
    normalized_group = str(holdout_group).removeprefix("g").zfill(2)
    root = Path(video_root)
    split_dir = root / split
    train_by_label: dict[str, list[str]] = {}
    holdout_by_label: dict[str, list[str]] = {}

    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"video_name", "tag"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata CSV missing columns: {sorted(missing)}")

        for row in reader:
            label = row["tag"]
            if allowed_labels is not None and label not in allowed_labels:
                continue
            match = UCF_GROUP_PATTERN.search(row["video_name"])
            if match is None:
                continue

            path = split_dir / row["video_name"]
            if not path.exists():
                path = root / row["video_name"]
            if not path.exists():
                continue

            target = holdout_by_label if match.group(1) == normalized_group else train_by_label
            target.setdefault(label, []).append(str(path))

    train_paths = []
    holdout_paths = []
    for label in sorted(set(train_by_label) | set(holdout_by_label)):
        train_candidates = train_by_label.get(label, [])
        holdout_candidates = holdout_by_label.get(label, [])
        if len(train_candidates) < train_clips_per_label:
            raise ValueError(
                f"Label '{label}' has only {len(train_candidates)} training clips "
                f"outside held-out group g{normalized_group}"
            )
        if len(holdout_candidates) < holdout_clips_per_label:
            raise ValueError(
                f"Label '{label}' has only {len(holdout_candidates)} clips in "
                f"held-out group g{normalized_group}"
            )
        train_paths.extend(train_candidates[:train_clips_per_label])
        holdout_paths.extend(holdout_candidates[:holdout_clips_per_label])

    return train_paths, holdout_paths


class VideoNLADataset(Dataset):
    def __init__(
        self,
        video_paths: List[str],
        processor: Callable,
        num_windows: int = 3,
        target_size: tuple = (224, 224),
    ):
        self.video_paths = video_paths
        self.processor_fn = processor
        self.num_windows = num_windows
        self.target_size = target_size

        self.samples = []
        for vp in video_paths:
            windows = sample_temporal_windows(vp, num_windows, target_size)
            for ts, frame in windows:
                self.samples.append((vp, ts, frame))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, timestamp, frame = self.samples[idx]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": "What do you see?"},
                ],
            }
        ]
        text = self.processor_fn.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor_fn(
            text=[text],
            images=[frame],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return {
            "inputs": inputs,
            "video_path": video_path,
            "timestamp": timestamp,
            "metadata": {"video_path": video_path, "timestamp": timestamp},
        }


def collate_video_samples(batch):
    keys = batch[0]["inputs"].keys()
    collated = {}
    for key in keys:
        values = [b["inputs"][key] for b in batch]
        if all(v.shape == values[0].shape for v in values):
            collated[key] = torch.stack(values)
        elif key == "input_ids":
            collated[key] = pad_sequence(values, batch_first=True, padding_value=0)
        elif key == "attention_mask":
            collated[key] = pad_sequence(values, batch_first=True, padding_value=0)
        else:
            raise ValueError(
                f"Cannot collate variable-shaped processor field '{key}': "
                f"{[tuple(v.shape) for v in values]}"
            )
    return {
        "inputs": collated,
        "video_paths": [b["video_path"] for b in batch],
        "timestamps": [b["timestamp"] for b in batch],
    }
