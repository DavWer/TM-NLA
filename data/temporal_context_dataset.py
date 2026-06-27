from dataclasses import dataclass
from pathlib import Path
import hashlib

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.temporal_sampler import sample_temporal_windows


@dataclass
class CachedVideoTrajectory:
    video_id: str
    video_path: str
    timestamps: list[float]
    frames: list[Image.Image]
    hidden_states: torch.Tensor


def frame_digest(frame: Image.Image) -> str:
    return hashlib.sha1(frame.tobytes()).hexdigest()[:12]


def prepare_frame_inputs(processor, frame: Image.Image) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frame},
                {"type": "text", "text": "What do you see?"},
            ],
        }
    ]
    text = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(
        text=[text],
        images=[frame],
        padding=True,
        return_tensors="pt",
    )


def build_hidden_trajectory_cache(
    video_paths: list[str],
    processor,
    extractor,
    device: str,
    num_windows: int,
    target_size: tuple = (224, 224),
    skip_errors: bool = False,
) -> list[CachedVideoTrajectory]:
    trajectories = []
    for video_path in video_paths:
        try:
            windows = sample_temporal_windows(
                video_path,
                num_windows=num_windows,
                target_size=target_size,
            )
            hidden_states = []
            timestamps = []
            frames = []
            for timestamp, frame in windows:
                inputs = prepare_frame_inputs(processor, frame)
                inputs = {key: value.to(device) for key, value in inputs.items()}
                hidden = extractor.extract(inputs, pool_visual=True)[0].detach().cpu()
                hidden_states.append(hidden)
                timestamps.append(timestamp)
                frames.append(frame)
        except Exception as exc:
            if not skip_errors:
                raise
            print(f"warning: skipping unreadable video {video_path}: {exc}", flush=True)
            continue

        trajectories.append(
            CachedVideoTrajectory(
                video_id=Path(video_path).name,
                video_path=str(video_path),
                timestamps=timestamps,
                frames=frames,
                hidden_states=torch.stack(hidden_states),
            )
        )
    return trajectories


class TemporalContextDataset(Dataset):
    def __init__(self, trajectories: list[CachedVideoTrajectory]):
        self.trajectories = trajectories
        self.samples = []
        for trajectory_idx, trajectory in enumerate(trajectories):
            for window_idx in range(len(trajectory.timestamps)):
                self.samples.append((trajectory_idx, window_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        trajectory_idx, center_idx = self.samples[idx]
        trajectory = self.trajectories[trajectory_idx]
        last_idx = trajectory.hidden_states.size(0) - 1
        prev_idx = max(center_idx - 1, 0)
        next_idx = min(center_idx + 1, last_idx)
        context = torch.stack(
            [
                trajectory.hidden_states[prev_idx],
                trajectory.hidden_states[center_idx],
                trajectory.hidden_states[next_idx],
            ]
        )
        return {
            "context": context,
            "center_h": trajectory.hidden_states[center_idx],
            "video_id": trajectory.video_id,
            "video_path": trajectory.video_path,
            "timestamp": trajectory.timestamps[center_idx],
            "window_idx": center_idx,
            "frame_sha1": frame_digest(trajectory.frames[center_idx]),
        }
