from PIL import Image
import av
from typing import List, Tuple


def _stream_duration_seconds(stream, container=None) -> float:
    if stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
        if duration > 0:
            return duration
    if stream.frames:
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        return float(stream.frames) / fps
    if container is not None and container.duration is not None:
        duration = float(container.duration / av.time_base)
        if duration > 0:
            return duration
    raise ValueError("Could not determine video duration")


def _frame_time_seconds(frame) -> float:
    if frame.pts is None or frame.time_base is None:
        return 0.0
    return float(frame.pts * frame.time_base)


def _resize_frame(frame, target_size: tuple) -> Image.Image:
    img = frame.to_image()
    return img.resize(target_size, Image.LANCZOS)


def _decode_frame_at_or_after(container, stream, timestamp: float, target_size: tuple):
    if stream.time_base is None:
        raise ValueError("Cannot seek video stream without time_base")
    seek_pts = int(timestamp / stream.time_base)
    container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
    fallback = None
    for frame in container.decode(stream):
        frame_time = _frame_time_seconds(frame)
        if fallback is None:
            fallback = frame
        if frame_time + 1e-3 >= timestamp:
            return _resize_frame(frame, target_size), frame_time
    if fallback is not None:
        return _resize_frame(fallback, target_size), _frame_time_seconds(fallback)
    raise ValueError(f"Could not decode frame at {timestamp:.3f}s")


def _sample_by_decoding_all(container, stream, num_windows: int, target_size: tuple):
    frames = []
    for index, frame in enumerate(container.decode(stream)):
        frame_time = _frame_time_seconds(frame)
        if frame_time == 0.0 and frame.pts is None:
            fps = float(stream.average_rate) if stream.average_rate else 30.0
            frame_time = index / fps
        frames.append((frame_time, frame))
    if not frames:
        raise ValueError("Could not decode any video frames")

    results = []
    total = len(frames)
    for i in range(num_windows):
        index = round((i + 1) * (total - 1) / (num_windows + 1))
        frame_time, frame = frames[min(max(index, 0), total - 1)]
        results.append((frame_time, _resize_frame(frame, target_size)))
    return results


def sample_temporal_windows(
    video_path: str,
    num_windows: int = 3,
    target_size: tuple = (224, 224),
) -> List[Tuple[float, Image.Image]]:
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        try:
            duration = _stream_duration_seconds(stream, container=container)
        except ValueError:
            return _sample_by_decoding_all(
                container,
                stream,
                num_windows,
                target_size,
            )
        timestamps = [
            duration * (i + 1) / (num_windows + 1)
            for i in range(num_windows)
        ]
        results = []
        for requested_ts in timestamps:
            try:
                img, actual_ts = _decode_frame_at_or_after(
                    container,
                    stream,
                    requested_ts,
                    target_size,
                )
            except ValueError:
                container.close()
                container = av.open(video_path)
                stream = container.streams.video[0]
                return _sample_by_decoding_all(
                    container,
                    stream,
                    num_windows,
                    target_size,
                )
            results.append((actual_ts, img))
        return results
    finally:
        container.close()


def extract_frame_at_timestamp(
    video_path: str,
    timestamp: float,
    target_size: tuple = (224, 224),
) -> Image.Image:
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        img, _ = _decode_frame_at_or_after(container, stream, timestamp, target_size)
        return img
    finally:
        container.close()
