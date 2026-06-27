import argparse
import csv
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.qwen_nla_text_autoencoder import (
    ActivationVerbalizer,
    QwenNLASettings,
    TextReconstructor,
    direction_cosine,
)
from utils.qwen_utils import cast_qwen_language_model, clean_nla_text, load_qwen


SPECIFIC_COMPATIBILITY_THRESHOLD = 0.85
SPECIFICITY_MARGIN_THRESHOLD = 0.03


def specificity_status(compatibility: float, specificity_margin: float) -> str:
    """Numerical activation-specificity label; phrase text is never inspected."""
    if specificity_margin < 0.0:
        return "non_specific"
    if (
        compatibility >= SPECIFIC_COMPATIBILITY_THRESHOLD
        and specificity_margin >= SPECIFICITY_MARGIN_THRESHOLD
    ):
        return "specific"
    return "weak"


def load_records(path: str) -> list[dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Archive/dataset missing records: {path}")


def settings_from_checkpoint(checkpoint: dict) -> QwenNLASettings:
    values = checkpoint.get("settings", {})
    allowed = QwenNLASettings.__dataclass_fields__.keys()
    return QwenNLASettings(**{key: values[key] for key in values if key in allowed})


def tensor_value(record: dict, key: str) -> torch.Tensor:
    if key not in record:
        raise KeyError(f"Record missing `{key}`; keys={sorted(record.keys())}")
    value = record[key]
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().reshape(-1)
    return torch.tensor(value, dtype=torch.float32).reshape(-1)


def window_key(record: dict) -> tuple[str, str, int]:
    return (
        str(record.get("source_archive", "")),
        str(record["video_id"]),
        int(record["window_idx"]),
    )


def dedupe_records_by_window(records: list[dict]) -> list[dict]:
    deduped = {}
    for record in records:
        deduped.setdefault(window_key(record), record)
    return list(deduped.values())


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def generate_candidate_readouts(
    records: list[dict],
    av_checkpoint: str,
    target_key: str,
    qwen_precision: str,
    lm_forward_precision: str,
    candidates_per_window: int,
    max_new_tokens: int,
    av_prompt_template: str | None = None,
) -> list[dict]:
    """Ask AV for a small set of activation-conditioned phrase proposals."""
    if candidates_per_window < 1:
        raise ValueError("candidates_per_window must be at least 1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(av_checkpoint, map_location=device, weights_only=False)
    settings = settings_from_checkpoint(checkpoint)
    if av_prompt_template:
        settings.av_prompt_template = av_prompt_template
    qwen, processor = load_qwen(device, precision=qwen_precision)
    if lm_forward_precision == "fp32":
        cast_qwen_language_model(qwen, torch.float32)
    av = ActivationVerbalizer(
        qwen,
        processor.tokenizer,
        settings=settings,
        train_backbone=False,
    ).to(device)
    av.adapter.load_state_dict(checkpoint["adapter_state_dict"])
    av.eval()

    rows = []
    for record_index, record in enumerate(records):
        activation = tensor_value(record, target_key)
        activation_cuda = activation.to(device)
        for candidate_index in range(candidates_per_window):
            text = clean_nla_text(
                av.generate(
                    activation_cuda,
                    max_new_tokens=max_new_tokens,
                    do_sample=candidate_index != 0,
                    temperature=0.6,
                    top_p=0.9,
                )
            )
            rows.append(
                make_candidate_row(
                    record,
                    record_index,
                    candidate_index,
                    text,
                    activation,
                )
            )

    del av, qwen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return rows


def make_candidate_row(
    record: dict,
    record_index: int,
    candidate_index: int,
    text: str,
    activation: torch.Tensor,
) -> dict:
    return {
        "record_index": record_index,
        "candidate_index": candidate_index,
        "video_id": record["video_id"],
        "label": record.get("label", "unknown"),
        "timestamp": float(record["timestamp"]),
        "window_idx": int(record["window_idx"]),
        "frame_sha1": record.get("frame_sha1", ""),
        "thumbnail_path": record.get("thumbnail_path", ""),
        "generated_text": text,
        "activation": activation,
    }


@torch.no_grad()
def score_candidates(
    candidate_rows: list[dict],
    ar_checkpoint: str,
    qwen_precision: str,
    lm_forward_precision: str,
) -> list[dict]:
    """Score phrase specificity by comparing target and control reconstruction."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(ar_checkpoint, map_location=device, weights_only=False)
    settings = settings_from_checkpoint(checkpoint)
    qwen, processor = load_qwen(device, precision=qwen_precision)
    if lm_forward_precision == "fp32":
        cast_qwen_language_model(qwen, torch.float32)
    ar = TextReconstructor(
        qwen,
        processor.tokenizer,
        settings=settings,
        train_backbone=False,
    ).to(device)
    ar.value_head.load_state_dict(checkpoint["value_head_state_dict"])
    ar.eval()

    unique_targets = {}
    for row in candidate_rows:
        unique_targets[int(row["record_index"])] = row["activation"].float()
    ordered_indices = sorted(unique_targets)
    target_stack = torch.stack([unique_targets[index] for index in ordered_indices]).to(device)
    shuffled_stack = torch.roll(target_stack, shifts=1, dims=0) if target_stack.size(0) > 1 else target_stack
    index_to_target_pos = {record_index: pos for pos, record_index in enumerate(ordered_indices)}

    scored = []
    for row in candidate_rows:
        pos = index_to_target_pos[int(row["record_index"])]
        target = target_stack[pos].unsqueeze(0)
        shuffled_target = shuffled_stack[pos].unsqueeze(0)
        predicted = ar.forward_text([row["generated_text"]])
        compatibility = direction_cosine(predicted, target).mean().item()
        control_compatibility = direction_cosine(predicted, shuffled_target).mean().item()
        output = {key: value for key, value in row.items() if key != "activation"}
        output["compatibility"] = compatibility
        output["control_compatibility"] = control_compatibility
        output["specificity_margin"] = compatibility - control_compatibility
        output["specificity_status"] = specificity_status(
            output["compatibility"],
            output["specificity_margin"],
        )
        scored.append(output)

    del ar, qwen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return scored


def select_readouts(
    scored_rows: list[dict],
) -> list[dict]:
    by_record: dict[int, list[dict]] = {}
    for row in scored_rows:
        by_record.setdefault(int(row["record_index"]), []).append(row)

    selected = []
    for record_index, rows in sorted(by_record.items()):
        best = max(rows, key=lambda row: float(row["specificity_margin"]))
        selected.append(
            {
                **best,
                "selected_text": best["generated_text"],
            }
        )
    return selected


def selected_readout_rows(selected: list[dict]) -> list[dict]:
    rows = []
    for row in selected:
        rows.append(
            {
                "timestamp": f"{float(row['timestamp']):.6f}",
                "generated_phrase": row["selected_text"],
                "compatibility": f"{float(row['compatibility']):.6f}",
                "control_compatibility": f"{float(row['control_compatibility']):.6f}",
                "specificity_margin": f"{float(row['specificity_margin']):.6f}",
                "specificity_status": row["specificity_status"],
            }
        )
    return rows


def selected_timeline_rows(selected: list[dict]) -> list[dict]:
    rows = []
    grouped: dict[str, list[dict]] = {}
    for row in selected:
        grouped.setdefault(str(row.get("label", "unknown")), []).append(row)
    for label, items in grouped.items():
        for row in sorted(items, key=lambda item: float(item["timestamp"])):
            rows.append(
                {
                    "timestamp": f"{float(row['timestamp']):.6f}",
                    "generated_phrase": row["selected_text"],
                    "compatibility": f"{float(row['compatibility']):.6f}",
                    "control_compatibility": f"{float(row['control_compatibility']):.6f}",
                    "specificity_margin": f"{float(row['specificity_margin']):.6f}",
                    "specificity_status": row["specificity_status"],
                }
            )
    return rows


def candidate_diagnostic_rows(scored_rows: list[dict]) -> list[dict]:
    rows = []
    for row in scored_rows:
        rows.append(
            {
                "record_index": int(row["record_index"]),
                "candidate_index": int(row["candidate_index"]),
                "video_id": row["video_id"],
                "label": row.get("label", "unknown"),
                "timestamp": f"{float(row['timestamp']):.6f}",
                "window_idx": int(row["window_idx"]),
                "generated_phrase": row["generated_text"],
                "compatibility": f"{float(row['compatibility']):.6f}",
                "control_compatibility": f"{float(row['control_compatibility']):.6f}",
                "specificity_margin": f"{float(row['specificity_margin']):.6f}",
                "specificity_status": row["specificity_status"],
            }
        )
    return rows


def print_summary(
    selected: list[dict],
    candidates_per_window: int,
    output_dir: Path | None = None,
) -> None:
    compatibilities = [float(row["compatibility"]) for row in selected]
    controls = [float(row["control_compatibility"]) for row in selected]
    specificity_margins = [float(row["specificity_margin"]) for row in selected]
    print("\n=== SELECTED SEMANTIC READOUT ===")
    print(f"windows:               {len(selected)}")
    print(f"candidates/window:     {candidates_per_window}")
    print(f"mean compatibility:         {float(np.mean(compatibilities)) if compatibilities else 0.0:.4f}")
    print(f"mean control compatibility: {float(np.mean(controls)) if controls else 0.0:.4f}")
    print(f"mean specificity margin:    {float(np.mean(specificity_margins)) if specificity_margins else 0.0:+.4f}")
    if output_dir is not None:
        print(f"saved:                 {output_dir}")


def print_timelines(selected: list[dict]) -> None:
    by_label: dict[str, list[dict]] = {}
    for row in selected:
        by_label.setdefault(str(row["label"]), []).append(row)
    print("\n=== SEMANTIC TIMELINES ===")
    for label, rows in by_label.items():
        rows = sorted(rows, key=lambda row: float(row["timestamp"]))
        print(f"\n{label}")
        for row in rows:
            print(
                f"  timestamp={float(row['timestamp']):.2f}s | "
                f"generated_phrase={row['selected_text']!r} | "
                f"compatibility={float(row['compatibility']):+.3f} | "
                f"control_compatibility={float(row['control_compatibility']):+.3f} | "
                f"specificity_margin={float(row['specificity_margin']):+.3f} | "
                f"specificity_status={row['specificity_status']}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="append", required=True)
    parser.add_argument("--av_checkpoint", type=str, required=True)
    parser.add_argument("--text_reconstructor_checkpoint", dest="ar_checkpoint", type=str)
    parser.add_argument("--ar_checkpoint", dest="ar_checkpoint", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--target_key", choices=["h_temporal", "center_h"], default="h_temporal")
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--candidates_per_window", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--qwen_precision", choices=["4bit", "fp16", "fp32"], default="fp16")
    parser.add_argument("--lm_forward_precision", choices=["model", "fp32"], default="fp32")
    parser.add_argument(
        "--av_prompt_template",
        type=str,
        default="",
        help="Optional AV prompt override. Must contain {injection_token}.",
    )
    parser.add_argument("--dedupe_windows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", type=str, default="outputs/readout")
    parser.add_argument(
        "--write_candidates",
        action="store_true",
        help="Write candidate-level diagnostics in addition to selected readouts.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.ar_checkpoint is None:
        parser.error("--text_reconstructor_checkpoint is required")
    verbose = args.verbose or args.debug

    records = []
    for archive_path in args.archive:
        archive_records = load_records(archive_path)
        for record in archive_records:
            records.append({**record, "source_archive": archive_path})
    if args.dedupe_windows:
        records = dedupe_records_by_window(records)
    records = sorted(records, key=lambda row: (str(row.get("label", "")), str(row["video_id"]), float(row["timestamp"])))
    if args.max_records > 0:
        records = records[: args.max_records]
    if not records:
        raise RuntimeError("No records to read out")
    if args.candidates_per_window > 16:
        print(
            "warning: candidates_per_window is above the intended release range "
            "(16). Report this value when discussing results.",
            flush=True,
        )

    print("=== TM-NLA SEMANTIC READOUT ===")
    print(f"windows: {len(records)}")
    if verbose:
        print(f"AV:      {args.av_checkpoint}")
        print(f"Text reconstructor: {args.ar_checkpoint}")

    candidate_rows = generate_candidate_readouts(
        records=records,
        av_checkpoint=args.av_checkpoint,
        target_key=args.target_key,
        qwen_precision=args.qwen_precision,
        lm_forward_precision=args.lm_forward_precision,
        candidates_per_window=args.candidates_per_window,
        max_new_tokens=args.max_new_tokens,
        av_prompt_template=args.av_prompt_template or None,
    )
    scored_rows = score_candidates(
        candidate_rows,
        ar_checkpoint=args.ar_checkpoint,
        qwen_precision=args.qwen_precision,
        lm_forward_precision=args.lm_forward_precision,
    )
    selected = select_readouts(
        scored_rows,
    )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "selected_readout_windows.csv", selected_readout_rows(selected))
    write_csv(output_dir / "selected_timeline.csv", selected_timeline_rows(selected))
    if args.write_candidates or args.debug:
        write_csv(output_dir / "candidate_diagnostics.csv", candidate_diagnostic_rows(scored_rows))

    print_summary(selected, candidates_per_window=args.candidates_per_window, output_dir=output_dir)
    print_timelines(selected)


if __name__ == "__main__":
    main()
