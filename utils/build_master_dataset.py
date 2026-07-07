import argparse
import json
import os
import random
from typing import Dict, List, Set, Tuple


def derive_summary_filename(dataset_filename: str) -> str:
    dataset_stem, dataset_ext = os.path.splitext(dataset_filename)
    if not dataset_ext:
        dataset_ext = ".jsonl"
    return f"{dataset_stem}_summary{dataset_ext}"


def find_dataset_pairs(dataset_dir: str, master_filename: str) -> List[Tuple[str, str]]:
    master_summary_filename = derive_summary_filename(master_filename)
    pairs: List[Tuple[str, str]] = []

    for entry in sorted(os.listdir(dataset_dir)):
        if not entry.endswith(".jsonl"):
            continue
        if entry.endswith("_summary.jsonl"):
            continue
        if entry == master_filename:
            continue

        summary_name = derive_summary_filename(entry)
        if summary_name == master_summary_filename:
            continue

        dataset_path = os.path.join(dataset_dir, entry)
        summary_path = os.path.join(dataset_dir, summary_name)
        if os.path.isfile(dataset_path) and os.path.isfile(summary_path):
            pairs.append((dataset_path, summary_path))

    return pairs


def build_master_dataset(
    dataset_dir: str, master_filename: str, seed: int | None = None
) -> Dict[str, int]:
    master_summary_filename = derive_summary_filename(master_filename)
    master_dataset_path = os.path.join(dataset_dir, master_filename)
    master_summary_path = os.path.join(dataset_dir, master_summary_filename)

    files_processed = 0
    included_games = 0
    written_transitions = 0
    rng = random.Random(seed)
    master_summary_rows: List[dict] = []
    master_dataset_rows: List[dict] = []

    for dataset_path, summary_path in find_dataset_pairs(dataset_dir, master_filename):
        winning_game_ids: Set[int] = set()

        with open(summary_path, "r", encoding="utf-8") as summary_file:
            for line in summary_file:
                if not line.strip():
                    continue
                summary_record = json.loads(line)
                if summary_record.get("result") != "win":
                    continue
                game_id = summary_record.get("game_id")
                winning_game_ids.add(game_id)
                master_summary_rows.append(summary_record)
                included_games += 1

        if winning_game_ids:
            with open(dataset_path, "r", encoding="utf-8") as dataset_file:
                for line in dataset_file:
                    if not line.strip():
                        continue
                    transition_record = json.loads(line)
                    game_id = transition_record.get("metadata", {}).get("game_id")
                    
                    if game_id in winning_game_ids:
                        master_dataset_rows.append(transition_record)
                        written_transitions += 1

        files_processed += 1

    rng.shuffle(master_summary_rows)
    rng.shuffle(master_dataset_rows)

    with open(master_dataset_path, "w", encoding="utf-8") as master_dataset:
        for transition_record in master_dataset_rows:
            master_dataset.write(json.dumps(transition_record) + "\n")

    with open(master_summary_path, "w", encoding="utf-8") as master_summary:
        for summary_record in master_summary_rows:
            master_summary.write(json.dumps(summary_record) + "\n")

    return {
        "files_processed": files_processed,
        "included_games": included_games,
        "written_transitions": written_transitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a master Minesweeper dataset by keeping only winning games."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./dataset",
        help="Directory containing dataset JSONL files and paired summary JSONL files.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="master_dataset.jsonl",
        help="Filename for the combined dataset output. The summary file uses the same base name with _summary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for deterministic row shuffling.",
    )
    args = parser.parse_args()

    stats = build_master_dataset(args.dataset_dir, args.output_filename, args.seed)
    output_summary = derive_summary_filename(args.output_filename)

    print(f"Master dataset written to: {os.path.join(args.dataset_dir, args.output_filename)}")
    print(f"Master summary written to: {os.path.join(args.dataset_dir, output_summary)}")
    print(f"Dataset/summary pairs processed: {stats['files_processed']}")
    print(f"Winning games included: {stats['included_games']}")
    print(f"Transitions written: {stats['written_transitions']}")


if __name__ == "__main__":
    main()
