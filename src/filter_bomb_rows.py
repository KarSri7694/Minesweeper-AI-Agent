from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def board_has_revealed_bomb(board_state: str) -> bool:
    board = json.loads(board_state)
    return any(cell == "B" for row in board for cell in row)


def filter_csv(input_path: str, output_path: str | None = None) -> Path:
    input_file = Path(input_path)
    if output_path is None:
        output_file = input_file.with_name(f"{input_file.stem}_filtered{input_file.suffix}")
    else:
        output_file = Path(output_path)

    with input_file.open("r", newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Input CSV has no header.")

        rows_to_keep = []
        for row in reader:
            before_has_bomb = board_has_revealed_bomb(row["board_state_before"])
            after_has_bomb = board_has_revealed_bomb(row["board_state_after"])
            if before_has_bomb or after_has_bomb:
                continue
            rows_to_keep.append(row)

    with output_file.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_keep)

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove rows containing revealed bombs from a generated Minesweeper CSV."
    )
    parser.add_argument("input_csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_file = filter_csv(args.input_csv, args.output)
    print(f"Saved filtered CSV to {output_file.resolve()}")


if __name__ == "__main__":
    main()
