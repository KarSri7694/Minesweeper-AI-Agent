import argparse
import ast
import csv
import json
from pathlib import Path


def convert_output_value(value: str) -> str:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a dictionary string, got {type(parsed).__name__}")
    for key in ("x", "y"):
        if key in parsed and isinstance(parsed[key], str) and parsed[key].isdigit():
            parsed[key] = int(parsed[key])
    return json.dumps(parsed, ensure_ascii=True)


def convert_csv(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} is missing a header row")
        if "output" not in reader.fieldnames:
            raise ValueError(f"{input_path} does not contain an 'output' column")

        rows = []
        for row in reader:
            row["output"] = convert_output_value(row["output"])
            rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_json{input_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a CSV 'output' dictionary string to proper JSON."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        help="One or more CSV files to convert.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file instead of writing a *_json.csv file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for input_path in args.inputs:
        output_path = input_path if args.in_place else build_default_output_path(input_path)
        convert_csv(input_path, output_path)
        print(f"Converted {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
