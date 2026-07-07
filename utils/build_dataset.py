import csv
import json
import os
from pathlib import Path

def convert_json_to_csv(json_file, csv_file):
    data = []
    print(f"Converting {json_file} to {csv_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(obj)

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['input', 'output', 'rows', 'columns', 'max_mines'])
        for item in data:
            writer.writerow([item['compact_board_before'], item['action'], item['metadata'].get('board_size',{})[0], item['metadata'].get('board_size',{})[1], item['metadata'].get('mine_count',{})])
    print(f"Conversion completed. CSV file saved at {csv_file}")

    
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    dataset_dir = parent_dir / 'dataset'
    json_file = dataset_dir / 'balanced_9x9_2000_test.jsonl'
    
    csv_file = dataset_dir / 'dataset_test_9x9.csv'
    convert_json_to_csv(json_file, csv_file)