import json
import sys
from pathlib import Path
import yaml
import argparse

from rules.engine import HighlightExtractor, rank_and_finalize_highlights

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_prediction_data(file_path: Path) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_highlights(prediction_data: dict, config: dict) -> list:

    extractor = HighlightExtractor(prediction_data, config=config)
    all_highlights = extractor.run()
    # Rank highlights
    top_highlights = rank_and_finalize_highlights(all_highlights, config=config)
    print(f"[INFO] Ranked {len(top_highlights)} highlights.")
    
    return top_highlights

def save_top_highlights(top_highlights: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(top_highlights, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Top highlights saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract and save top highlights from prediction data.")
    parser.add_argument(
        "--predictions-json-path",
        type=Path
    )
    parser.add_argument(
        "--output-dir",
        type=Path
    )
    parser.add_argument(
        "--config",
        type=Path,
        default='rules/config.yaml'
    )
    args = parser.parse_args()

    rules_config = load_config(args.config)

    predictions_path = args.predictions_json_path or rules_config.get('predictions_json_path')
    output_dir = args.output_dir or rules_config.get('output_dir')

    predictions_path = Path(predictions_path)
    output_dir = Path(output_dir)

    prediction_data = load_prediction_data(predictions_path)
    
    top_highlights = process_highlights(prediction_data, rules_config)
    
    video_basename = Path(prediction_data.get('UrlLocal', predictions_path.stem)).stem
    output_file = output_dir / f"{video_basename}_highlights.json"
    
    save_top_highlights(top_highlights, output_file)

if __name__ == '__main__':
    main() 
