import argparse
import logging
import random
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

def create_directory_structure(dest_dir: Path, region: str, splits: list[str]) -> None:
    """Creates the required parallel 'images' and 'labels' directory structure for YOLOv8.

    Args:
        dest_dir (Path): The root directory for the target dataset.
        region (str): The specific region name (e.g., 'eu', 'us', 'br').
        splits (list[str]): List of split names (e.g., ['train', 'val', 'test']).
    """
    for split in splits:
        (dest_dir / region / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dest_dir / region / 'labels' / split).mkdir(parents=True, exist_ok=True)

def process_and_split_region(
    region_path: Path, 
    dest_path: Path, 
    region_name: str, 
    split_ratios: tuple[float, float, float],
    base_seed: int
) -> dict[str, int]:
    """Validates, shuffles, and copies region data into stratified target splits.

    This function ensures that image-label pairs remain intact and prevents 
    data leakage by purging the target directory before copying.

    Args:
        region_path (Path): Path to the source region directory.
        dest_path (Path): Path to the target root directory.
        region_name (str): Name of the region being processed.
        split_ratios (tuple[float, float, float]): Ratios for (train, val, test).
        base_seed (int): Base random seed for reproducibility.

    Returns:
        dict[str, int]: Statistics mapping each split to the number of processed files.
    """
    region_dest = dest_path / region_name
    if region_dest.exists():
        logger.info(f"Purging old data: {region_dest}")
        shutil.rmtree(region_dest)

    valid_extensions = {'.jpg', '.jpeg', '.png'}
    images = sorted([f for f in region_path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions])
    
    valid_pairs: list[tuple[Path, Path]] = []
    for img in images:
        txt_file = img.with_suffix('.txt')
        if txt_file.exists():
            valid_pairs.append((img, txt_file))
        else:
            logger.warning(f"Missing label detected and skipped: {img.name}")
            
    total_valid = len(valid_pairs)
    if total_valid == 0:
        logger.warning(f"No valid data found for region '{region_name}'.")
        return {'train': 0, 'val': 0, 'test': 0}

    create_directory_structure(dest_path, region_name, ['train', 'val', 'test'])

    rng = random.Random(f"{base_seed}_{region_name}")
    rng.shuffle(valid_pairs)
    
    train_end = int(total_valid * split_ratios[0])
    val_end = train_end + int(total_valid * split_ratios[1])
    
    splits_data = {
        'train': valid_pairs[:train_end],
        'val': valid_pairs[train_end:val_end],
        'test': valid_pairs[val_end:]
    }
    
    for split_name, pairs in splits_data.items():
        for img_path, txt_path in pairs:
            shutil.copy2(img_path, region_dest / 'images' / split_name / img_path.name)
            shutil.copy2(txt_path, region_dest / 'labels' / split_name / txt_path.name)
            
    return {k: len(v) for k, v in splits_data.items()}

def main() -> None:
    """Main execution flow for splitting the OpenALPR dataset into YOLOv8 format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    parser = argparse.ArgumentParser(description="Splits OpenALPR dataset into YOLOv8 format.")
    parser.add_argument("--source", type=str, required=True, help="Source directory containing raw data.")
    parser.add_argument("--dest", type=str, required=True, help="Target directory for the YOLO dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()

    source_dir = Path(args.source)
    dest_dir = Path(args.dest)
    split_ratios = (0.7, 0.1, 0.2)
    regions = ['eu', 'us', 'br']

    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return

    logger.info("Starting dataset organization and splitting process...")
    
    final_stats = {}

    for region in regions:
        region_path = source_dir / region
        if region_path.exists() and region_path.is_dir():
            stats = process_and_split_region(region_path, dest_dir, region, split_ratios, args.seed)
            final_stats[region] = stats
        else:
            logger.warning(f"Region folder skipped (not found): {region_path}")

    logger.info("--- PROCESS SUMMARY ---")
    for reg, stats in final_stats.items():
        if sum(stats.values()) > 0:
            logger.info(f"[{reg.upper()}] Total: {sum(stats.values())} -> Train: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}")
    logger.info("Dataset is ready for YOLOv8.")

if __name__ == "__main__":
    main()