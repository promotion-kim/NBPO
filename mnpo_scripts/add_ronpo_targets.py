import argparse
import logging
from pathlib import Path

from datasets import load_from_disk

from mnpo_scripts.precompute import add_ronpo_target, transform_chat_to_str


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add RONPO relative-label columns to an existing precomputed HF dataset."
    )
    parser.add_argument("--input_dir", required=True, help="Existing Dataset or DatasetDict saved with save_to_disk.")
    parser.add_argument("--output_dir", required=True, help="Destination directory for the augmented dataset.")
    parser.add_argument(
        "--ronpo_target_mode",
        default="score_diff_sign",
        choices=["score_diff_sign", "ordered"],
        help="Target construction. Use ordered when chosen/rejected are already strict win/loss pairs.",
    )
    parser.add_argument("--ronpo_target_column", default="ronpo_target")
    parser.add_argument("--ronpo_tie_threshold", type=float, default=0.0)
    parser.add_argument("--num_proc", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading dataset from %s", args.input_dir)
    dataset = load_from_disk(args.input_dir)

    logger.info("Normalizing chosen/rejected text fields when needed.")
    dataset = dataset.map(transform_chat_to_str, num_proc=args.num_proc)

    logger.info("Adding %s using mode=%s", args.ronpo_target_column, args.ronpo_target_mode)
    dataset = dataset.map(
        lambda ex: add_ronpo_target(
            ex,
            mode=args.ronpo_target_mode,
            target_column=args.ronpo_target_column,
            tie_threshold=args.ronpo_tie_threshold,
        ),
        num_proc=args.num_proc,
    )

    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving augmented dataset to %s", output_dir)
    dataset.save_to_disk(str(output_dir))
    logger.info("Done.")


if __name__ == "__main__":
    main()
