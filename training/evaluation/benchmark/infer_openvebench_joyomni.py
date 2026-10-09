"""Run cached video editing on OpenVE-Bench."""

from pathlib import Path

if __package__:
    from .common import build_parser, load_items, read_metadata, run_inference
else:
    from common import build_parser, load_items, read_metadata, run_inference


def parse_args(argv=None):
    parser = build_parser(__doc__, num_frames=81)
    parser.add_argument("--csv-path", type=Path, default=None, help="Defaults to DATASET_ROOT/benchmark_videos.csv.")
    parser.add_argument("--task-types", default="all", help="Comma-separated editing categories, or 'all'.")
    parser.add_argument("--pe-csv-path", type=Path, default=None, help="Optional prompt overrides matched by original_video.")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    metadata_path = args.csv_path or args.dataset_root / "benchmark_videos.csv"
    overrides = None
    if args.pe_csv_path is not None:
        overrides = {row["original_video"].strip(): row["prompt"].strip() for row in read_metadata(args.pe_csv_path)}
    items = load_items(metadata_path, args.dataset_root, args.task_types, overrides)
    run_inference(args, items)


if __name__ == "__main__":
    main()
