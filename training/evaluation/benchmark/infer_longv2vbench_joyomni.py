"""Run cached video editing on LongV2VBench."""

from pathlib import Path

if __package__:
    from .common import build_parser, load_items, run_inference
else:
    from common import build_parser, load_items, run_inference


def parse_args(argv=None):
    parser = build_parser(__doc__, num_frames=1440)
    parser.add_argument("--metadata-path", type=Path, required=True, help="Benchmark CSV or JSONL metadata.")
    parser.add_argument("--segments", default="all", help="Comma-separated editing categories, or 'all'.")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    items = load_items(args.metadata_path, args.dataset_root, args.segments)
    run_inference(args, items)


if __name__ == "__main__":
    main()
