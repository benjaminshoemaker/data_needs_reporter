from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import write_default_config
from .util import write_schemas_to_dir
from .generator import generate_pack
from .validate import validate_pack
from .listing import list_packs


def _cmd_init(args: argparse.Namespace) -> int:
    cwd = Path.cwd()
    cfg_path = cwd / "config.yaml"
    if cfg_path.exists() and not args.force:
        print(f"config.yaml already exists at {cfg_path}. Use --force to overwrite.", file=sys.stderr)
    else:
        write_default_config(cfg_path)
        print(f"Wrote {cfg_path}")
    schemas_dir = cwd / "schemas"
    schemas_dir.mkdir(parents=True, exist_ok=True)
    wrote = write_schemas_to_dir(schemas_dir)
    print(f"Wrote schemas to {schemas_dir} ({len(wrote)} files)")
    return 0


def _cmd_gen(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    cfg_path = Path(args.config)
    seed = args.seed
    result = generate_pack(cfg_path=cfg_path, out_dir=out_dir, seed=seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    pack_dir = Path(args.pack)
    ok, summary = validate_pack(pack_dir)
    print(summary)
    return 0 if ok else 1


def _cmd_list(args: argparse.Namespace) -> int:
    root = Path(args.root)
    lines = list_packs(root)
    for line in lines:
        print(line)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="datagap-synth", description="Synthetic telemetry pack generator")
    p.add_argument("--version", action="version", version=f"datagap-synth {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="Write starter config.yaml and schemas/")
    sp.add_argument("--force", action="store_true", help="Overwrite existing config.yaml if present")
    sp.set_defaults(func=_cmd_init)

    sp = sub.add_parser("gen", help="Generate a pack from config")
    sp.add_argument("--config", required=True, help="Path to config.yaml")
    sp.add_argument("--out", required=True, help="Output pack directory (e.g., packs/<pack_id>)")
    sp.add_argument("--seed", type=int, default=None, help="Deterministic seed for reproducible packs")
    sp.set_defaults(func=_cmd_gen)

    sp = sub.add_parser("validate", help="Validate a generated pack")
    sp.add_argument("--pack", required=True, help="Path to pack directory")
    sp.set_defaults(func=_cmd_validate)

    sp = sub.add_parser("list", help="List packs under a root directory")
    sp.add_argument("--root", required=True, help="Root directory containing packs")
    sp.set_defaults(func=_cmd_list)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
