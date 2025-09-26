from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import logging
from .report import generate_report, validate_outputs, print_sample


def _cmd_gen(args: argparse.Namespace) -> int:
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    pack = Path(args.pack)
    out = Path(args.out)
    owners = Path(args.owners) if args.owners else None
    rc = generate_report(
        pack_dir=pack,
        out_dir=out,
        owners_yaml=owners,
        p95_ms=args.p95_ms,
        llm_on=(args.llm == "on"),
        model=args.model,
        embed_model=args.embed,
        api_base=args.api_base,
        api_key=os.environ.get("LLM_API_KEY"),
        llm_limit=args.limit_llm,
    )
    print(json.dumps(rc, indent=2, sort_keys=True))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    ok, msg = validate_outputs(Path(args.out))
    print(msg)
    return 0 if ok else 1


def _cmd_print_sample(args: argparse.Namespace) -> int:
    print_sample(Path(args.out))
    return 0


def _build() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="datagap-report", description="Data Gaps Report generator (1-week)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("gen", help="Generate a report from a pack")
    sp.add_argument("--pack", required=True, help="Path to synth pack directory")
    sp.add_argument("--out", required=True, help="Output directory for the report")
    sp.add_argument("--owners", dest="owners", help="Owners YAML file", default=None)
    sp.add_argument("--p95-ms", dest="p95_ms", type=int, default=5000, help="Slow threshold in ms")
    sp.add_argument("--llm", choices=["on", "off"], default="on")
    sp.add_argument("--model", default="gpt-5-mini")
    sp.add_argument("--embed", default="text-embedding-3-small")
    sp.add_argument("--api-base", dest="api_base", default="https://api.openai.com")
    sp.add_argument("--limit-llm", dest="limit_llm", type=int, default=0, help="Cap number of LLM calls (debug)")
    sp.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    sp.set_defaults(func=_cmd_gen)

    sp = sub.add_parser("validate", help="Validate report outputs")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=_cmd_validate)

    sp = sub.add_parser("print-sample", help="Print sample backlog items")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=_cmd_print_sample)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
