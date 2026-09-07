"""Read-only capability probing for a specific llama-bench binary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


_OPTION_NAMES = {
    "n_depth": "--n-depth",
    "cache_type_k": "--cache-type-k",
    "cache_type_v": "--cache-type-v",
    "n_cpu_moe": "--n-cpu-moe",
    "split_mode": "--split-mode",
    "device": "--device",
    "main_gpu": "--main-gpu",
    "tensor_split": "--tensor-split",
    "load_mode": "--load-mode",
    "flash_attn": "--flash-attn",
}


@dataclass(frozen=True)
class LlamaBenchCapabilities:
    binary: str
    returncode: int | None
    help_sha256: str
    flags: dict[str, bool]
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def probe_llama_bench(binary: Path, *, timeout_seconds: float = 10.0) -> LlamaBenchCapabilities:
    """Call ``--help`` once and report only capabilities evidenced by that output."""

    try:
        proc = subprocess.run(
            [str(binary), "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return _failed_probe(binary, "llama-bench binary was not found")
    except subprocess.TimeoutExpired:
        return _failed_probe(binary, "llama-bench --help timed out")

    help_text = f"{proc.stdout}\n{proc.stderr}"
    return LlamaBenchCapabilities(
        binary=str(binary),
        returncode=proc.returncode,
        help_sha256=hashlib.sha256(help_text.encode("utf-8")).hexdigest(),
        flags={name: _has_option(help_text, option) for name, option in _OPTION_NAMES.items()},
        error="" if proc.returncode == 0 else f"llama-bench --help exited with code {proc.returncode}",
    )


def _failed_probe(binary: Path, error: str) -> LlamaBenchCapabilities:
    return LlamaBenchCapabilities(
        binary=str(binary),
        returncode=None,
        help_sha256="",
        flags={name: False for name in _OPTION_NAMES},
        error=error,
    )


def _has_option(help_text: str, option: str) -> bool:
    """Match an option token, never a prefix of another long option."""

    return bool(re.search(rf"(?<![\\w-]){re.escape(option)}(?=$|[=\s,\\[])", help_text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe llama-bench capabilities from --help")
    parser.add_argument("--llama-bench", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    result = probe_llama_bench(args.llama_bench, timeout_seconds=args.timeout)
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if result.error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
