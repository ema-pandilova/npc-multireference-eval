"""Configuration loading and programmatic verification of dataset paths.

The dataset roots live outside this repository and are read-only. Nothing in
this project may assume a path is correct: `load_config` resolves every root and
raises on a missing one, so a typo in paths.yaml fails loudly at startup instead
of quietly yielding an empty manifest downstream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paths.yaml"


class ConfigError(RuntimeError):
    """Raised when the configuration does not match what is on disk."""


@dataclass
class Config:
    raw: dict[str, Any]
    repo_root: Path
    images_root: Path
    primary_data_root: Path
    reliability_root: Path
    modalities: list[str]
    seed: int
    ignore_dir_names: set[str] = field(default_factory=set)
    ignore_file_names: set[str] = field(default_factory=set)
    ignore_file_prefixes: tuple[str, ...] = ()

    # -- output helpers -------------------------------------------------
    def out(self, key: str) -> Path:
        """Resolve a key from the `outputs:` block to an absolute path."""
        outputs = self.raw.get("outputs", {})
        if key not in outputs:
            raise ConfigError(f"outputs.{key} is not defined in paths.yaml")
        path = self.repo_root / outputs[key]
        parent = path if path.suffix == "" else path.parent
        parent.mkdir(parents=True, exist_ok=True)
        return path

    def primary_file(self, key: str) -> Path:
        files = self.raw.get("primary_data_files", {})
        if key not in files:
            raise ConfigError(f"primary_data_files.{key} is not defined in paths.yaml")
        return self.primary_data_root / files[key]

    # -- filesystem filtering -------------------------------------------
    def is_noise(self, name: str) -> bool:
        """True for OS metadata that must never be treated as data."""
        if name in self.ignore_file_names or name in self.ignore_dir_names:
            return True
        return any(name.startswith(p) for p in self.ignore_file_prefixes)

    def clean_listdir(self, path: Path) -> list[str]:
        """List a directory with all OS metadata filtered out, sorted."""
        if not path.is_dir():
            return []
        return sorted(n for n in os.listdir(path) if not self.is_noise(n))


def load_config(
    config_path: str | Path | None = None,
    verify: bool = True,
    allow_missing: bool = False,
) -> Config:
    """Load paths.yaml and verify every dataset root actually exists.

    Set `verify=False` only for unit tests that do not touch the filesystem.

    `allow_missing=True` downgrades absent roots from an error to a printed
    warning. That is for the genuine intermediate state where the dataset is
    still being staged and only some archives have been extracted; it must never
    be the default, because a silently missing root yields an empty manifest that
    still looks valid.
    """
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    repo_root = config_path.resolve().parent.parent
    ds = raw.get("dataset", {})
    required = ("images_root", "primary_data_root", "reliability_root")
    missing_keys = [k for k in required if k not in ds]
    if missing_keys:
        raise ConfigError(f"paths.yaml is missing dataset keys: {missing_keys}")

    legacy = raw.get("legacy", {}) or {}

    def resolve(key: str) -> Path:
        """Resolve a dataset root, falling back to a legacy path during staging.

        While the dataset is being re-staged, the new root may not exist yet while
        an older extraction of the same content still does. Falling back keeps the
        structural pipeline runnable, and the substitution is printed rather than
        applied silently.
        """
        primary = (repo_root / ds[key]).resolve()
        if primary.is_dir() or key not in legacy:
            return primary
        alt = (repo_root / legacy[key]).resolve()
        if alt.is_dir():
            print(f"NOTE: dataset.{key} not found at {primary}")
            print(f"      falling back to legacy.{key} -> {alt}")
            return alt
        return primary

    ignore = raw.get("ignore", {})
    cfg = Config(
        raw=raw,
        repo_root=repo_root,
        images_root=resolve("images_root"),
        primary_data_root=resolve("primary_data_root"),
        reliability_root=resolve("reliability_root"),
        modalities=list(raw.get("modalities", [])),
        seed=int(raw.get("seed", 0)),
        ignore_dir_names=set(ignore.get("dir_names", [])),
        ignore_file_names=set(ignore.get("file_names", [])),
        ignore_file_prefixes=tuple(ignore.get("file_prefixes", [])),
    )

    if verify:
        problems = []
        for key in required:
            path = resolve(key)
            if not path.is_dir():
                problems.append(f"  dataset.{key} -> {path}  (NOT A DIRECTORY)")
        if problems:
            if not allow_missing:
                raise ConfigError(
                    "Configured dataset roots do not exist:\n"
                    + "\n".join(problems)
                    + f"\n\nConfig file: {config_path}"
                    + "\nFix paths.yaml, or pass --config pointing at a corrected copy."
                    + "\nTo proceed anyway on a partially-staged dataset, pass --allow-missing-roots."
                )
            print("WARNING: dataset roots missing; proceeding with partial data:")
            for p in problems:
                print(p)
        if not cfg.modalities:
            raise ConfigError("paths.yaml defines no modalities")

    return cfg


def describe_roots(cfg: Config) -> str:
    """Human-readable confirmation of what the config actually resolved to."""
    lines = ["Resolved dataset roots (read-only):"]
    for label, path in (
        ("images_root     ", cfg.images_root),
        ("primary_data    ", cfg.primary_data_root),
        ("reliability_root", cfg.reliability_root),
    ):
        n = len(cfg.clean_listdir(path))
        lines.append(f"  {label} {path}  [{n} entries]")
    return "\n".join(lines)
