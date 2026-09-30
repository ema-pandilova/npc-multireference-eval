#!/usr/bin/env python3
"""Re-download the NPC dataset from Zenodo and verify it by MD5.

The extracted copy in `oncology-paper/` contains filenames but no bytes (see
data_audit.md), so the source archives must be fetched again. This script
downloads them, checks each against the MD5 published in the Zenodo REST API,
and extracts them into a NEW directory. It never writes to, renames or deletes
the existing dataset directories.

Record: https://zenodo.org/records/13131827   (CC BY 4.0)

  inner_reliability.zip     1.2 MB   rater-2 masks, 3D NIfTI, 30 patients
  images.zip                753 MB   2D TIFF slice/mask pairs (what we hold now)
  primary_data.zip          4.9 GB   DICOM series + ROI NIfTI + the three CSVs
                                     (info_diagnoses, morphological_parameters,
                                      meta_data) — the only source of scanner
                                     metadata, staging and follow-up

Usage:
    python3 scripts/00_fetch_dataset.py --list
    python3 scripts/00_fetch_dataset.py --which inner_reliability
    python3 scripts/00_fetch_dataset.py --which all --dest ../dataset_v2
    python3 scripts/00_fetch_dataset.py --which all --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ZENODO_API = "https://zenodo.org/api/records/13131827"

# Verified against the Zenodo REST API on 2026-07-27.
EXPECTED = {
    "inner_reliability.zip": ("8c50459a565df4ce1f6550998f25f2cc", 1_262_533),
    "images.zip": ("e9461cba21be1d088d4b82a15ecd4f3b", 753_084_647),
    "primary_data.zip": ("e149ceb587925ee66320bb6b03de32c7", 4_899_785_110),
}


def md5sum(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch_manifest() -> dict[str, tuple[str, int]]:
    """Query Zenodo for the current file list; fall back to the pinned values."""
    try:
        with urllib.request.urlopen(ZENODO_API, timeout=60) as resp:
            data = json.load(resp)
        out = {}
        for f in data.get("files", []):
            checksum = str(f.get("checksum", "")).replace("md5:", "")
            out[f["key"]] = (checksum, int(f.get("size", 0)))
        if out:
            return out
    except Exception as exc:  # noqa: BLE001
        print(f"  (Zenodo API unreachable: {exc}; using pinned checksums)")
    return EXPECTED


def download(name: str, dest: Path, expected_md5: str, expected_size: int) -> Path:
    url = f"https://zenodo.org/records/13131827/files/{name}?download=1"
    out = dest / name
    if out.exists() and out.stat().st_size == expected_size:
        print(f"  {name}: already present, verifying ...")
        if md5sum(out) == expected_md5:
            print("  MD5 OK — skipping download")
            return out
        print("  MD5 mismatch — re-downloading")

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {name} ({expected_size / 1e6:.1f} MB) ...")
    tmp = out.with_suffix(out.suffix + ".part")

    def hook(done: int, bs: int, total: int) -> None:
        if total > 0 and done % 200 == 0:
            pct = min(100.0, done * bs / total * 100)
            sys.stdout.write(f"\r    {pct:5.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    sys.stdout.write("\r    100.0%\n")
    tmp.rename(out)

    actual = md5sum(out)
    if actual != expected_md5:
        raise RuntimeError(f"{name}: MD5 mismatch (got {actual}, expected {expected_md5})")
    print("  MD5 OK")
    return out


def extract(archive: Path, dest: Path) -> int:
    """Extract, skipping OS metadata. Refuses to overwrite a populated target."""
    target = dest / archive.stem
    if target.exists() and any(target.iterdir()):
        print(f"  {target} already exists and is not empty — skipping extraction")
        return 0
    target.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            base = info.filename.split("/")[-1]
            if "__MACOSX" in info.filename or base.startswith("._") or base == ".DS_Store":
                continue
            z.extract(info, target)
            if not info.is_dir():
                n += 1
    print(f"  extracted {n} files to {target}")
    return n


def stage_local(repo: Path, local: Path, dest: Path, manifest: dict, no_extract: bool) -> int:
    """Verify hand-supplied archives against Zenodo MD5s, then extract the complete ones.

    Filenames are matched loosely, so `images (1).zip` is recognised as `images.zip`.
    A partially-transferred archive is reported with its completion percentage and
    skipped rather than extracted, since a truncated zip yields a silently
    incomplete dataset.
    """
    if not local.is_absolute():
        local = (repo / local).resolve()
    if not local.is_dir():
        print(f"ERROR: {local} is not a directory", file=sys.stderr)
        return 2

    print(f"\nScanning {local} for dataset archives\n")
    found = {}
    for path in sorted(local.glob("*.zip")):
        stem = path.stem.split("(")[0].strip().rstrip("_-").strip()
        key = f"{stem}.zip"
        if key in manifest:
            found.setdefault(key, []).append(path)

    if not found:
        print("No recognised dataset archives found. Expected one of: "
              + ", ".join(manifest))
        return 3

    ready, partial = [], []
    for key, paths in found.items():
        exp_md5, exp_size = manifest[key]
        # Prefer the largest candidate when duplicates exist.
        path = max(paths, key=lambda p: p.stat().st_size)
        size = path.stat().st_size
        pct = size / exp_size * 100 if exp_size else 0.0
        print(f"{path.name}")
        print(f"  size {size:,} / {exp_size:,} ({pct:.1f}%)")
        if size != exp_size:
            print("  INCOMPLETE — still transferring or truncated; not extracting")
            partial.append((key, path, pct))
            continue
        actual = md5sum(path)
        if actual == exp_md5:
            print("  MD5 OK — complete and verified")
            ready.append((key, path))
        else:
            print(f"  MD5 MISMATCH (got {actual})")
            print("  size is right but content differs; not extracting")
            partial.append((key, path, pct))

    print()
    if ready and not no_extract:
        for key, path in ready:
            print(f"extracting {path.name}")
            extract_named(path, dest, key)
        print()

    print(f"Verified and extracted : {len(ready)}  ({', '.join(k for k, _ in ready) or 'none'})")
    print(f"Incomplete / mismatched: {len(partial)} "
          f"({', '.join(f'{k} {p:.0f}%' for k, _, p in partial) or 'none'})")
    if partial:
        print("\nRe-run this command once the remaining transfers finish.")
    else:
        print(f"\nAll archives staged under {dest}.")
        print("Point configs/paths.yaml at the new roots, then run scripts/01_audit_dataset.py.")
    return 0


def extract_named(archive: Path, dest: Path, key: str) -> int:
    """Extract to a directory named after the canonical archive key.

    These archives wrap their contents in a top-level folder whose name repeats
    the archive name, which would otherwise produce `inner_reliability/
    inner_reliability/<id>`. That redundant level is stripped so every staged
    root has the same shape.
    """
    stem = Path(key).stem
    target = dest / stem
    if target.exists() and any(target.iterdir()):
        print(f"  {target} already populated — skipping")
        return 0
    target.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as z:
        members = [
            i for i in z.infolist()
            if "__MACOSX" not in i.filename
            and not i.filename.split("/")[-1].startswith("._")
            and i.filename.split("/")[-1] != ".DS_Store"
        ]
        tops = {m.filename.split("/")[0] for m in members if m.filename.strip("/")}
        strip = len(tops) == 1 and tops.pop() == stem

        n = 0
        for info in members:
            rel = info.filename
            if strip:
                parts = rel.split("/", 1)
                rel = parts[1] if len(parts) > 1 else ""
                if not rel:
                    continue
            out = target / rel
            if info.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    print(f"  extracted {n} files to {target}" + ("  (stripped redundant top level)" if strip else ""))
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", default="inner_reliability",
                    help="inner_reliability | images | primary_data | all")
    ap.add_argument("--dest", default="../dataset_v2",
                    help="download/extract target, relative to this repo (default ../dataset_v2)")
    ap.add_argument("--list", action="store_true", help="show the Zenodo manifest and exit")
    ap.add_argument("--verify-only", action="store_true", help="check existing archives, do not download")
    ap.add_argument("--no-extract", action="store_true", help="download only")
    ap.add_argument("--local-dir", default=None,
                    help="verify and extract archives already on disk (e.g. uploaded by hand) "
                         "instead of downloading; accepts partially-transferred files and reports "
                         "their completion percentage")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    dest = (repo / args.dest).resolve()

    print("=" * 78)
    print("ZENODO DATASET FETCH — record 13131827 (CC BY 4.0)")
    print("=" * 78)
    manifest = fetch_manifest()

    if args.list:
        print(f"\n{'file':28s} {'size':>16s}  md5")
        for k, (m, s) in manifest.items():
            print(f"{k:28s} {s:>16,d}  {m}")
        total = sum(s for _, s in manifest.values())
        print(f"\ntotal: {total / 1e9:.2f} GB")
        free = shutil.disk_usage(repo).free
        print(f"free on target filesystem: {free / 1e9:.1f} GB")
        return 0

    if args.local_dir:
        return stage_local(repo, Path(args.local_dir), dest, manifest, args.no_extract)

    wanted = list(manifest) if args.which == "all" else [f"{args.which}.zip"]
    unknown = [w for w in wanted if w not in manifest]
    if unknown:
        print(f"ERROR: unknown archive(s) {unknown}. Options: "
              f"{', '.join(k.replace('.zip', '') for k in manifest)}, all", file=sys.stderr)
        return 2

    need = sum(manifest[w][1] for w in wanted)
    free = shutil.disk_usage(repo).free
    print(f"\nTarget    : {dest}")
    print(f"Archives  : {', '.join(wanted)}")
    print(f"Download  : {need / 1e9:.2f} GB   (extraction needs roughly the same again)")
    print(f"Free space: {free / 1e9:.1f} GB")
    if free < need * 2.2:
        print("ERROR: insufficient free space for download plus extraction.", file=sys.stderr)
        return 3
    print()

    for name in wanted:
        exp_md5, exp_size = manifest[name]
        print(f"{name}")
        path = dest / name
        if args.verify_only:
            if not path.exists():
                print("  not present")
                continue
            ok = md5sum(path) == exp_md5
            print(f"  MD5 {'OK' if ok else 'MISMATCH'}")
            continue
        try:
            archive = download(name, dest, exp_md5, exp_size)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}", file=sys.stderr)
            return 4
        if not args.no_extract:
            extract(archive, dest)
        print()

    if not args.verify_only and not args.no_extract:
        print("Next: point configs/paths.yaml at the new roots, then re-run the audit.")
        print("  images_root:        \"" + f"{args.dest}/images/imgs\"")
        print("  primary_data_root:  \"" + f"{args.dest}/primary_data/data\"")
        print("  reliability_root:   \"" + f"{args.dest}/inner_reliability/inner_reliability\"")
        print("\n  python3 scripts/01_audit_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
