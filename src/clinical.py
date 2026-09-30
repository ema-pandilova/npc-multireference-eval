"""Clinical / morphological table loading, column profiling and ID matching.

The two CSVs (info_diagnoses.csv, morphological_parameters.csv) carry the
scanner metadata, staging, morphology and follow-up fields that the study
depends on. Their exact schema is not known in advance, so this module
*profiles* whatever arrives rather than hard-coding column names, and it
matches patient IDs defensively: imaging IDs are zero-padded strings like
"009" while CSV IDs are frequently integers like 9. Comparing those naively
yields a silent zero-overlap join, which is the single most damaging quiet
failure in this pipeline — so `match_ids` reports the overlap explicitly and
callers are expected to fail loudly on a bad match rate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Column-name patterns used to *suggest* the semantic role of each column.
# These are hints for the audit report, never silent renames.
ROLE_PATTERNS: dict[str, list[str]] = {
    "patient_id": [r"^id$", r"patient", r"case", r"subject", r"number", r"^no\.?$"],
    "scanner_manufacturer": [r"manufactur", r"vendor", r"\bmake\b"],
    "scanner_model": [r"model", r"scanner", r"device", r"equipment", r"machine"],
    "field_strength": [r"field.?strength", r"tesla", r"\b1\.5t\b", r"\b3t\b", r"magnet"],
    "pixel_spacing": [r"pixel.?spac", r"in.?plane", r"resolution", r"voxel"],
    "slice_thickness": [r"slice.?thick", r"thickness", r"slice.?gap", r"spacing.?between"],
    "sex": [r"^sex$", r"gender"],
    "age": [r"^age", r"years"],
    "t_stage": [r"^t.?stage", r"\bt_?category", r"^ct\b"],
    "n_stage": [r"^n.?stage", r"\bn_?category"],
    "m_stage": [r"^m.?stage", r"metasta"],
    "overall_stage": [r"stage", r"tnm", r"ajcc"],
    "survival_time": [r"pfs", r"progression.?free", r"follow.?up", r"time", r"month", r"day", r"os\b", r"survival"],
    "event": [r"event", r"status", r"progress", r"recur", r"relaps", r"death", r"censor"],
    "volume": [r"volume", r"\bvol\b"],
    "diameter": [r"diamet", r"length", r"width", r"extent", r"axis"],
}


@dataclass
class TableProfile:
    path: Path
    status: str
    n_rows: int = 0
    n_cols: int = 0
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    missing_frac: dict[str, float] = field(default_factory=dict)
    n_unique: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[Any]] = field(default_factory=dict)
    role_guess: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "OK"


def guess_role(column: str) -> str:
    """Best-guess semantic role for a column name, or '' when nothing matches."""
    col = column.strip().lower()
    for role, patterns in ROLE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, col):
                return role
    return ""


def _sniff_layout(path: Path, max_probe: int = 5) -> tuple[str, int]:
    """Detect the delimiter and how many leading rows to skip.

    Returns (separator, skiprows). The header row is taken to be the first line
    whose field count under the winning delimiter is maximal; anything above it is
    a decorative title row.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = [fh.readline().rstrip("\n") for _ in range(max_probe)]
    except OSError:
        return ",", 0
    lines = [ln for ln in lines if ln != ""]
    if not lines:
        return ",", 0

    # Pick the delimiter that splits the probed lines into the most fields.
    best_sep, best_count = ",", 1
    for sep in (";", ",", "\t", "|"):
        count = max((ln.count(sep) for ln in lines), default=0) + 1
        if count > best_count:
            best_sep, best_count = sep, count

    # The header is the first line that actually uses the full field width.
    widths = [ln.count(best_sep) + 1 for ln in lines]
    if not widths:
        return best_sep, 0
    target = max(widths)
    header_idx = next((i for i, w in enumerate(widths) if w == target), 0)

    # A line that is mostly empty fields is decorative, not a header, even if it
    # has the full width (e.g. "info_diagnoses;;;;;;;;;;").
    while header_idx < len(lines) - 1:
        fields = [f.strip() for f in lines[header_idx].split(best_sep)]
        if sum(1 for f in fields if f) > max(1, len(fields) // 3):
            break
        header_idx += 1
    return best_sep, header_idx


def load_table(path: Path) -> tuple[pd.DataFrame | None, TableProfile]:
    """Load and profile a CSV, distinguishing missing from empty from malformed."""
    if not path.exists():
        return None, TableProfile(path=path, status="MISSING", error="path does not exist")
    if path.stat().st_size == 0:
        return None, TableProfile(path=path, status="EMPTY_FILE", error="file is 0 bytes")

    df = None
    last_err = None
    # This dataset's CSVs are semicolon-delimited and carry a decorative first row
    # ("info_diagnoses;;;;" / ";ROI-T1;;;;"), so a bare read_csv yields a single
    # garbage column whose values still contain digits — which then "matches" the
    # imaging IDs and passes the join guard for the wrong reason. The separator and
    # header offset are therefore detected explicitly rather than assumed.
    # Chinese-institution clinical CSVs are also commonly gbk/gb18030, not utf-8.
    sep, skiprows = _sniff_layout(path)
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, sep=sep, skiprows=skiprows, engine="python")
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if df is None:
        return None, TableProfile(path=path, status="UNREADABLE", error=str(last_err))

    # Drop the all-empty padding columns those decorative rows leave behind.
    df = df.loc[:, [c for c in df.columns if not (str(c).startswith("Unnamed") and df[c].isna().all())]]
    df.columns = [str(c).strip() for c in df.columns]

    prof = TableProfile(
        path=path,
        status="OK",
        n_rows=len(df),
        n_cols=df.shape[1],
        columns=[str(c) for c in df.columns],
        dtypes={str(c): str(t) for c, t in df.dtypes.items()},
        missing_frac={str(c): float(df[c].isna().mean()) for c in df.columns},
        n_unique={str(c): int(df[c].nunique(dropna=True)) for c in df.columns},
        examples={
            str(c): [_scalar(v) for v in df[c].dropna().unique()[:5]] for c in df.columns
        },
        role_guess={str(c): guess_role(str(c)) for c in df.columns},
    )
    return df, prof


def _scalar(v: Any) -> Any:
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:  # noqa: BLE001
        pass
    return v


def normalise_id(value: Any, width: int = 3) -> str | None:
    """Coerce any ID representation to the zero-padded imaging convention.

    9, "9", "009", "9.0", "P009", "patient_9" all become "009".
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    digits = re.findall(r"\d+", s)
    if not digits:
        return None
    # Take the longest digit run; handles both "P009" and "case-2019-9".
    core = max(digits, key=len) if len(digits) > 1 else digits[0]
    return core.lstrip("0").zfill(width) if core.strip("0") else "0".zfill(width)


def find_id_column(df: pd.DataFrame, imaging_ids: set[str]) -> tuple[str | None, float]:
    """Pick the CSV column that best matches the imaging patient IDs.

    Rather than trusting a column *name*, this scores every column by the
    fraction of its normalised values that land in the imaging ID set. That
    survives arbitrary naming and non-English headers.
    """
    best_col, best_score = None, 0.0
    for col in df.columns:
        norm = df[col].map(lambda v: normalise_id(v))
        valid = norm.dropna()
        if valid.empty:
            continue
        score = float(valid.isin(imaging_ids).mean())
        # Prefer near-unique columns; an ID column should not repeat much.
        uniqueness = valid.nunique() / max(1, len(valid))
        score *= 0.5 + 0.5 * uniqueness
        if score > best_score:
            best_col, best_score = str(col), score
    return best_col, best_score


@dataclass
class MatchReport:
    id_column: str | None
    match_score: float
    n_csv_rows: int
    n_csv_ids: int
    n_imaging_ids: int
    n_matched: int
    only_in_csv: list[str]
    only_in_imaging: list[str]
    duplicated_csv_ids: list[str]

    @property
    def match_rate(self) -> float:
        return self.n_matched / self.n_imaging_ids if self.n_imaging_ids else 0.0


def match_ids(df: pd.DataFrame, imaging_ids: set[str]) -> MatchReport:
    """Match CSV rows to imaging patients and report the join explicitly."""
    id_col, score = find_id_column(df, imaging_ids)
    if id_col is None:
        return MatchReport(None, 0.0, len(df), 0, len(imaging_ids), 0, [], sorted(imaging_ids), [])

    norm = df[id_col].map(lambda v: normalise_id(v)).dropna()
    csv_ids = set(norm)
    dupes = sorted(norm[norm.duplicated()].unique().tolist())
    matched = csv_ids & imaging_ids

    return MatchReport(
        id_column=id_col,
        match_score=score,
        n_csv_rows=len(df),
        n_csv_ids=len(csv_ids),
        n_imaging_ids=len(imaging_ids),
        n_matched=len(matched),
        only_in_csv=sorted(csv_ids - imaging_ids)[:50],
        only_in_imaging=sorted(imaging_ids - csv_ids)[:50],
        duplicated_csv_ids=dupes[:50],
    )


def assert_match_quality(report: MatchReport, min_rate: float = 0.80) -> None:
    """Fail loudly when clinical IDs do not line up with imaging IDs.

    A silent zero-overlap join would produce a manifest full of NaN clinical
    fields that still looks superficially valid, so this is a hard error.
    """
    if report.id_column is None:
        raise ValueError(
            "No column in the clinical table matches the imaging patient IDs. "
            "The join cannot be performed; inspect the CSV headers manually."
        )
    if report.match_rate < min_rate:
        raise ValueError(
            f"Clinical/imaging ID match rate is {report.match_rate:.1%} "
            f"({report.n_matched}/{report.n_imaging_ids}) using column '{report.id_column}', "
            f"below the required {min_rate:.0%}. Refusing to build a manifest on a bad join.\n"
            f"  IDs only in imaging (first 10): {report.only_in_imaging[:10]}\n"
            f"  IDs only in CSV (first 10):     {report.only_in_csv[:10]}"
        )


def attach_clinical(
    manifest: pd.DataFrame,
    df: pd.DataFrame,
    report: MatchReport,
    prefix: str = "",
) -> pd.DataFrame:
    """Left-join a profiled clinical table onto the patient manifest by normalised ID."""
    if report.id_column is None:
        return manifest
    aux = df.copy()
    aux["_patient_id"] = aux[report.id_column].map(lambda v: normalise_id(v))
    aux = aux.dropna(subset=["_patient_id"]).drop_duplicates(subset=["_patient_id"], keep="first")
    aux = aux.drop(columns=[report.id_column])
    if prefix:
        aux = aux.rename(columns={c: f"{prefix}{c}" for c in aux.columns if c != "_patient_id"})
    return manifest.merge(aux, how="left", left_on="patient_id", right_on="_patient_id").drop(
        columns=["_patient_id"]
    )
