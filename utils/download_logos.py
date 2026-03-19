"""
Download ESPN team logos and map them to Kaggle team IDs.

Matching priority:
  1. Exact match on normalized shortDisplayName against MTeamSpellings.csv
  2. Exact match on normalized displayName against MTeamSpellings.csv
  3. Fuzzy match (SequenceMatcher) on shortDisplayName against all Kaggle team names

Usage:
    poetry run python utils/download_logos.py           # download all missing
    poetry run python utils/download_logos.py --limit 10  # test with 10 teams
    poetry run python utils/download_logos.py --gender W  # women's only
    poetry run python utils/download_logos.py --dry-run   # show matches without downloading
    poetry run python utils/download_logos.py --report    # show what's missing, no download

Logos saved to:
    data/logos/M/{kaggle_team_id}.png
    data/logos/W/{kaggle_team_id}.png

The script is idempotent — skips teams whose logo already exists.
"""

import argparse
import os
import re
import time
import urllib.request
from difflib import SequenceMatcher

import pandas as pd


# Manual overrides for ESPN names that don't match via spellings or fuzzy.
# Key = ESPN shortDisplayName, Value = Kaggle TeamName.
_MANUAL_OVERRIDES = {
    "UT Rio Grande": "UTRGV",
    "UT Rio Grande Valley": "UTRGV",
    "FIU": "Florida Intl",
    "SIU Edwardsville": "SIUE",
    "Purdue Fort Wayne": "PFW",
    "IUPUI": "IUPUI",
    "LIU": "LIU Brooklyn",
    "St. Francis (NY)": "St Francis NY",
    "TAM-Corpus Chris": "TAM C. Christi",
    "Lindenwood": "Lindenwood",
    "Queens (NC)": "Queens NC",
    "West Georgia": "West Georgia",
    "Mercyhurst": "Mercyhurst",
    "Northern Iowa": "Northern Iowa",
}

ESPN_API_M = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/teams?limit=700"
)
ESPN_API_W = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "womens-college-basketball/teams?limit=700"
)

KAGGLE_DATA_DIR = "data/2026"
LOGO_DIR = "data/logos"


# ── name normalization ────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ── spelling lookup ───────────────────────────────────────────────────────────

def _build_spelling_lookup(gender: str) -> dict[str, int]:
    """Build normalized_spelling → TeamID from {gender}TeamSpellings.csv."""
    spellings_file = os.path.join(KAGGLE_DATA_DIR, f"{gender}TeamSpellings.csv")
    if not os.path.exists(spellings_file):
        return {}
    df = pd.read_csv(spellings_file)
    lookup = {}
    for _, row in df.iterrows():
        key = _normalize(str(row["TeamNameSpelling"]))
        lookup[key] = int(row["TeamID"])
    return lookup


# ── fuzzy match ───────────────────────────────────────────────────────────────

def _fuzzy_match(name: str, candidates: list[str], threshold: float = 0.6) -> str | None:
    """Return the closest match from candidates, or None if below threshold."""
    name_norm = _normalize(name)
    best_score = 0.0
    best_match = None
    for cand in candidates:
        score = SequenceMatcher(None, name_norm, _normalize(cand)).ratio()
        if score > best_score:
            best_score = score
            best_match = cand
    return best_match if best_score >= threshold else None


# ── ESPN API ──────────────────────────────────────────────────────────────────

def fetch_espn_teams(api_url: str) -> list[dict]:
    """Fetch team list from ESPN API.

    Returns list of dicts with keys: display_name, short_name, logo_url.
    """
    import json

    try:
        with urllib.request.urlopen(api_url, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  ESPN API error: {e}")
        return []

    teams = []
    for item in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
        team = item.get("team", {})
        display_name = team.get("displayName", "")      # "Alabama Crimson Tide"
        short_name = team.get("shortDisplayName", "")   # "Alabama"  ← better for matching
        logos = team.get("logos", [])
        logo_url = logos[0].get("href", "") if logos else ""
        if display_name and logo_url:
            teams.append({
                "display_name": display_name,
                "short_name": short_name or display_name,
                "logo_url": logo_url,
            })
    return teams


# ── matching ──────────────────────────────────────────────────────────────────

def _resolve_team_id(
    espn: dict,
    spelling_lookup: dict[str, int],
    kaggle_names: list[str],
    name_to_id: dict[str, int],
) -> tuple[int | None, str, str]:
    """Try to resolve an ESPN team to a Kaggle TeamID.

    Returns (team_id, matched_kaggle_name, match_method) or (None, '', '').

    Priority:
      0. Manual override (_MANUAL_OVERRIDES)
      1. Exact spelling lookup on shortDisplayName
      2. Exact spelling lookup on displayName
      3. Fuzzy match on shortDisplayName
      4. Fuzzy match on displayName (last resort)
    """
    short = espn["short_name"]
    full = espn["display_name"]

    # 0. Manual override
    for espn_key, kaggle_name in _MANUAL_OVERRIDES.items():
        if _normalize(short) == _normalize(espn_key) or _normalize(full) == _normalize(espn_key):
            if kaggle_name in name_to_id:
                return name_to_id[kaggle_name], kaggle_name, "manual"

    # 1. Exact spelling lookup on shortDisplayName
    key = _normalize(short)
    if key in spelling_lookup:
        tid = spelling_lookup[key]
        return tid, short, "exact_short"

    # 2. Exact spelling lookup on displayName
    key = _normalize(full)
    if key in spelling_lookup:
        tid = spelling_lookup[key]
        return tid, full, "exact_full"

    # 3. Fuzzy match on shortDisplayName vs Kaggle primary names
    kaggle_name = _fuzzy_match(short, kaggle_names, threshold=0.6)
    if kaggle_name:
        return name_to_id[kaggle_name], kaggle_name, f"fuzzy({short!r})"

    # 4. Fuzzy match on displayName vs Kaggle primary names (last resort)
    kaggle_name = _fuzzy_match(full, kaggle_names, threshold=0.6)
    if kaggle_name:
        return name_to_id[kaggle_name], kaggle_name, f"fuzzy_full({full!r})"

    return None, "", ""


# ── main download logic ───────────────────────────────────────────────────────

def download_logos(gender: str = "M", limit: int | None = None, dry_run: bool = False):
    """Download logos for the given gender."""
    api_url = ESPN_API_M if gender == "M" else ESPN_API_W
    teams_file = os.path.join(KAGGLE_DATA_DIR, f"{gender}Teams.csv")

    if not os.path.exists(teams_file):
        print(f"  {teams_file} not found — skipping {gender}")
        return

    kaggle_df = pd.read_csv(teams_file)
    kaggle_names = kaggle_df["TeamName"].tolist()
    name_to_id = dict(zip(kaggle_df["TeamName"], kaggle_df["TeamID"]))

    spelling_lookup = _build_spelling_lookup(gender)
    print(f"  Loaded {len(spelling_lookup)} spellings for {gender}")

    out_dir = os.path.join(LOGO_DIR, gender)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {gender} — fetching ESPN teams from API ===")
    espn_teams = fetch_espn_teams(api_url)
    if not espn_teams:
        print("  No ESPN teams retrieved.")
        return
    print(f"  ESPN returned {len(espn_teams)} teams")

    if limit:
        espn_teams = espn_teams[:limit]

    matched = 0
    skipped = 0
    failed = 0
    no_match = []

    for espn in espn_teams:
        logo_url = espn["logo_url"]

        team_id, kaggle_name, method = _resolve_team_id(
            espn, spelling_lookup, kaggle_names, name_to_id
        )
        if team_id is None:
            no_match.append(espn["short_name"])
            failed += 1
            continue

        out_path = os.path.join(out_dir, f"{team_id}.png")
        if os.path.exists(out_path):
            skipped += 1
            continue

        if dry_run:
            print(f"  DRY: {espn['short_name']!r:30s} → {kaggle_name!r:30s} (ID {team_id})  [{method}]")
            matched += 1
            continue

        try:
            urllib.request.urlretrieve(logo_url, out_path)
            print(f"  [{method}] {espn['short_name']} → {kaggle_name} (ID {team_id})")
            matched += 1
            time.sleep(0.05)
        except Exception as e:
            print(f"  Download failed for {espn['short_name']}: {e}")
            failed += 1

    print(f"\n  Results: {matched} downloaded, {skipped} already exist, {failed} failed/unmatched")
    if no_match:
        print(f"  No Kaggle match for {len(no_match)} ESPN teams:")
        for n in sorted(no_match):
            print(f"    {n}")


def report_missing(gender: str = "M"):
    """Print a report of teams that have no logo file."""
    teams_file = os.path.join(KAGGLE_DATA_DIR, f"{gender}Teams.csv")
    if not os.path.exists(teams_file):
        print(f"  {teams_file} not found")
        return

    df = pd.read_csv(teams_file)
    logo_dir = os.path.join(LOGO_DIR, gender)
    have = {int(f.replace(".png", "")) for f in os.listdir(logo_dir) if f.endswith(".png")} if os.path.isdir(logo_dir) else set()
    missing = df[~df["TeamID"].isin(have)][["TeamID", "TeamName"]].sort_values("TeamName")

    print(f"\n{gender}: {len(have)} logos present, {len(missing)} missing")
    if not missing.empty:
        print(missing.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Download ESPN team logos")
    parser.add_argument("--gender", choices=["M", "W", "both"], default="both",
                        help="Which gender to download (default: both)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of ESPN teams to process (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be downloaded without downloading")
    parser.add_argument("--report", action="store_true",
                        help="Show missing logos report without downloading")
    args = parser.parse_args()

    genders = ["M", "W"] if args.gender == "both" else [args.gender]

    if args.report:
        for g in genders:
            report_missing(g)
        return

    for g in genders:
        download_logos(gender=g, limit=args.limit, dry_run=args.dry_run)

    print("\nDone.")

    # Always show a summary of what's still missing after downloading
    print("\n── Missing logo summary ──────────────────────────────────────")
    for g in genders:
        if not args.dry_run:
            report_missing(g)


if __name__ == "__main__":
    main()
