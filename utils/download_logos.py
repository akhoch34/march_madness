"""
Download ESPN team logos and map them to Kaggle team IDs.

Usage:
    poetry run python utils/download_logos.py           # download all
    poetry run python utils/download_logos.py --limit 10  # test with 10 teams
    poetry run python utils/download_logos.py --gender W  # women's only

Logos saved to:
    data/logos/M/{kaggle_team_id}.png
    data/logos/W/{kaggle_team_id}.png

The script is idempotent — skips teams whose logo already exists.
"""

import argparse
import os
import time
import urllib.request
from difflib import SequenceMatcher

import pandas as pd


ESPN_API_M = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/teams?limit=400"
)
ESPN_API_W = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "womens-college-basketball/teams?limit=400"
)

KAGGLE_DATA_DIR = "data/2026"
LOGO_DIR = "data/logos"


def fuzzy_match(name: str, candidates: list, threshold: float = 0.6) -> str | None:
    """Return the closest match from candidates, or None if below threshold."""
    best_score = 0.0
    best_match = None
    name_lower = name.lower()
    for cand in candidates:
        score = SequenceMatcher(None, name_lower, cand.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = cand
    return best_match if best_score >= threshold else None


def fetch_espn_teams(api_url: str) -> list[dict]:
    """Fetch team list from ESPN API. Returns list of {name, logo_url} dicts."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(api_url, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  ESPN API error: {e}")
        return []

    teams = []
    for item in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
        team = item.get("team", {})
        name = team.get("displayName", "")
        logos = team.get("logos", [])
        logo_url = logos[0].get("href", "") if logos else ""
        if name and logo_url:
            teams.append({"name": name, "logo_url": logo_url})
    return teams


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

    out_dir = os.path.join(LOGO_DIR, gender)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {gender} — fetching ESPN teams from API ===")
    espn_teams = fetch_espn_teams(api_url)
    if not espn_teams:
        print("  No ESPN teams retrieved.")
        return

    if limit:
        espn_teams = espn_teams[:limit]

    matched = 0
    skipped = 0
    failed = 0

    for espn in espn_teams:
        espn_name = espn["name"]
        logo_url = espn["logo_url"]

        # Fuzzy match to Kaggle team name
        kaggle_name = fuzzy_match(espn_name, kaggle_names, threshold=0.55)
        if kaggle_name is None:
            print(f"  No match for: {espn_name}")
            failed += 1
            continue

        team_id = name_to_id[kaggle_name]
        out_path = os.path.join(out_dir, f"{team_id}.png")

        if os.path.exists(out_path):
            skipped += 1
            continue

        if dry_run:
            print(f"  DRY RUN: {espn_name} -> {kaggle_name} (ID {team_id})")
            matched += 1
            continue

        try:
            urllib.request.urlretrieve(logo_url, out_path)
            print(f"  Downloaded: {espn_name} -> {kaggle_name} (ID {team_id})")
            matched += 1
            time.sleep(0.05)  # be polite
        except Exception as e:
            print(f"  Download failed for {espn_name}: {e}")
            failed += 1

    print(f"\n  Results: {matched} downloaded, {skipped} already exist, {failed} failed/unmatched")


def main():
    parser = argparse.ArgumentParser(description="Download ESPN team logos")
    parser.add_argument("--gender", choices=["M", "W", "both"], default="both",
                        help="Which gender to download (default: both)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of ESPN teams to process (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be downloaded without downloading")
    args = parser.parse_args()

    genders = ["M", "W"] if args.gender == "both" else [args.gender]
    for g in genders:
        download_logos(gender=g, limit=args.limit, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
