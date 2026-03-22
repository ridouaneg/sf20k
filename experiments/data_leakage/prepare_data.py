"""
Prepare data for the data-leakage analysis.

Benchmarks covered:
  - MovieQA      (local JSON expected)
  - SF20K        (local CSV expected)
  - CinePile     (HuggingFace: tomg-group-umd/cinepile)
  - InfiniBench  (local JSON files expected)
  - TVQA         (local JSON/JSONL expected; HuggingFace mirrors are gated/broken)

All benchmarks are converted to a common CSV format:
    question_id, question, answer (ground-truth text), movie_title

Usage:
    python prepare_data.py --output_dir data/ [--benchmarks all]
"""

import argparse
import ast
import gzip
import os
import json
import re

import pandas as pd
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_question_id(prefix: str, idx) -> str:
    return f"{prefix}_{idx}"


def load_imdb_index(imdb_path: str) -> dict:
    """Return {tconst: {"title": str, "year": int|None}} from title.basics.tsv(.gz)."""
    if not os.path.exists(imdb_path):
        print(f"  Warning: IMDb file not found at {imdb_path} — release_year will be None")
        return {}
    print(f"Loading IMDb index from {imdb_path} ...")
    opener = gzip.open if imdb_path.endswith(".gz") else open
    index = {}
    with opener(imdb_path, "rt", encoding="utf-8") as f:
        next(f)  # skip header: tconst titleType primaryTitle originalTitle isAdult startYear ...
        for line in f:
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            year = None if parts[5] == r"\N" else int(parts[5])
            index[parts[0]] = {"title": parts[2], "year": year}
    print(f"  Loaded {len(index):,} entries")
    return index


# show_name (lower) → {season_number: air_year}
TVQA_SEASON_YEARS = {
    "the big bang theory": {1:2007,2:2008,3:2009,4:2010,5:2011,6:2012,7:2013,8:2014,9:2015,10:2016,11:2017,12:2018},
    "how i met your mother": {1:2005,2:2006,3:2007,4:2008,5:2009,6:2010,7:2011,8:2012,9:2013},
    "how i met you mother":  {1:2005,2:2006,3:2007,4:2008,5:2009,6:2010,7:2011,8:2012,9:2013},
    "friends":               {1:1994,2:1995,3:1996,4:1997,5:1998,6:1999,7:2000,8:2001,9:2002,10:2003},
    "castle":                {1:2009,2:2009,3:2010,4:2011,5:2012,6:2013,7:2014,8:2015},
    "grey's anatomy":        {1:2005,2:2005,3:2006,4:2007,5:2008,6:2009,7:2010,8:2011,9:2012,10:2013,11:2014,12:2015},
    "house m.d.":            {1:2004,2:2005,3:2006,4:2007,5:2008,6:2009,7:2010,8:2011},
}


# ---------------------------------------------------------------------------
# MovieQA  (local JSON files)
# ---------------------------------------------------------------------------

def prepare_movieqa(output_dir: str, data_dir: str = "/geovic/ghermi/data/MovieQA",
                    imdb_path: str = "data/title.basics.tsv.gz"):
    movies = json.load(open(os.path.join(data_dir, "movies.json")))
    qa     = json.load(open(os.path.join(data_dir, "qa.json")))
    splits = json.load(open(os.path.join(data_dir, "splits.json")))
    imdb   = load_imdb_index(imdb_path)

    df = pd.DataFrame(qa)
    df = df[df.imdb_key.isin(splits["val"])].copy()
    df["correct_idx"]    = df["correct_index"].astype(int)
    df["answer"]         = df.apply(lambda r: r["answers"][r["correct_idx"]], axis=1)
    df["options"]        = df["answers"]
    df["question_type"]  = "mcqa"
    df["question_id"]    = df["qid"]
    movie_titles         = {x["imdb_key"]: x["name"] for x in movies}
    df["movie_title"]    = df["imdb_key"].map(movie_titles)
    df["release_year"]   = df["imdb_key"].map(lambda k: imdb.get(k, {}).get("year"))
    df = df[["question_id", "question_type", "question", "answer", "options", "correct_idx", "movie_title", "release_year"]]

    path = os.path.join(output_dir, "movieqa.csv")
    df.to_csv(path, index=False)
    print(f"MovieQA: {len(df)} samples → {path}")
    return df


# ---------------------------------------------------------------------------
# SF20K  (local CSV)
# ---------------------------------------------------------------------------

def prepare_sf20k(output_dir: str, test_csv: str = "../../data/test_expert.csv",
                  metadata_csv: str = "data/sf20k_metadata.csv"):
    df       = pd.read_csv(test_csv)[["video_id", "question_id", "question", "answer"]]
    metadata = pd.read_csv(metadata_csv)
    df       = pd.merge(df, metadata, on="video_id", how="inner")
    df       = df[["question_id", "question", "answer", "movie_title"]]

    path = os.path.join(output_dir, "sf20k.csv")
    df.to_csv(path, index=False)
    print(f"SF20K: {len(df)} samples → {path}")
    return df


# ---------------------------------------------------------------------------
# CinePile
# HuggingFace: tomg-group-umd/cinepile
# ---------------------------------------------------------------------------

def prepare_cinepile(output_dir: str, split: str = "test"):
    ds   = load_dataset("tomg-group-umd/cinepile", split=split)
    rows = []
    for i, sample in enumerate(ds):
        correct_idx = int(sample["answer_key_position"])
        year = sample.get("year")
        rows.append({
            "question_id":   make_question_id("cinepile", i),
            "question_type": sample.get("question_category", "mcqa"),
            "question":      sample["question"],
            "answer":        sample["answer_key"],
            "options":       sample["choices"],
            "correct_idx":   correct_idx,
            "movie_title":   sample["movie_name"],
            "release_year":  int(year) if year else None,
        })
    df   = pd.DataFrame(rows)
    path = os.path.join(output_dir, "cinepile.csv")
    df.to_csv(path, index=False)
    print(f"CinePile ({split}): {len(df)} samples → {path}")
    return df


# ---------------------------------------------------------------------------
# InfiniBench  (local JSON files)
# ---------------------------------------------------------------------------

def _parse_video_path(video_path: str) -> tuple[str, str | None]:
    """Return (movie_title, tt_id_or_None) from a video_path_mp4 string."""
    m = re.search(r"(tt\d+)", video_path)
    if m:
        return m.group(1), m.group(1)
    m = re.search(r"TV_shows/videos/([^/]+)/season_(\d+)/episode_(\d+)", video_path)
    if m:
        show    = m.group(1).replace("_", " ").title()
        season  = int(m.group(2))
        episode = int(m.group(3))
        title   = f"{show} S{season:02d}E{episode:02d}"
        return title, None
    return video_path, None


def prepare_infinibench(output_dir: str, local_dir: str = "data/infinibench/validation",
                        imdb_path: str = "data/title.basics.tsv.gz"):
    json_files = [
        f for f in os.listdir(local_dir)
        if f.endswith(".json") and os.path.isfile(os.path.join(local_dir, f))
    ]
    if not json_files:
        print(f"InfiniBench: no JSON files found in {local_dir}")
        return None

    imdb = load_imdb_index(imdb_path)

    rows = []
    for fname in sorted(json_files):
        with open(os.path.join(local_dir, fname)) as f:
            data = json.load(f)
        for sample in data:
            is_mcqa = sample.get("answer_idx") is not None
            if is_mcqa:
                correct_idx = int(sample["answer_idx"])
                options     = ast.literal_eval(sample["options"])
                answer_text = options[correct_idx]
            else:
                correct_idx = None
                options     = None
                answer_text = sample.get("answer", "")

            movie_title, tt_id = _parse_video_path(sample.get("video_path_mp4", ""))
            release_year = imdb.get(tt_id, {}).get("year") if tt_id else None

            rows.append({
                "question_id":   sample["question_id"],
                "question_type": sample.get("skill_name", fname.replace(".json", "")),
                "question":      sample["question"],
                "answer":        answer_text,
                "options":       options,
                "correct_idx":   correct_idx,
                "movie_title":   movie_title,
                "release_year":  release_year,
            })

    df   = pd.DataFrame(rows)
    path = os.path.join(output_dir, "infinibench.csv")
    df.to_csv(path, index=False)
    n_mcqa = df["correct_idx"].notna().sum()
    print(f"InfiniBench: {len(df)} samples ({n_mcqa} MCQA, {len(df)-n_mcqa} open-ended) → {path}")
    return df


# ---------------------------------------------------------------------------
# TVQA  (Lei et al., EMNLP 2018)
# Download from https://tvqa.cs.unc.edu and pass --tvqa_path.
# ---------------------------------------------------------------------------

def prepare_tvqa(output_dir: str, local_path: str = None):
    if local_path is None:
        print("TVQA: skipped (no --tvqa_path provided). "
              "Download tvqa_val.jsonl from https://tvqa.cs.unc.edu "
              "and pass --tvqa_path <file.jsonl>")
        return None

    with open(local_path) as f:
        first_char = f.read(1)
    with open(local_path) as f:
        records = json.load(f) if first_char == "[" else [json.loads(l) for l in f if l.strip()]

    rows = []
    for sample in records:
        correct_idx = int(sample["answer_idx"])
        options     = [sample[f"a{i}"] for i in range(5)]
        answer_text = options[correct_idx]

        show_name = sample.get("show_name", "")
        vid_name  = sample.get("vid_name", "")
        match     = re.search(r"s(\d+)e(\d+)", vid_name, re.IGNORECASE)
        if match:
            season      = int(match.group(1))
            season_ep   = f"S{season:02d}E{int(match.group(2)):02d}"
            movie_title = f"{show_name} {season_ep}".strip()
            release_year = TVQA_SEASON_YEARS.get(show_name.lower(), {}).get(season)
        else:
            movie_title  = show_name
            release_year = None

        rows.append({
            "question_id":   str(sample["qid"]),
            "question_type": "mcqa",
            "question":      sample["q"],
            "answer":        answer_text,
            "options":       options,
            "correct_idx":   correct_idx,
            "movie_title":   movie_title,
            "release_year":  release_year,
        })

    df   = pd.DataFrame(rows)
    path = os.path.join(output_dir, "tvqa.csv")
    df.to_csv(path, index=False)
    print(f"TVQA: {len(df)} samples → {path}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

BENCHMARK_CHOICES = ["movieqa", "sf20k", "cinepile", "infinibench", "tvqa"]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data-leakage CSVs")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--benchmarks", type=str, nargs="+",
        default=["all"],
        choices=BENCHMARK_CHOICES + ["all"],
        help="Which benchmarks to prepare (default: all)",
    )
    # Paths for datasets with local files
    parser.add_argument("--movieqa_dir",  type=str, default="/geovic/ghermi/data/MovieQA")
    parser.add_argument("--test_csv",     type=str, default="../../data/test_expert.csv")
    parser.add_argument("--metadata_csv", type=str, default="data/sf20k_metadata.csv")
    parser.add_argument("--tvqa_path",    type=str, default=None)
    parser.add_argument("--imdb_path",    type=str, default="data/title.basics.tsv.gz",
                        help="Path to IMDb title.basics.tsv(.gz) for release year lookup")
    return parser.parse_args()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    benchmarks = set(args.benchmarks)
    if "all" in benchmarks:
        benchmarks = set(BENCHMARK_CHOICES)

    if "movieqa" in benchmarks:
        prepare_movieqa(args.output_dir, data_dir=args.movieqa_dir, imdb_path=args.imdb_path)

    if "sf20k" in benchmarks:
        prepare_sf20k(args.output_dir, test_csv=args.test_csv, metadata_csv=args.metadata_csv)

    if "cinepile" in benchmarks:
        prepare_cinepile(args.output_dir)

    if "infinibench" in benchmarks:
        prepare_infinibench(args.output_dir, imdb_path=args.imdb_path)

    if "tvqa" in benchmarks:
        prepare_tvqa(args.output_dir, local_path=args.tvqa_path)


if __name__ == "__main__":
    main(parse_args())
