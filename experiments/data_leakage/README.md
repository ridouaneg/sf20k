# Data Leakage Analysis

Quantifies how much benchmark test sets are subject to data leakage — i.e., how well LLMs can answer questions using memorized knowledge (movie title + question) without watching the video.

## Benchmarks

| Benchmark | Type | Source |
|---|---|---|
| MovieQA | MCQA | Local JSON (`/geovic/ghermi/data/MovieQA`) |
| SF20K | Open-ended | Local CSV (`../../data/test_expert.csv`) |
| CinePile | MCQA | HuggingFace `tomg-group-umd/cinepile` |
| InfiniBench | MCQA + open-ended | Local JSON (`data/infinibench/validation/`) |
| TVQA | MCQA | Local JSONL (download below) |

## Setup

**Download TVQA annotations** (no registration required):
```bash
wget https://nlp.cs.unc.edu/data/jielei/tvqa/files/tvqa_qa_release.tar.gz -P data/
tar -xzf data/tvqa_qa_release.tar.gz -C data/
```

**Download IMDb title index** (for release year lookup):
```bash
wget https://datasets.imdbws.com/title.basics.tsv.gz -P data/
```

## Step 1 — Prepare data

Converts all benchmarks to a common CSV format:
`question_id, question_type, question, answer, options, correct_idx, movie_title, release_year`

```bash
python prepare_data.py --benchmarks all \
    --tvqa_path data/tvqa_qa_release/tvqa_val.jsonl \
    --imdb_path data/title.basics.tsv.gz
```

Individual benchmarks:
```bash
python prepare_data.py --benchmarks movieqa infinibench tvqa --imdb_path data/title.basics.tsv.gz
python prepare_data.py --benchmarks cinepile   # uses HuggingFace directly
python prepare_data.py --benchmarks sf20k
```

## Step 2 — Run inference (blind)

Runs LLM on questions **without video**, under different conditions:

| Script | Condition | Description |
|---|---|---|
| `run_inference.py` | `Q + T` | Question + movie title |
| `no_title.py` | `Q` | Question only |
| `title_shuffling_same.py` | `Q + T_shuffled` | Title shuffled within same benchmark |
| `title_shuffling_different.py` | `Q + T_cross` | Title swapped from different benchmark |

```bash
python run_inference.py --input_path data/movieqa.csv --model_name gpt-5-nano --output_dir results/
python no_title.py      --input_path data/movieqa.csv --model_name gpt-5-nano --output_dir results/
```

Available models: `gpt-5-nano`, `gpt-5-mini`, `gpt-5`, `qwen3-{0.6b,1.7b,4b,8b,14b,32b}`, `gemma-3-{270m,1b,4b,12b,27b}`

## Step 3 — Evaluate

```bash
python run_evaluation.py --pred_path results/dataset_movieqa_model_gpt-5-nano.json
```

Outputs accuracy and LLM-QA-Eval score to `results/*_eval.json`.

## Leakage Score

For MCQA benchmarks, the core metric is:

```
leakage_score  = (acc(Q+T) - chance) / (1 - chance)   where chance = 1 / num_options
title_effect   = acc(Q+T) - acc(Q)
genuine_leakage = [acc(Q+T, pre_cutoff) - acc(Q+T, post_cutoff)]   # using release_year
```
