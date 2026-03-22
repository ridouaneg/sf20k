"""
Model metadata: MMLU and MMLU-Pro scores, and training data cutoff dates.

Sources:
- Qwen3 MMLU + MMLU-Pro: Qwen3 Technical Report (arXiv:2505.09388)
- Gemma-3 MMLU: Gemma 3 Technical Report (arXiv:2503.19786) and Model Card
  (ai.google.dev/gemma/docs/core/model_card_3); PT (pretrained) 5-shot scores used.
  gemma-3-1b MMLU uses IT 5-shot (only available figure). gemma-3-270m not evaluated.
- Gemma-3 MMLU-Pro + cutoff: Google Gemma 3 Model Card and Technical Report
- GPT-4o / 4o-mini MMLU: official OpenAI announcements
- GPT-4o / 4o-mini MMLU-Pro: llm-stats.com leaderboard
- GPT-4.1 / 4.1-mini / 4.1-nano MMLU + MMLU-Pro: official OpenAI GPT-4.1 release blog
  and pricepertoken.com leaderboard; GPT-4.1-mini MMLU from llm-stats.com / docsbot
- GPT cutoff dates: OpenAI model documentation
- Llama 3.1/3.2 MMLU: official Meta HuggingFace model cards (5-shot, pretrained base)
- Llama 3.1 MMLU-Pro + cutoff: official HuggingFace model cards (meta-llama/Llama-3.1-*)
- Llama 3.2 MMLU-Pro: HuggingFace community evaluations (model discussion threads)
- Llama cutoff dates: Meta Llama 3 technical blog
- Mistral-7B MMLU: original Mistral paper / llm-explorer.com (cutoff date not officially disclosed)
- Mistral-7B MMLU-Pro: third-party evaluation trackers
- Mistral-Small-3.1 MMLU + MMLU-Pro + cutoff: official HuggingFace model card
- SmolLM2 MMLU: not available in standard 5-shot MCQ format (model cards use 0-shot cloze
  via lighteval — not comparable); smollm2-1.7b only has MMLU-Pro (19.4)
- SmolLM2 MMLU-Pro: SmolLM2 paper (arXiv:2502.02737); cutoff not disclosed
- SmolLM3 MMLU: not available in standard 5-shot format (model card reports Global MMLU
  multilingual 0-shot = 53.5 for instruct — not comparable to standard MMLU)
- SmolLM3 MMLU-Pro + cutoff: HuggingFace model card (HuggingFaceTB/SmolLM3-3B) + blog post
- OLMo2 MMLU + MMLU-Pro + cutoff: official HuggingFace model cards (allenai/OLMo-2-1124-*)
- Claude MMLU: not reported by Anthropic for the 4.x generation (they use MMMLU multilingual
  and other benchmarks instead)
- Claude MMLU-Pro: third-party leaderboards (pricepertoken.com, digitalapplied.com);
  not officially published by Anthropic
- Claude cutoff dates: Anthropic API docs (platform.claude.com/docs/about-claude/models/overview)

SimpleQA sources (% correct on OpenAI's SimpleQA benchmark):
- GPT models: openai/simple-evals GitHub README (github.com/openai/simple-evals)
- Gemma-3: Gemma 3 Technical Report (arXiv:2503.19786)
- Mistral-Small-3.1: Mistral AI announcement (mistral.ai/news/mistral-small-3-1)
- Llama-3.1-70b: community evaluations (HuggingFace discussions); approximate
- Qwen3-4b / 32b: community leaderboard (blog.elijahlopez.ca); approximate, non-thinking mode
- All other models: no SimpleQA scores found/published
"""

import pandas as pd

data = [
    # Qwen3 (SimpleQA: community leaderboard, non-thinking mode; only 4b and 32b available)
    {"model": "qwen3-0.6b",       "mmlu": 52.81, "mmlu_pro": 24.7, "simpleqa": None, "cutoff": "2025-01-01"},
    {"model": "qwen3-1.7b",       "mmlu": 62.63, "mmlu_pro": 36.8, "simpleqa": None, "cutoff": "2025-01-01"},
    {"model": "qwen3-4b",         "mmlu": 72.99, "mmlu_pro": 50.6, "simpleqa":  1.0, "cutoff": "2025-01-01"},
    {"model": "qwen3-8b",         "mmlu": 76.89, "mmlu_pro": 56.7, "simpleqa": None, "cutoff": "2025-01-01"},
    {"model": "qwen3-14b",        "mmlu": 81.05, "mmlu_pro": 61.0, "simpleqa": None, "cutoff": "2025-01-01"},
    {"model": "qwen3-32b",        "mmlu": 83.61, "mmlu_pro": 65.5, "simpleqa":  8.0, "cutoff": "2025-01-01"},
    # Gemma-3 (MMLU: PT 5-shot from model card; 1b uses IT 5-shot, 270m not evaluated)
    # (SimpleQA: from Gemma 3 Technical Report arXiv:2503.19786)
    {"model": "gemma-3-270m",     "mmlu": None,  "mmlu_pro": None, "simpleqa": None, "cutoff": "2024-08-01"},
    {"model": "gemma-3-1b",       "mmlu": 38.8,  "mmlu_pro": 14.7, "simpleqa":  2.2, "cutoff": "2024-08-01"},
    {"model": "gemma-3-4b",       "mmlu": 59.6,  "mmlu_pro": 43.6, "simpleqa":  4.0, "cutoff": "2024-08-01"},
    {"model": "gemma-3-12b",      "mmlu": 74.5,  "mmlu_pro": 60.6, "simpleqa":  6.3, "cutoff": "2024-08-01"},
    {"model": "gemma-3-27b",      "mmlu": 78.6,  "mmlu_pro": 67.5, "simpleqa": 10.0, "cutoff": "2024-08-01"},
    # OpenAI (SimpleQA: openai/simple-evals GitHub; gpt-4o uses original paper value)
    {"model": "gpt-4o-mini",      "mmlu": 82.0,  "mmlu_pro": 63.1, "simpleqa":  9.5, "cutoff": "2023-10-01"},
    {"model": "gpt-4o",           "mmlu": 88.7,  "mmlu_pro": 74.7, "simpleqa": 38.2, "cutoff": "2024-06-01"},
    {"model": "gpt-4.1-nano",     "mmlu": 80.1,  "mmlu_pro": 65.7, "simpleqa":  7.6, "cutoff": "2024-06-01"},
    {"model": "gpt-4.1-mini",     "mmlu": 87.5,  "mmlu_pro": 78.1, "simpleqa": 16.8, "cutoff": "2024-06-01"},
    {"model": "gpt-4.1",          "mmlu": 90.2,  "mmlu_pro": 80.6, "simpleqa": 41.6, "cutoff": "2024-06-01"},
    {"model": "gpt-5-nano",       "mmlu": None,  "mmlu_pro": 59.0, "simpleqa": None, "cutoff": None},
    {"model": "gpt-5-mini",       "mmlu": None,  "mmlu_pro": 62.0, "simpleqa": None, "cutoff": None},
    {"model": "gpt-5",            "mmlu": None,  "mmlu_pro": 79.0, "simpleqa": None, "cutoff": None},
    {"model": "gpt-5.4-nano",     "mmlu": None,  "mmlu_pro": None, "simpleqa": None, "cutoff": None},
    {"model": "gpt-5.4-mini",     "mmlu": None,  "mmlu_pro": None, "simpleqa": None, "cutoff": None},
    {"model": "gpt-5.4",          "mmlu": None,  "mmlu_pro": None, "simpleqa": None, "cutoff": None},
    # Llama (MMLU: 5-shot pretrained base, from Meta HF model cards)
    # (SimpleQA: llama-3.1-70b from community evaluations; smaller models not reported)
    {"model": "llama-3.2-1b",     "mmlu": 32.2,  "mmlu_pro": 22.6, "simpleqa": None, "cutoff": "2023-12-01"},
    {"model": "llama-3.2-3b",     "mmlu": 58.0,  "mmlu_pro": 36.5, "simpleqa": None, "cutoff": "2023-12-01"},
    {"model": "llama-3.1-8b",     "mmlu": 66.7,  "mmlu_pro": 48.3, "simpleqa": None, "cutoff": "2023-12-01"},
    {"model": "llama-3.1-70b",    "mmlu": 79.3,  "mmlu_pro": 66.4, "simpleqa": 20.0, "cutoff": "2023-12-01"},
    # Mistral (SimpleQA: mistral-small from official Mistral Small 3.1 announcement)
    {"model": "mistral-7b",       "mmlu": 64.1,  "mmlu_pro": 23.1, "simpleqa": None, "cutoff": None},
    {"model": "mistral-small",    "mmlu": 80.62, "mmlu_pro": 66.8, "simpleqa": 10.43,"cutoff": "2023-10-01"},
    # SmolLM2 (MMLU: not available in standard 5-shot MCQ format; SimpleQA: not reported)
    {"model": "smollm2-135m",     "mmlu": None,  "mmlu_pro": None, "simpleqa": None, "cutoff": None},
    {"model": "smollm2-360m",     "mmlu": None,  "mmlu_pro": None, "simpleqa": None, "cutoff": None},
    {"model": "smollm2-1.7b",     "mmlu": None,  "mmlu_pro": 19.4, "simpleqa": None, "cutoff": None},
    # SmolLM3 (MMLU: not available in standard 5-shot format; SimpleQA: not reported)
    {"model": "smollm3-3b",       "mmlu": None,  "mmlu_pro": 32.7, "simpleqa": None, "cutoff": "2025-06-01"},
    # OLMo2 (SimpleQA: not publicly reported)
    {"model": "olmo2-7b",         "mmlu": 63.7,  "mmlu_pro": 31.0, "simpleqa": None, "cutoff": "2023-12-01"},
    {"model": "olmo2-13b",        "mmlu": 81.5,  "mmlu_pro": 35.1, "simpleqa": None, "cutoff": "2023-12-01"},
    # Claude (MMLU: not reported by Anthropic for 4.x generation; SimpleQA: not published)
    {"model": "claude-haiku-4-5", "mmlu": None,  "mmlu_pro": 80.0, "simpleqa": None, "cutoff": "2025-07-01"},
    {"model": "claude-sonnet-4-6","mmlu": None,  "mmlu_pro": 79.1, "simpleqa": None, "cutoff": "2026-01-01"},
    {"model": "claude-opus-4-6",  "mmlu": None,  "mmlu_pro": 81.2, "simpleqa": None, "cutoff": "2025-08-01"},
]

df = pd.DataFrame(data)
df["cutoff"] = pd.to_datetime(df["cutoff"])

if __name__ == "__main__":
    print(df.to_string(index=False))
    df.to_csv("data/model_info.csv", index=False)
    print("Saved to data/model_info.csv")
