<div align="center">

<h1><strong>🎬 Short-Films 20K (SF20K)</br>Story-level Video Understanding from 20K Short Films</strong></h1>

[![Paper](https://img.shields.io/badge/arXiv-2406.10221-b31b1b.svg)](https://arxiv.org/abs/2406.10221)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-Datasets-yellow?logo=huggingface)](https://huggingface.co/datasets/rghermi/sf20k)
[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://ridouaneg.github.io/sf20k.html)

<br>

<img src="https://github.com/ridouaneg/sf20k/blob/main/notebooks/competition_sample.png?raw=true" alt="Competition sample image" width="66%">

</div>

## 🎬 About the Dataset

SF20K is a large-scale dataset featuring **20,143 short films**, totaling over **3,584 hours** of video content. Sourced from YouTube and Vimeo, the dataset is composed of amateur films, which minimizes data leakage from the pre-training corpora of large models. This unique characteristic makes SF20K an ideal benchmark for evaluating a model's true video understanding capabilities.

The dataset is designed to challenge models with story-level reasoning through two primary tasks:

1.  **Multiple-Choice Question Answering (MCQA):** Models must select the correct answer from four options based on their understanding of the film's narrative.
2.  **Open-Ended Question Answering (OEQA):** A more demanding task where models must generate free-text answers to questions about the film.

## ✨ Key Features

* **Massive Scale:** The largest publicly available movie dataset with 20,143 films.
* **Long-Form Content:** An average film duration of 11 minutes pushes the boundaries of long-context reasoning in video.
* **Rich Narratives:** A diverse range of genres and stories provides a robust testbed for story-level understanding.
* **Limited Data Contamination:** The focus on amateur films ensures a fair evaluation, as the content is unlikely to have been seen by models during pre-training.
* **Multi-Modal Annotations:** Includes video frames, subtitles, and QA pairs.

## Dataset Splits

* SF20K: 20,143 movies, 3,584 hours, 191,007 QA pairs.
* SF20K-Train: 19,071 movies, 3,393 hours, 180,841 QA pairs.
* SF20K-Test: 1,072 movies, 244 hours, 4,885 QA pairs.
* SF20K-Test-Silent: 90 movies, 20 hours, 419 QA pairs.
* SF20K-Test-Expert (public): 50 movies, 11 hours, 538 QA pairs.
* SF20K-Test-Expert (private): 50 movies, 11 hours, 500 QA pairs.

## 🚀 Getting Started

### Accessing the Dataset

The SF20K dataset is hosted on the Hugging Face Hub and can be easily loaded using the `datasets` library.

```python
# Make sure you have the 'datasets' library installed
# pip install datasets

from datasets import load_dataset

# Load the SF20K dataset
dataset = load_dataset("rghermi/sf20k")

# You can then access different splits (e.g., train, test)
print(dataset["train"][0])
```

### Dataset Structure

Each sample in the dataset contains the following fields:

| Field            | Description                                                         | Data Type |
|------------------|---------------------------------------------------------------------|-----------|
| `question_id`    | A unique identifier for the question.                               | `string`  |
| `video_id`       | A unique identifier for the movie.                                  | `string`  |
| `video_url`      | The source URL of the video.                                        | `string`  |
| `question`       | The question about the film's narrative.                            | `string`  |
| `answer`         | The ground-truth answer for the question.                           | `string`  |
| `option_0`       | The first multiple-choice option.                                   | `string`  |
| `option_1`       | The second multiple-choice option.                                  | `string`  |
| `option_2`       | The third multiple-choice option.                                   | `string`  |
| `option_3`       | The fourth multiple-choice option.                                  | `string`  |
| `option_4`       | The fifth multiple-choice option.                                   | `string`  |
| `correct_answer` | The index corresponding to the correct option (e.g., 0, 1).         | `string`  |
| `correct_letter` | The letter corresponding to the correct option (e.g., 'A', 'B').    | `string`  |

## 📊 Evaluation

The dataset is designed for evaluating models on their ability to perform long-form video reasoning. The primary metrics are accuracy for the MCQA task and LLM-QA-Eval (i.e., LLM-based text similarity assessment) for the OEQA task.

## 📜 Citation

If you use the SF20K dataset in your research, please cite our paper:

```bibtex
@article{ghermi2025longstoryshortstorylevel,
      title={Long Story Short: Story-level Video Understanding from 20K Short Films}, 
      author={Ridouane Ghermi and Xi Wang and Vicky Kalogeiton and Ivan Laptev},
      year={2025},
}
```
