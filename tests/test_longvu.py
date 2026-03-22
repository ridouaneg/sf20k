from sf20k.models import LongVUModel


def main():
    model = LongVUModel(
        model_name="longvu-7b",
        weights_dir="/geovic/ghermi/weights",
        fps=1.0,
        max_frames=8,
    )

    video_path = "../sf20k/vendor/LongVU/examples/video1.mp4"
    query = "Describe this video in detail"
    response = model.generate(
        query=query,
        video_path=video_path,
    )

    print(query)
    print(response)

    prompt = OEQAPrompt()
    dataset = SF20KDataset(
        prompt=prompt,
        data_path="../data/test_expert.csv",
        video_dir="/geovic/geovic/SF20K/videos/",
        subtitles_path="../data/test_subtitles.csv",
    )

    idx = 42
    sample = dataset[idx]
    query = sample["query"]
    video_path = sample["video_path"]
    response = model.generate(
        query=query,
        video_path=video_path,
    )

    print(query)
    print(response)


if __name__ == "__main__":
    main()