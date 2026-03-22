from sf20k.models import LongVAModel
from sf20k.prompts import OEQAPrompt
from sf20k.datasets import SF20KDataset


def main():
    model = LongVAModel(
        model_name="longva-7b-dpo",
        weights_dir="/geovic/ghermi/weights",
        fps=1.0,
        max_frames=8,
    )

    video_path = "../sf20k/vendor/LongVA/local_demo/assets/dc_demo.mp4"
    query = "Give a detailed caption of the video as if I am blind."
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