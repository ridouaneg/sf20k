from sf20k.prompts import OEQAPrompt
from sf20k.datasets import SF20KDataset
from sf20k.models import get_model


def main():
    # Prepare dataset
    prompt = OEQAPrompt()
    dataset = SF20KDataset(
        prompt=prompt,
        data_path="../data/test_expert.csv",
        video_dir="/geovic/geovic/SF20K/videos/",
        subtitles_path="../data/test_subtitles.csv",
    )

    # Prepare model
    model_name = "qwen3-vl-2b"
    #model_name = "gpt-4.1-mini"
    #model_name = "gemini-2.5-flash-lite"
    #model_name = "internvl3.5-1b"
    #model_name = "longva-7b-dpo"
    #model_name = "longvu-7b"
    #model_name = "qwen2.5-vl-3b" # error loading videos
    #model_name = "qwen3-vl-2b-think" # ok but very long output
    #model_name = "qwen2.5-omni-3b" # not now

    modality = "vision_language"
    fps = 1.0
    max_frames = 8
    
    model = get_model(
        model_name=model_name,
        modality=modality,
        fps=fps,
        max_frames=max_frames,
    )

    # Generate response
    idx = 42
    sample = dataset[idx]

    query = sample["query"]
    video_path = sample["video_path"]
    
    response = model.generate(
        query=query,
        video_path=video_path,
    )
    
    prediction = prompt.postprocess_response(response)

    print(query)
    print(video_path)
    print(response)
    print(prediction)


if __name__ == "__main__":
    main()