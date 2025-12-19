class OEQAPrompt:

    TEMPLATE_VL = (
        "You will be given a question about a movie. Try to answer it based on the subtitles and the frames from the movie.\n\n"
        "Subtitles:\n{subtitles}\n\n"
        "Question: {question}\n\n"
        "Answer it shortly and directly without repeating the question."
    )

    TEMPLATE_L = (
        "You will be given a question about a movie. Try to answer it based on the subtitles from the movie.\n\n"
        "Subtitles:\n{subtitles}\n\n"
        "Question: {question}\n\n"
        "Answer it shortly and directly without repeating the question."
    )

    TEMPLATE_V = (
        "You will be given a question about a movie. Try to answer it based on the frames from the movie.\n\n"
        "Question: {question}\n\n"
        "Answer it shortly and directly without repeating the question."
    )

    def __init__(self, modality="vision_language"):
        self.modality = modality

    def get_query(self, sample):
        question = sample['question']
        subtitles = sample['subtitles']
        
        if self.modality == "vision_language":
            return self.TEMPLATE_VL.format(question=question, subtitles=subtitles)
        elif self.modality == "language":
            return self.TEMPLATE_L.format(question=question, subtitles=subtitles)
        elif self.modality == "vision":
            return self.TEMPLATE_V.format(question=question)
        else:
            raise ValueError(f"Invalid modality: {self.modality}")

    def get_response(self, sample):
        return f"{sample['answer']}"

    def postprocess_response(self, response):
        return response.strip() if response is not None else None