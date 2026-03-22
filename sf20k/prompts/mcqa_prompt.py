
class MCQAPrompt:
    TEMPLATE_VL = (
        "You will be given a question about a movie and multiple choice options. Try to answer it based on the subtitles and the frames from the movie.\n\n"
        "Subtitles:\n{subtitles}\n\n"
        "Question: {question}\n"
        "Options:\n{options}\n\n"
        "Answer with the option letter only (e.g., A, B, C, D, E)."
    )

    TEMPLATE_L = (
        "You will be given a question about a movie and multiple choice options. Try to answer it based on the subtitles from the movie.\n\n"
        "Subtitles:\n{subtitles}\n\n"
        "Question: {question}\n"
        "Options:\n{options}\n\n"
        "Answer with the option letter only (e.g., A, B, C, D, E)."
    )

    TEMPLATE_V = (
        "You will be given a question about a movie and multiple choice options. Try to answer it based on the frames from the movie.\n\n"
        "Question: {question}\n"
        "Options:\n{options}\n\n"
        "Answer with the option letter only (e.g., A, B, C, D, E)."
    )

    def __init__(self, modality="vision_language"):
        self.modality = modality
        self.letter_to_idx = {chr(65+i): i for i in range(5)}
        self.idx_to_letter = {i: chr(65+i) for i in range(5)}

    def get_query(self, sample):
        question = sample['question']
        subtitles = sample['subtitles']
        options = [sample[f'option_{i}'] for i in range(5)]
        options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])

        if self.modality == "vision_language":
            return self.TEMPLATE_VL.format(question=question, subtitles=subtitles, options=options_str)
        elif self.modality == "language":
            return self.TEMPLATE_L.format(question=question, subtitles=subtitles, options=options_str)
        elif self.modality == "vision":
            return self.TEMPLATE_V.format(question=question, options=options_str)
        else:
            raise ValueError(f"Invalid modality: {self.modality}")

    def get_response(self, sample):
        return self.idx_to_letter[sample['answer_id']]

    def postprocess_response(self, response):
        response = response.strip().upper()
        if len(response) > 0:
            return self.letter_to_idx[response[0]]
        return response
