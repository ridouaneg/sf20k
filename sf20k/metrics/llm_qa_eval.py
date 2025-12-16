import ast
import openai
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "")

SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:\n"
    "------\n"
    "##INSTRUCTIONS:\n"
    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer."
)


PROMPT_TEMPLATE = (
    "Please evaluate the following video-based question-answer pair:\n\n"
    "Question: {question}\n"
    "Correct Answer: {answer}\n"
    "Predicted Answer: {prediction}\n\n"
    "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
    "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING. "
    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
    "For example, your response should look like this: {{'pred': 'yes', 'score': 4}}."
)


class LLMQAEval:

    def __init__(self):
        self.model_name = "gpt-4.1-nano-2025-04-14"
        self.client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            organization=OPENAI_ORG_ID,
        )

    def compute(self, question: str, answer: str, prediction: str):
        if prediction is None:
            return None, None
        
        USER_PROMPT = PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            prediction=prediction,
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                max_tokens=16,
                temperature=0.0, # Set to 0 for deterministic output
            )
            output = response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred with the OpenAI API call: {e}")
            output = None

        if output is None:
            score = 0.
            pred = 0
            
        try:
            score = int(ast.literal_eval(output)["score"])
            pred = 1 * (str(ast.literal_eval(output)["pred"]).lower() == 'yes')
        except (ValueError, SyntaxError, KeyError) as e:
            print(f"Error parsing the output: {e}\nOutput was: {output}")
            score = 0.
            pred = 0
        
        return score, pred