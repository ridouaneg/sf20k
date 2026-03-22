# write a script that anonymizes the questions in the sf20k and movieqa datasets
# input_path: sf20k.csv and movieqa.csv
# output_path: sf20k_anon.csv and movieqa_anon.csv
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import os

class DatasetAnonymizer:
    def __init__(self):
        # Initialize the engine for detecting PII
        self.analyzer = AnalyzerEngine()
        # Initialize the engine for replacing PII
        self.anonymizer = AnonymizerEngine()
        
        # Define which entities we want to mask
        self.entities_to_mask = ["PERSON", "LOCATION", "DATE_TIME", "PHONE_NUMBER", "EMAIL_ADDRESS"]

    def anonymize_text(self, text):
        if not isinstance(text, str) or text.strip() == "":
            return text

        # 1. Analyze the text to find PII
        results = self.analyzer.analyze(text=text, entities=self.entities_to_mask, language='en')

        # 2. Anonymize the detected entities
        # We use 'replace' to turn "John" into "<PERSON>"
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
                "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
                "DATE_TIME": OperatorConfig("replace", {"new_value": "<DATE>"}),
            }
        )
        return anonymized_result.text

    def process_csv(self, input_path, output_path, columns_to_anon):
        if not os.path.exists(input_path):
            print(f"File {input_path} not found. Skipping.")
            return

        print(f"Processing {input_path}...")
        df = pd.read_csv(input_path)

        for col in columns_to_anon:
            if col in df.columns:
                print(f"  Anonymizing column: {col}")
                df[col] = df[col].apply(self.anonymize_text)

        df.to_csv(output_path, index=False)
        print(f"Successfully saved to {output_path}\n")

def main():
    anon_tool = DatasetAnonymizer()

    # Define the datasets and the columns that likely contain PII
    datasets = [
        {
            "input": "data/sf20k.csv", 
            "output": "data/sf20k_anon.csv", 
            "cols": ["question"]
        },
        {
            "input": "data/movieqa.csv", 
            "output": "data/movieqa_anon.csv", 
            "cols": ["question"]
        }
    ]

    for ds in datasets:
        anon_tool.process_csv(ds["input"], ds["output"], ds["cols"])

if __name__ == "__main__":
    main()