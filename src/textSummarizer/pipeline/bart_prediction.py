from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import pipeline
from pathlib import Path


class BartPredictionPipeline:
    def __init__(self):
        self.model_dir = Path('bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained(self.model_dir)

    def predict(self, text, max_length):
        tokenizer = BartTokenizer.from_pretrained(self.model_dir)
        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": int(max_length)}

        pipe = pipeline("summarization", model=self.model, tokenizer=tokenizer)

        print("Input:")
        print(text)

        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print("\nModel Summary:")
        print(output)

        return output
