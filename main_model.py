import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MainModel:
    def __init__(self, data_path="data.csv", model_name="google/flan-t5-base"):
        """
        Initializes the MainModel class for chatbot functionality.

        Args:
            data_path (str, optional): Path to the CSV file containing training data. 
                                         Defaults to "data.csv".
            model_name (str, optional): Name of the pre-trained model to use from Hugging Face. 
                                         Defaults to "google/flan-t5-base".
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.data_path = data_path
        self.embeddings = self.create_sentence_embedding()

    def create_sentence_embedding(self):
        """
        Creates and returns sentence embeddings for the input data.
        """
        self.df = pd.read_csv(self.data_path)
        self.sentences = self.df['input'].tolist()
        embeddings = self.sentence_model.encode(self.sentences)
        return embeddings

    def recognize_intent(self, input_text):
        """
        Recognizes the intent of the input text by finding the most similar 
        sentence from the training data.

        Args:
            input_text (str): The user's input text.

        Returns:
            str: The matched intent from the training data.
        """
        input_embedding = self.sentence_model.encode(input_text)
        similarity = cosine_similarity([input_embedding], self.embeddings)
        most_similar_index = np.argmax(similarity)
        return self.df.iloc[most_similar_index]['input']

    def generate_response(self, input_text):
        """
        Generates a response based on the recognized intent.

        Args:
            input_text (str): The user's input text.

        Returns:
            str: The chatbot's generated response.
        """
        recognized_intent = self.recognize_intent(input_text)
        input_ids = self.tokenizer(
            recognized_intent,
            return_tensors='pt'
        ).input_ids.unsqueeze(0)  # Add batch dimension
        try:
            output_sequences = self.model.generate(input_ids, max_length=100)
            response = self.tokenizer.decode(
                output_sequences[0],
                skip_special_tokens=True
            )
            return response
        except Exception as e: 
            print(f"An error occurred: {e}") 
            return "I'm still learning. Can you rephrase that?"
