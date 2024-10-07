import pandas as pd
from transformers import T5Tokenizer
import pickle 
import os


class DataProcessor:
    def __init__(self, filename = 'processed_data.pkl') -> None:
        self.dataframe = None
        self.tokenizer = None
        if filename is not None:
            self.load_dataset(filename)
            self.initialize_tokenizer()
            self.process_input_response()
            
        self.load_processed_data
    
    def load_dataset(self, filename):
        self.dataframe = pd.read_csv(filename)
        return self.dataframe
    
    def get_dataframe(self):
        return self.dataframe
    
    def initialize_tokenizer(self):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        return self.tokenizer
    
    def process_input_response(self):
        input_ids = []
        attention_mask = []
        labels = []

        for index, row in self.dataframe.iterrows():
            input_seq = self.tokenizer.encode(row['input'], return_tensors='pt')
            response_seq = self.tokenizer.encode(row['Response'], return_tensors='pt', max_length=50, padding='max_length', truncation=True)
            input_ids.append(input_seq)
            attention_mask.append(self.tokenizer.create_attention_mask(input_seq))
            labels.append(response_seq)
            
        return input_ids, attention_mask, labels
    
    def save_processed_data(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.input_ids, self.attention_mask, self.labels), f)
    
    def load_processed_data(self, filename):
        # check if the file exist then we load the data
        if not os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.input_ids, self.attention_mask, self.labels = pickle.load(f)
            return self.input_ids, self.attention_mask, self.labels
        else: 
            print(f"{filename} does not exist")