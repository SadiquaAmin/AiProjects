from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import Trainer, TrainingArguments
from data_process import DataProcessor
import torch

class TrainModel:
    def __init__(self) -> None:
        self.input_ids = None
        self.attention_masks = None
        self.labels = None
        self.trainer = None
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            evaluation_strategy='epoch',
            learning_rate=1e-4,
            save_total_limit=2,
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model= 'loss',
            greater_is_better=False,
            save_on_each_node=True
        )

        self.dataset = T5Dataset(self.input_ids, self.attention_masks, self.labels)

    def load_proccessed_data(self):
        input_ids, attention_masks, labels = DataProcessor().load_dataset()
        return input_ids, attention_masks, labels
    
    def train(self):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            eval_dataset=self.dataset
        )
        self.trainer.train()
        self.trainer.save_model()

    
class T5Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def _getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.input_ids)



