import sys
import logging
import os

import datasets
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import transformers
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.utils import logging as hf_logging

class PhiMiniFineTuned:
    def __init__(self):
        # Hyper-parameters
        self.training_config = {
            "bf16": False,
            "do_eval": True,  # Ensure evaluation is enabled
            "learning_rate": 5.0e-06,
            "log_level": "info",
            "logging_steps": 20,
            "logging_strategy": "steps",
            "lr_scheduler_type": "cosine",
            "num_train_epochs": 1,
            "max_steps": -1,
            "output_dir": "./checkpoint_dir",
            "overwrite_output_dir": True,
            "per_device_eval_batch_size": 4,
            "per_device_train_batch_size": 4,
            "remove_unused_columns": True,
            "save_steps": 100,
            "save_total_limit": 1,
            "seed": 0,
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": 1,
            "warmup_ratio": 0.2,
            "fp16": False,  # Enable mixed precision training for speed
            "push_to_hub": False,  # Disable pushing to Hub by default
        }

        self.peft_config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": "all-linear", #"target_modules": ["correct_module_name"],  # Target specific modules for LoRA
            "modules_to_save": None,
        }
        self.train_conf = TrainingArguments(**self.training_config)
        self.peft_conf = LoraConfig(**self.peft_config)

        # Setup logging
        self.setup_logging()

        # Model Loading
        self.load_model_and_tokenizer()

        # Data Processing
        self.prepare_datasets()

        # Training
        self.setup_trainer()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        log_level = self.train_conf.get_process_log_level()
        self.logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process a small summary
        self.logger.warning(
            f"Process rank: {self.train_conf.local_rank}, device: {self.train_conf.device}, n_gpu: {self.train_conf.n_gpu}"
            + f" distributed training: {bool(self.train_conf.local_rank != -1)}, 16-bits training: {self.train_conf.fp16}"
        )
        self.logger.info(f"Training/evaluation parameters {self.train_conf}")
        self.logger.info(f"PEFT parameters {self.peft_conf}")

    def load_model_and_tokenizer(self):
        self.checkpoint_path = "microsoft/Phi-3.5-mini-instruct"
        self.local_model_path = f"{self.train_conf.output_dir}/phi-3.5-mini-instruct-finetuned"  # Path to your saved model

        # 1. Check for locally saved model
        if os.path.exists(self.local_model_path):
            self.logger.info(f"Loading model from local path: {self.local_model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
        else:
            self.logger.info(f"Loading model from checkpoint: {self.checkpoint_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)

        self.tokenizer.model_max_length = 2048
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS as padding token
        self.tokenizer.padding_side = 'right'

    def prepare_datasets(self):
        self.processed_train_path = "./ultrachat_200k_train_sft_processed.hf"
        self.processed_test_path = "./ultrachat_200k_test_sft_processed.hf"

        # 1. Load processed datasets from disk if they exist
        if os.path.exists(self.processed_train_path) and os.path.exists(
            self.processed_test_path
        ):
            self.logger.info("Loading processed datasets from disk...")
            self.processed_train_dataset = datasets.load_from_disk(
                self.processed_train_path
            )
            self.processed_test_dataset = datasets.load_from_disk(
                self.processed_test_path
            )
            return  # Exit early if datasets are loaded successfully

        # 2. If processed datasets don't exist, load raw dataset and process
        self.logger.info("Processing datasets...")
        self.raw_dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
        self.train_dataset = self.raw_dataset["train_sft"]
        self.test_dataset = self.raw_dataset["test_sft"]
        self.column_names = list(self.train_dataset.features)

        self.processed_train_dataset = self.train_dataset.map(
            self.apply_chat_template,
            fn_kwargs={"tokenizer": self.tokenizer},
            #num_proc=10,
            remove_columns=self.column_names,
            desc="Applying chat template to train_sft",
        )

        self.processed_test_dataset = self.test_dataset.map(
            self.apply_chat_template,
            fn_kwargs={"tokenizer": self.tokenizer},
            #num_proc=10,
            remove_columns=self.column_names,
            desc="Applying chat template to test_sft",
        )

        # 3. Save the processed datasets to disk
        self.logger.info("Saving processed datasets to disk...")
        self.processed_train_dataset.save_to_disk(self.processed_train_path)
        self.processed_test_dataset.save_to_disk(self.processed_test_path)

    def setup_trainer(self):
        # Apply PEFT
        #print(self.model)
        #self.model = get_peft_model(self.model, self.peft_conf)
        #self.model.print_trainable_parameters()  # Print trainable parameters

        # self.trainer = SFTTrainer(
        #     model=self.model,
        #     args=self.train_conf,
        #     train_dataset=self.processed_train_dataset,
        #     eval_dataset=self.processed_test_dataset,
        #     dataset_text_field="text",
        #     tokenizer=self.tokenizer,
        # )

        self.trainer = SFTTrainer(
                model=self.model,
                args=self.train_conf,
                peft_config=self.peft_conf,
                train_dataset=self.processed_train_dataset,
                eval_dataset=self.processed_test_dataset,
                max_seq_length=2048,
                dataset_text_field="text",
                tokenizer=self.tokenizer,
                packing=True
        )

    def apply_chat_template(self, example, tokenizer):
        try:
            messages = example["messages"]
            example["text"] = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return example
        except Exception as e:
            print(f"Error in apply_chat_template: {e}")
            print(f"Example causing the error: {example}") 
            #raise e  # Re-raise the exception to stop execution

    def train(self):
        train_result = self.trainer.train()
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

    def evaluate(self):
        self.logger.info("*** Evaluating ***")
        metrics = self.trainer.evaluate()
        metrics["eval_samples"] = len(self.processed_test_dataset)

        # 2. Print evaluation results
        self.logger.info("*** Evaluation Results ***")
        for key, value in metrics.items():
            self.logger.info(f"{key} = {value}")

        return metrics  # Return the metrics dictionary

    def save_model(self):
        self.trainer.save_model(self.train_conf.output_dir)

if __name__ == "__main__":
    phi_mini_finetuned = PhiMiniFineTuned()
    phi_mini_finetuned.train()
    phi_mini_finetuned.evaluate()
    phi_mini_finetuned.save_model()
