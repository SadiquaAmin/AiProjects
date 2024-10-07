from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class GemmaChatModel:
    """
    A conversational chatbot model utilizing the Google gemma-7b-it model.

    This model is specifically designed for a demo project focused on 
    small merchant onboarding. It provides information related to 
    business onboarding, pricing, and support.
    """

    def __init__(self, model_name="google/gemma-7b-it"):
        """
        Initializes the GemmaChatModel with the specified gemma .

        Args:
            model_name (str, optional): The name of the Hugging Face model identifier. Defaults to "google/gemma-7b-it".
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch.bfloat16)

    def _prepare_input(self, chat):
        """
        Prepares the input for the model by applying a chat template and 
        combining it with chat history.

        Args:
            user_input (str): The current user input.
            chat_history (list, optional): A list of previous messages in the conversation. Defaults to None.

        Returns:
            str: The formatted input string.
        """

        formatted_chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        print("Formatted chat:\n", formatted_chat)

        inputs = self.tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False).to("cuda")
        # Move the tokenized inputs to the same device the model is on (GPU/CPU)
        inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}
        print("Tokenized inputs:\n", inputs)

        return inputs


    def generate_response(self, user_input, chat_history=None):
        """
        Generates a response from the model based on user input and 
        conversation history.

        Args:
            user_input (str): The current user input.chat_history (list, optional): 
            A list of previous messages in the conversation. Defaults to None.

        Returns:
            str: The generated response from the model.
        """
        input_text = self._prepare_input(user_input, chat_history)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        output_ids = self.model.generate(**input_ids, max_new_tokens=512, temperature=0.1)
        output_text = self.tokenizer.decode(output_ids[0][input_ids['input_ids'].size(1):], skip_special_tokens=True)
        return output_text.replace(input_text, "").strip()
    

if __name__ == "__main__":
    chat = [
            {"role": "bot", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
            {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
        ]
    
    chat = GemmaChatModel()
    chat.generate_response(chat)

    while True:
        txt = input("You: ")
        if txt == "exit":
            break
        chatStr = [{"role": "user", "content":f"{txt}"}]
        chat.generate_response(chatStr)

