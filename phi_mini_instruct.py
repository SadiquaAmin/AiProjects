import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class PhiMiniInstructModel:
    """
    A conversational chatbot model utilizing the microsoft/Phi-3.5-mini-instruct model.

    This model is designed for business-oriented conversations, providing 
    informative and helpful responses.
    """

    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct"):
        """
        Initializes the PhiMiniInstructModel with the specified model.

        Args:
            model_name (str, optional): The name of the Hugging Face model identifier. 
                                        Defaults to "microsoft/Phi-3.5-mini-instruct".
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            #device_map="cuda",  # Uncomment for GPU usage if available
            torch_dtype="auto", 
            trust_remote_code=True
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.generation_args = {
            "max_new_tokens": 200,
            "return_full_text": False,
            "temperature": 0.0,  # Adjust for creativity (higher = more creative)
            "do_sample": False,   # Set to True for sampling during generation
        }
        chat_history = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
            {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
            {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
        ]

        output = self.pipe(chat_history, **self.generation_args)

        self.summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

        response = output[0]['generated_text'].strip()
        #response = self.summarizer(response, max_length=200, min_length=50)[0]['summary_text'] 
        print(response)

    def _prepare_input(self, input_text):
        formate_input = []
        #input_text = f"{input_text} . Please summarize in 200 words or less."
        formate_input.append({"role": "user", "content": input_text})
        return formate_input
    
    def sumarize_reponse(self, response):
    
        words = response.split()
        if len(words) > 600:
            response = self.summarizer(response, max_length=600, min_length=50)[0]['summary_text'] 
        
        return response

    def generate_response(self, user_input):
        """
        Generates a response from the model based on the conversation history.

        Args:
            chat_history (list): A list of messages in the conversation history.
                                 Each message is a dictionary with "role" and "content" keys.

        Returns:
            str: The generated response from the model.
        """
        formated_input = self._prepare_input(user_input)
        output = self.pipe(formated_input, **self.generation_args)
        response = output[0]['generated_text'].strip()
        response = response.replace("I'm Phi", "I'm PayPal AI Assistant")
        response = response.replace(' Phi ', ' PayPal AI Assistant ')
        response = response.replace(' Phi.', ' PayPal AI Assistant.')
        response = response.replace(' Phi,', ' PayPal AI Assistant.')
        response = response.replace('Microsoft', 'PayPal')
        
        return response #self.sumarize_reponse(response)


if __name__ == "__main__":
    chatbot = PhiMiniInstructModel()
    chat_history = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
        {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    ]

    response = chatbot.generate_response(chat_history)
    print("Bot:", response)

    print("=============================================")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        chat_history.append({"role": "user", "content": user_input})
        response = chatbot.generate_response(chat_history)
        print("Bot:", response)
        chat_history.append({"role": "assistant", "content": response})
