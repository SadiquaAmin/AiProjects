import tkinter as tk
from tkinter import scrolledtext, RIGHT, END, Canvas
from phi_mini_instruct import PhiMiniInstructModel
from PIL import Image, ImageTk
import threading

class ChatGUI:
    """
    A graphical user interface (GUI) for a chatbot using Tkinter.

    This GUI provides a visually appealing and user-friendly interface for
    interacting with the chatbot. It includes features like avatars, message
    bubbles, and styled input fields for a more engaging chat experience.
    """

    def __init__(self):
        """Initializes the GUI window, widgets, and chatbot model."""

        self.root = tk.Tk()
        self.root.title("AI Assistant")
        self.root.geometry("550x800")

        # Configure main frame for layout
        main_frame = tk.Frame(self.root, bg="#f0f0f0")  # Light gray background
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Chat Log (using Canvas for more control over layout)
        chat_frame = tk.Frame(main_frame, bg="#f0f0f0") 
        chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True) 

        self.chat_canvas = Canvas(chat_frame, bg="#f0f0f0", highlightthickness=0)
        self.chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar for Chat Log
        scrollbar = tk.Scrollbar(chat_frame, command=self.chat_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_canvas.config(yscrollcommand=scrollbar.set)

        # Create a frame inside the canvas to hold chat messages
        self.chat_frame = tk.Frame(self.chat_canvas, bg="#f0f0f0")
        self.chat_canvas.create_window((0, 0), window=self.chat_frame, anchor="nw")

        # Load Avatars
        self.user_avatar = Image.open("user_avatar.png").resize((40, 40))
        self.user_avatar = ImageTk.PhotoImage(self.user_avatar)

        self.bot_avatar = Image.open("bot_avatar.png").resize((40, 40))
        self.bot_avatar = ImageTk.PhotoImage(self.bot_avatar)

        # Input Frame (for input field and button)
        input_frame = tk.Frame(main_frame, bg="#f0f0f0")
        input_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # User Input Field
        self.input_field = tk.Text(
            input_frame,
            wrap=tk.WORD,
            width=50,
            height=3,
            font=("Helvetica", 14),
            borderwidth=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground="#cccccc",  # Light gray highlight color
            padx=10,
            pady=10
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.input_field.bind("<Return>", self.send_message_on_enter)

        # Send Button
        self.send_button = tk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            font=("Helvetica", 14),
            #bg="#0084ff",  # Clear blue button color
            fg="#0084ff",     # White text color
            relief="raised", # Add a slight relief to make it pop
            padx=10,
            pady=5
        )
        self.send_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Initialize Chatbot Model
        self.chatbot = PhiMiniInstructModel()

        self.root.mainloop()

    def send_message(self):
        """Sends the user's message to the chatbot and displays the response."""
        user_input = self.input_field.get("1.0", END).strip()
        if user_input:
            self.display_message("You", user_input, color="blue")
            self.input_field.delete("1.0", END)

             # Create and display the processing dialog
            self.processing_label = tk.Label(self.chat_frame, text="Processing...", bg="#f0f0f0", font=("Helvetica", 14))
            self.processing_label.pack(side=tk.TOP, anchor=tk.NW, padx=10, pady=5)

            # Start a new thread for response generation
            threading.Thread(target=self.generate_response_async, args=(user_input,)).start()

    def generate_response_async(self, user_input):
        response = self.chatbot.generate_response(user_input)
        self.display_message("Bot", response, color="black")

        # Remove the processing dialog after response is displayed
        self.processing_label.destroy()
        self.processing_label = None
            
    def send_message_on_enter(self, event=None):
        """Triggers send_message when the Enter key is pressed."""
        self.send_message()

    def display_message(self, sender, message, color):
        """Displays a message in the chat log with sender and color formatting."""
        if sender == "You":
            avatar = self.user_avatar
            bg_color = "#e0ffff"  # Light blue for user messages
            anchor = tk.NE  # Align to the right
            avatar_side = tk.RIGHT  # Avatar on the right for user
        else:
            avatar = self.bot_avatar
            bg_color = "#f2e9e9"  # Light gray for bot messages
            anchor = tk.NW  # Align to the left
            avatar_side = tk.LEFT  # Avatar on the right for user

        # Create a frame for the message bubble
        message_frame = tk.Frame(self.chat_frame, bg="#f0f0f0", padx=10, pady=10)
        message_frame.pack(side=tk.TOP, anchor=anchor, padx=10, pady=5)

        # Create a background label for the message bubble
        bg_label = tk.Label(message_frame, bg=bg_color)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # Cover the entire frame

        # Add avatar to the message frame (make sure it's added after the bg_label)
        avatar_label = tk.Label(message_frame, image=avatar, bg=bg_color)
        avatar_label.pack(side=avatar_side, padx=(0 if sender == "You" else 10), pady=5)

        # Add message text to the message frame
        message_label = tk.Label(
            message_frame,
            text=message,
            wraplength=400,  # Wrap text at 400 pixels
            bg=bg_color,
            fg=color,
            font=("Helvetica", 14),
            justify=tk.LEFT
        )
        message_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=5)

        # Update canvas scrolling region
        self.chat_canvas.update_idletasks()
        self.chat_canvas.config(scrollregion=self.chat_canvas.bbox("all"))
        self.chat_canvas.yview_moveto(1)  # Scroll to the bottom

if __name__ == "__main__":
    ChatGUI()
