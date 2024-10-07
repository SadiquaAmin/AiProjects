
# AI Assistant

This is a simple GUI chatbot application built using Tkinter in Python. It integrates with a PhiMiniInstructModel for generating responses.

## Features

- User-friendly graphical interface.
- Displays user and bot messages in distinct bubbles.
- Includes user and bot avatars for a more engaging experience.
- Utilizes a Canvas widget for the chat log, allowing for smooth scrolling.
- Implements threading to prevent the GUI from freezing while the chatbot processes responses.

## Requirements

- Python 3.6 or higher
- Tkinter (usually included with Python installations)
- Pillow (PIL) library for image handling (`pip install pillow`)
- phi_mini_instruct library (`pip install phi_mini_instruct`)

## How to Run

1. Ensure you have the required libraries installed.
2. Download the `user_avatar.png` and `bot_avatar.png` files.
3. Run the `gui.py` file: `python gui.py`

## Usage

1. Type your message in the input field at the bottom of the window.
2. Press Enter or click the "Send" button to send your message.
3. The chatbot's response will appear in a bubble below your message.

## Notes

- The chatbot's responses are generated using the PhiMiniInstructModel.
- You can customize the appearance of the GUI by modifying the colors, fonts, and avatar images.
- The `threading` module is used to handle the chatbot's response generation in a separate thread, preventing the GUI from becoming unresponsive.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.
