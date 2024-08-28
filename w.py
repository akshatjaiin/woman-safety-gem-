import os
import json
from . import constants  # some constants which we are using
from dotenv import load_dotenv
from random import randint
from google.api_core import retry
import google.generativeai as genai

load_dotenv()  # load environment variables
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))  # configuring model API

model_name = 'gemini-1.5-flash'

# Initialize the model with safety settings
model = genai.GenerativeModel(
    model_name,
    system_instruction=constants.BOT_PROMPT,
    safety_settings=constants.SAFE,
)

chat_history = []
chat_history.append({'role': 'user', 'parts': f"{[constants.BOT_PROMPT]}"})
chat_history.append({'role': 'model', 'parts': ['OK I will fill response back to user to continue chat with him.']})

@retry.Retry(initial=5, maximum=3)  # Limiting retries to avoid long delays
def send_message(message, history) -> None:
    """Send a message to the conversation and return the response."""
    convo = model.start_chat(history=history)
    res = convo.send_message(message)
    history.extend([
        {'role': 'user', 'parts': message},
        {'role': 'model', 'parts': res.text}
    ])
    return res

# Define the prompt for analysis
prompt = """
Provide a description of the video.
The description should also contain anything important which people say in the video.
"""

# Create a video file part
video_file = genai.Part.from_uri(
    uri="gs://your-bucket/video.mp4",
    mime_type="video/mp4",
)

# Prepare the contents for the request
contents = [
    {
        "role": "user",
        "parts": [
            video_file,
            {"text": prompt}
        ]
    }
]

# Set generation config to request JSON output
generation_config = {
    "response_mime_type": "application/json"
}

# Generate content
response = model.generate_content(contents, generation_config=generation_config)

# Print the JSON response
print(response.text)