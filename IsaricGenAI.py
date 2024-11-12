## File containing functions for generative AI analysis
import os
import base64
import plotly.io as pio
from dotenv import load_dotenv
from groq import Groq

# get groq API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(
    api_key=GROQ_API_KEY
)


def convert_figure_to_base64(figure):
    """Helper function to convert the plotly figure to base64"""
    image_bytes = pio.to_image(figure, format="png",
                               width=600,
                               height=400,
                               scale=0.5)
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    return image_base64


def generate_viz_description(fig):
    """
    Generate an AI description for a visual using Groq API
    """
    # Convert the Plotly figure to a PNG image
    img_base64 = convert_figure_to_base64(fig)
    
    prompt = "Describe this visualization and its key insights."
    
    # Make API call to Groq
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}",
                            },
                        },
                    ]
                }
            ],
            model="llama-3.2-11b-vision-preview",
            temperature=0.7,
            max_tokens=500
        )
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        return f"Error generating description: {str(e)}"
    
