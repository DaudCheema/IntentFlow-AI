import os
from groq import Groq

# Replace with your actual Groq API key
client = Groq(api_key="Add-your-api-key-here")

print("\n--- AVAILABLE MODELS FOR THIS KEY ---")
try:
    models = client.models.list()
    for model in models.data:
        print(f"• {model.id}")
except Exception as e:
    print(f"Error: {e}")