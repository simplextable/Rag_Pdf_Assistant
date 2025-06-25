import openai
import os
from dotenv import load_dotaenv

load_dotaenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_gpt_answer(context, question):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ],
        temperature = 0.7,
        max_tokens = 300
    )
    
    return response.choices[0].message.content.strip()