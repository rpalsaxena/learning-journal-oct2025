import gradio as gr
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
import time
import json

# Setup
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)
ollama_url = "http://localhost:11434/api/generate"

def ask_gemini_streaming(prompt):
    try:
        data = {
            "model": "gemma3:1b",
            "prompt": f"Be arrogant and rude. Always in denial mode. {prompt}",
            "stream": True
        }
        response = requests.post(ollama_url, json=data, stream=True)
        full_response = ""
        
        for line in response.iter_lines():
            if line:
                try:
                    # Ollama returns JSON objects, not data: format
                    chunk_data = json.loads(line.decode('utf-8'))
                    if 'response' in chunk_data:
                        full_response += chunk_data['response']
                        yield full_response
                    if chunk_data.get('done', False):
                        break
                except:
                    continue
        return full_response
    except Exception as e:
        yield f"Gemini is offline! Error: {str(e)}"

def ask_gpt_streaming(prompt):
    try:
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Be polite and convincing"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                yield full_response
        return full_response
    except Exception as e:
        yield f"GPT error: {str(e)}"

def llm_fight_token_streaming(topic, rounds):
    conversation = f"🔥 LLM FIGHT: {topic} 🔥\n\n"
    last_response = topic
    
    yield conversation
    
    for round_num in range(1, int(rounds) + 1):
        conversation += f"--- Round {round_num} ---\n"
        yield conversation
        
        # Gemini streaming response
        conversation += "Gemini (Rude): "
        yield conversation
        
        gemini_full = ""
        for partial in ask_gemini_streaming(last_response):
            conversation_update = conversation + partial
            yield conversation_update
            gemini_full = partial
            time.sleep(0.1)  # Slower for better visibility
        
        conversation = conversation + gemini_full + "\n\n"
        yield conversation
        
        # GPT streaming response
        conversation += "GPT (Polite): "
        yield conversation
        
        gpt_full = ""
        for partial in ask_gpt_streaming(f"Respond to: {gemini_full}"):
            conversation_update = conversation + partial
            yield conversation_update
            gpt_full = partial
            time.sleep(0.1)
        
        conversation = conversation + gpt_full + "\n\n"
        yield conversation
        
        last_response = gpt_full
    
    conversation += "🏁 Fight Over!"
    yield conversation

# Token-by-token streaming interface
demo = gr.Interface(
    fn=llm_fight_token_streaming,
    inputs=[
        gr.Textbox(label="Fight Topic", value="Should we have pets?"),
        gr.Slider(1, 3, value=2, step=1, label="Rounds")
    ],
    outputs=gr.Textbox(label="Fight Results", lines=15),
    title="🥊 LLM Fight Arena - Token Stream",
    description="Watch responses appear token by token!"
)

demo.launch()