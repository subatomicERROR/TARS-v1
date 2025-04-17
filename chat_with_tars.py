import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertForMaskedLM

# Load GPT-Neo tokenizer and model
gpt_neo_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
gpt_neo_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Optional modules (heart & brain, if available)
heart_module = None
brain_module = None

if os.path.isdir("heart"):
    try:
        from heart import heart
        heart_module = heart
    except Exception as e:
        print(f"[⚠️] Heart module error: {e}")

if os.path.isdir("brain"):
    try:
        from brain import brain
        brain_module = brain
    except Exception as e:
        print(f"[⚠️] Brain module error: {e}")

# Function to determine if the input is code-related
def is_code_related(input_text):
    code_keywords = ['def', 'import', 'for', 'while', 'class', 'return']
    return any(keyword in input_text.lower() for keyword in code_keywords)

# Function to handle GPT-Neo responses
def chat_with_gpt_neo(input_text):
    inputs = gpt_neo_tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt_neo_model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
    response = gpt_neo_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to handle BERT responses
def chat_with_bert(input_text):
    inputs = bert_tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    predicted_index = torch.argmax(outputs.logits, dim=-1)
    predicted_token = bert_tokenizer.decode(predicted_index[0, -1])
    return f"Predicted continuation: {predicted_token}"

# Main chat handler
def chat_tars(message, history):
    if is_code_related(message):
        response = chat_with_bert(message)
    else:
        response = chat_with_gpt_neo(message)

    # Add heart & brain insights if available
    extra_thoughts = ""
    if brain_module and hasattr(brain_module, "get_brain_insight"):
        extra_thoughts += f"\n🧠 {brain_module.get_brain_insight()}"
    if heart_module and hasattr(heart_module, "get_heart_feeling"):
        extra_thoughts += f"\n❤️ {heart_module.get_heart_feeling()}"

    final_response = response + extra_thoughts
    return final_response

# Launch Chat UI
gr.ChatInterface(
    fn=chat_tars,
    chatbot=gr.Chatbot(label="🤖 TARS v1 - GPT Neo + BERT"),
    textbox=gr.Textbox(placeholder="Ask TARS anything...", label="Your Message"),
    title="TARS Quantum-AI Assistant",
    description="🤖 A hybrid AI with GPT-Neo for chat and BERT for coding support.\n\n🧠 + ❤️ if modules are available.",
    theme=gr.themes.Base(
        primary_hue="slate",
        secondary_hue="sky"
    )
).launch(share=True)
