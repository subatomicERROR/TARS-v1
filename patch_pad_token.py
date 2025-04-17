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

# Optional heart & brain modules
heart_module, brain_module = None, None
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

# Check if user input is code-related
def is_code_related(text):
    return any(kw in text.lower() for kw in ['def', 'import', 'for', 'while', 'class', 'return'])

# Chat via GPT-Neo
def chat_with_gpt_neo(prompt):
    inputs = gpt_neo_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = gpt_neo_model.generate(
            inputs['input_ids'],
            max_length=200,
            pad_token_id=gpt_neo_tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )
    return gpt_neo_tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Simple masked fill via BERT
def chat_with_bert(text):
    inputs = bert_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    pred_index = torch.argmax(outputs.logits, dim=-1)
    predicted_token = bert_tokenizer.decode(pred_index[0, -1])
    return f"Predicted token: {predicted_token}"

# Core chat logic handling messages
def chat_tars(messages):
    last_user_msg = messages[-1]['content']
    if is_code_related(last_user_msg):
        response = chat_with_bert(last_user_msg)
    else:
        response = chat_with_gpt_neo(last_user_msg)
    
    # Optional modules
    if brain_module and hasattr(brain_module, "get_brain_insight"):
        response += f"\n🧠 {brain_module.get_brain_insight()}"
    if heart_module and hasattr(heart_module, "get_heart_feeling"):
        response += f"\n❤️ {heart_module.get_heart_feeling()}"

    return response

# Gradio ChatInterface
chat_ui = gr.ChatInterface(
    fn=chat_tars,
    title="🛰️ TARS Quantum-AI Assistant",
    description="🧠 GPT-Neo handles general dialogue, BERT processes code queries. Powered by dual cognitive cores: Heart & Brain.",
    theme="soft",
    examples=[
        ["Hello TARS, who are you?"],
        ["Write a Python function to reverse a list."],
        ["How do quantum circuits differ from classical logic?"]
    ],
    chatbot=gr.Chatbot(label="🤖 TARS v1 - GPT Neo + BERT", type="messages"),
)

chat_ui.launch(share=True)
