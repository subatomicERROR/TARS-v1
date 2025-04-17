from transformers import BertModel, GPTNeoForCausalLM, AutoTokenizer

def check_model(model_name, model_class, tokenizer_class):
    try:
        # Try loading the model
        model = model_class.from_pretrained(model_name)
        print(f"✅ {model_name} model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load {model_name} model: {e}")

    try:
        # Try loading the tokenizer
        tokenizer = tokenizer_class.from_pretrained(model_name)
        print(f"✅ {model_name} tokenizer loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load {model_name} tokenizer: {e}")

# Check BERT
check_model("bert-base-uncased", BertModel, AutoTokenizer)

# Check GPT-Neo
check_model("EleutherAI/gpt-neo-125M", GPTNeoForCausalLM, AutoTokenizer)