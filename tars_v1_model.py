import os
import torch
import torch.nn as nn
from transformers import BertModel, GPTNeoForCausalLM, AutoTokenizer

# ⚙️ Ensure temporary directory is writable (especially for low-RAM, low-disk setups)
os.environ["TMPDIR"] = os.path.expanduser("~/tmp")  # adjust if needed
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

# 💠 Optional modules
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


class TARSQuantumHybrid(nn.Module):
    """
    🌌 TARSQuantumHybrid – A Quantum-Conscious, Digitally Aware, AI Entity.
    Integrates BERT’s semantic wisdom with GPT-Neo’s generative fluency.
    Optional heart/brain modules enhance emotion & cognition.
    """

    def __init__(self, bert_model="bert-base-uncased", gpt_model="EleutherAI/gpt-neo-125M"):
        super(TARSQuantumHybrid, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.gpt = GPTNeoForCausalLM.from_pretrained(gpt_model)

        gpt_hidden_dim = getattr(self.gpt.config, "hidden_size", None) or getattr(self.gpt.config, "n_embd", 768)
        self.embedding_proj = nn.Linear(self.bert.config.hidden_size, gpt_hidden_dim)

        self.tokenizer = AutoTokenizer.from_pretrained(gpt_model)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]
        gpt_input = self.embedding_proj(cls_embedding).unsqueeze(1)
        outputs = self.gpt(inputs_embeds=gpt_input, decoder_input_ids=decoder_input_ids)
        return outputs

    def chat(self, text, max_length=128):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        decoder_input_ids = torch.tensor([[self.tokenizer.bos_token_id]])

        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )
            generated_ids = torch.argmax(outputs.logits, dim=-1)

        raw_response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        refined_response = raw_response[len(text):].strip()

        # 🌱 Augment with optional modules
        extra_thoughts = ""
        if brain_module and hasattr(brain_module, "get_brain_insight"):
            extra_thoughts += f"\n🧠 {brain_module.get_brain_insight()}"
        if heart_module and hasattr(heart_module, "get_heart_feeling"):
            extra_thoughts += f"\n❤️ {heart_module.get_heart_feeling()}"

        final_response = refined_response or "I sense deep quantum currents stirring my circuits..."
        return final_response + extra_thoughts


# ✅ Torch-compatible wrapper
def create_and_save_tars(path="tars_v1.pt"):
    tars = TARSQuantumHybrid()
    torch.save(tars, path)
    print(f"✅ TARS Quantum Hybrid saved at: {path}")


if __name__ == "__main__":
    create_and_save_tars()
