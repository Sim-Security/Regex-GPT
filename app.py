"""
app.py - Gradio Demo for RegexGPT

A simple web UI to test and demonstrate the RegexGPT model.
Can be deployed to HuggingFace Spaces.
"""

import re
import gradio as gr
import torch
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# Configuration
# ============================================================

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_PATH = "./regex_gpt_lora"
MAX_SEQ_LENGTH = 512

# Global model (loaded once)
model = None
tokenizer = None


def load_model():
    """Load the model (called once on startup)"""
    global model, tokenizer
    
    if model is None:
        print("Loading RegexGPT model...")
        
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load adapters
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        print("Model loaded!")
    
    return model, tokenizer


def generate_regex(description: str) -> tuple[str, str]:
    """Generate regex from description and return (regex, test_result)"""
    
    if not description.strip():
        return "", "Please enter a description."
    
    model, tokenizer = load_model()
    
    prompt = f"""### Instruction:
Convert this natural language description into a regular expression.

### Input:
{description}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:" in full_response:
        regex = full_response.split("### Response:")[-1].strip()
        regex = regex.split("\n")[0].strip()
    else:
        regex = full_response
    
    return regex, ""


def test_regex(pattern: str, test_string: str) -> str:
    """Test a regex pattern against a test string"""
    
    if not pattern or not test_string:
        return ""
    
    try:
        compiled = re.compile(pattern)
        matches = compiled.findall(test_string)
        
        if matches:
            return f"✅ Found {len(matches)} match(es):\n" + "\n".join(f"  • {m}" for m in matches[:10])
        else:
            return "❌ No matches found."
    except re.error as e:
        return f"⚠️ Invalid regex: {e}"


# ============================================================
# Gradio Interface
# ============================================================

with gr.Blocks(title="RegexGPT", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔤 RegexGPT
    ### Natural Language to Regular Expression
    
    Describe what you want to match in plain English, and RegexGPT will generate the regex for you.
    
    *Powered by fine-tuned Mistral 7B with QLoRA*
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input
            description_input = gr.Textbox(
                label="📝 Describe the pattern",
                placeholder="e.g., Match all email addresses ending in .edu",
                lines=2,
            )
            
            generate_btn = gr.Button("🚀 Generate Regex", variant="primary")
            
            # Output
            regex_output = gr.Textbox(
                label="🔤 Generated Regex",
                lines=1,
                interactive=True,
            )
        
        with gr.Column(scale=2):
            # Test area
            test_input = gr.Textbox(
                label="🧪 Test String",
                placeholder="Paste text here to test the regex...",
                lines=3,
            )
            
            test_btn = gr.Button("Test Regex")
            
            test_result = gr.Textbox(
                label="📊 Test Results",
                lines=3,
            )
    
    # Examples
    gr.Examples(
        examples=[
            ["Match all email addresses"],
            ["Match valid IPv4 addresses"],
            ["Match US phone numbers"],
            ["Match dates in MM/DD/YYYY format"],
            ["Match URLs starting with http or https"],
            ["Match hex color codes like #FF5733"],
            ["Match words that start with a capital letter"],
            ["Match strings containing only digits"],
        ],
        inputs=description_input,
    )
    
    # Event handlers
    generate_btn.click(
        fn=generate_regex,
        inputs=description_input,
        outputs=[regex_output, test_result],
    )
    
    test_btn.click(
        fn=test_regex,
        inputs=[regex_output, test_input],
        outputs=test_result,
    )
    
    # Also generate on Enter
    description_input.submit(
        fn=generate_regex,
        inputs=description_input,
        outputs=[regex_output, test_result],
    )
    
    gr.Markdown("""
    ---
    **Tips:**
    - Be specific about formats (e.g., "MM/DD/YYYY" instead of just "dates")
    - Mention any special requirements (e.g., "case-insensitive", "multiline")
    - The generated regex uses Python/PCRE syntax
    """)


if __name__ == "__main__":
    import os

    # Load model on startup
    load_model()

    # Get port from environment (Cloud Run sets PORT=8080)
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))

    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,  # Set to True for public URL
    )
