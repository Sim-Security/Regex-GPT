"""
inference.py - Test the fine-tuned RegexGPT model

Run this after training to test the model's regex generation capabilities.
"""

import torch
from unsloth import FastLanguageModel

# Configuration
MODEL_PATH = "./regex_gpt_lora"
MAX_SEQ_LENGTH = 512


def load_model():
    """Load the fine-tuned model"""
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Enable fast inference mode
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer


def generate_regex(model, tokenizer, description: str) -> str:
    """Generate a regex from a natural language description"""
    
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
            temperature=0.1,        # Low temperature for deterministic output
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the regex (after "### Response:")
    if "### Response:" in full_response:
        regex = full_response.split("### Response:")[-1].strip()
        # Take only the first line (the regex)
        regex = regex.split("\n")[0].strip()
        return regex
    
    return full_response


def main():
    print("=" * 60)
    print("RegexGPT Inference")
    print("=" * 60)
    
    model, tokenizer = load_model()
    
    # Test cases
    test_descriptions = [
        "Match all email addresses",
        "Match valid IPv4 addresses",
        "Match US phone numbers in format (XXX) XXX-XXXX",
        "Match dates in format MM/DD/YYYY",
        "Match URLs starting with http or https",
        "Match all words that start with a capital letter",
        "Match strings that contain only digits",
        "Match hex color codes like #FF5733",
    ]
    
    print("\nGenerating regex patterns...\n")
    print("-" * 60)
    
    for desc in test_descriptions:
        regex = generate_regex(model, tokenizer, desc)
        print(f"📝 {desc}")
        print(f"🔤 {regex}")
        print("-" * 60)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n📝 Describe the pattern: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            if not user_input:
                continue
            
            regex = generate_regex(model, tokenizer, user_input)
            print(f"🔤 Regex: {regex}")
            
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
