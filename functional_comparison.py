"""
functional_comparison.py - Compare Base vs Fine-tuned on Functional Eval

Tests both models on the same regex samples to measure improvement.
"""

import json
import re
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configuration
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
FINETUNED_PATH = "./regex_gpt_lora"
MAX_SAMPLES = 50


def load_base_model():
    """Load base Mistral model (no fine-tuning)"""
    print("Loading BASE Mistral model (no fine-tuning)...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def load_finetuned_model():
    """Load fine-tuned RegexGPT model"""
    print("Loading FINE-TUNED RegexGPT model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = PeftModel.from_pretrained(base_model, FINETUNED_PATH)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_regex(model, tokenizer, description: str) -> str:
    """Generate regex from natural language"""
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
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:" in response:
        regex = response.split("### Response:")[-1].strip()
        regex = regex.split("\n")[0].strip()
        return regex
    return response.strip()


def test_regex_functional(pattern: str, matches: list, non_matches: list) -> dict:
    """Test if regex produces correct matches."""
    result = {
        "valid": False,
        "matches_correct": 0,
        "matches_total": len(matches) if matches else 0,
        "non_matches_correct": 0,
        "non_matches_total": len(non_matches) if non_matches else 0,
        "match_accuracy": 0.0,
        "non_match_accuracy": 0.0,
        "overall_accuracy": 0.0,
    }
    
    try:
        compiled = re.compile(pattern)
        result["valid"] = True
    except re.error:
        return result
    
    if matches:
        for test_str in matches:
            try:
                if compiled.search(str(test_str)):
                    result["matches_correct"] += 1
            except:
                pass
        result["match_accuracy"] = result["matches_correct"] / result["matches_total"] * 100
    
    if non_matches:
        for test_str in non_matches:
            try:
                if not compiled.search(str(test_str)):
                    result["non_matches_correct"] += 1
            except:
                pass
        result["non_match_accuracy"] = result["non_matches_correct"] / result["non_matches_total"] * 100
    
    total_correct = result["matches_correct"] + result["non_matches_correct"]
    total_tests = result["matches_total"] + result["non_matches_total"]
    if total_tests > 0:
        result["overall_accuracy"] = total_correct / total_tests * 100
    
    return result


def evaluate_model(name: str, model, tokenizer, samples: list) -> dict:
    """Evaluate a model on functional tests."""
    print(f"\nEvaluating: {name}")
    print("=" * 50)
    
    total_match_acc = 0
    total_non_match_acc = 0
    total_overall_acc = 0
    valid_count = 0
    perfect_count = 0
    
    for row in tqdm(samples, desc=name):
        description = row.get("prompt", row.get("raw_prompt", ""))
        matches = row.get("matches", [])
        non_matches = row.get("non_matches", [])
        
        generated = generate_regex(model, tokenizer, description)
        result = test_regex_functional(generated, matches, non_matches)
        
        if result["valid"]:
            valid_count += 1
            total_match_acc += result["match_accuracy"]
            total_non_match_acc += result["non_match_accuracy"]
            total_overall_acc += result["overall_accuracy"]
            
            if result["overall_accuracy"] >= 99.9:
                perfect_count += 1
    
    n = len(samples)
    return {
        "model": name,
        "total": n,
        "valid_regex": valid_count,
        "valid_rate": valid_count / n * 100 if n > 0 else 0,
        "perfect_count": perfect_count,
        "perfect_rate": perfect_count / n * 100 if n > 0 else 0,
        "avg_match_acc": total_match_acc / valid_count if valid_count > 0 else 0,
        "avg_non_match_acc": total_non_match_acc / valid_count if valid_count > 0 else 0,
        "avg_overall_acc": total_overall_acc / valid_count if valid_count > 0 else 0,
    }


def main():
    print("=" * 60)
    print("Functional Evaluation: BASE vs FINE-TUNED")
    print("=" * 60)
    
    # Load test data
    print("\nLoading s2e-lab/RegexEval dataset...")
    ds = load_dataset("s2e-lab/RegexEval", split="train")
    
    samples = [row for row in ds if row.get("matches") or row.get("non_matches")]
    samples = samples[:MAX_SAMPLES]
    print(f"Using {len(samples)} samples with test cases")
    
    results = []
    
    # 1. Evaluate Base Model
    print("\n" + "=" * 60)
    print("1. BASE MODEL (Mistral 7B - no fine-tuning)")
    print("=" * 60)
    base_model, base_tokenizer = load_base_model()
    base_results = evaluate_model("Base Mistral 7B", base_model, base_tokenizer, samples)
    results.append(base_results)
    
    # Free memory
    del base_model
    torch.cuda.empty_cache()
    
    # 2. Evaluate Fine-tuned Model
    print("\n" + "=" * 60)
    print("2. FINE-TUNED MODEL (RegexGPT)")
    print("=" * 60)
    ft_model, ft_tokenizer = load_finetuned_model()
    ft_results = evaluate_model("RegexGPT (Fine-tuned)", ft_model, ft_tokenizer, samples)
    results.append(ft_results)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"\n{'Model':<25} {'Valid %':>10} {'Perfect %':>12} {'Overall Acc':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<25} {r['valid_rate']:>9.1f}% {r['perfect_rate']:>11.1f}% {r['avg_overall_acc']:>11.1f}%")
    
    # Show improvement
    if len(results) == 2:
        improvement = results[1]["avg_overall_acc"] - results[0]["avg_overall_acc"]
        perfect_improvement = results[1]["perfect_rate"] - results[0]["perfect_rate"]
        print("-" * 60)
        print(f"\n📈 IMPROVEMENT from fine-tuning:")
        print(f"   Overall accuracy: +{improvement:.1f}%")
        print(f"   Perfect patterns: +{perfect_improvement:.1f}%")
    
    # Save results
    with open("functional_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Results saved to functional_comparison_results.json")


if __name__ == "__main__":
    main()
