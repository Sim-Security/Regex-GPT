"""
prepare_data.py - Download and prepare NL-to-Regex datasets

This script:
1. Downloads datasets from HuggingFace and GitHub
2. Normalizes them into a consistent format
3. Splits into train/validation sets
4. Saves as JSONL files for training
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Output directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def load_phongo_regex():
    """
    Load phongo/RegEx dataset from HuggingFace
    
    This dataset already has the correct format:
    - instruction: The task instruction
    - input: Natural language description  
    - output: The regex pattern
    - text: Full formatted example (we don't need this)
    """
    print("Loading phongo/RegEx dataset...")
    try:
        ds = load_dataset("phongo/RegEx", split="train")
        examples = []
        for row in ds:
            # Use existing columns directly
            nl_desc = row.get("input", "")
            regex = row.get("output", "")
            
            if nl_desc and regex:
                examples.append({
                    "instruction": "Convert this natural language description into a regular expression.",
                    "input": nl_desc.strip(),
                    "output": regex.strip()
                })
        print(f"  Loaded {len(examples)} examples from phongo/RegEx")
        return examples
    except Exception as e:
        print(f"  Warning: Could not load phongo/RegEx: {e}")
        return []


def load_regexeval():
    """
    Load s2e-lab/RegexEval dataset from HuggingFace
    
    This dataset has:
    - expression: the regex pattern
    - prompt: refined natural language description
    - raw_prompt: original description
    - matches/non_matches: test cases (not used for training)
    """
    print("Loading s2e-lab/RegexEval dataset...")
    try:
        ds = load_dataset("s2e-lab/RegexEval", split="train")
        examples = []
        for row in ds:
            # Use the refined prompt for better quality
            nl_desc = row.get("prompt", row.get("raw_prompt", ""))
            regex = row.get("expression", "")
            
            if nl_desc and regex:
                examples.append({
                    "instruction": "Convert this natural language description into a regular expression.",
                    "input": nl_desc.strip(),
                    "output": regex.strip()
                })
        print(f"  Loaded {len(examples)} examples from s2e-lab/RegexEval")
        return examples
    except Exception as e:
        print(f"  Warning: Could not load s2e-lab/RegexEval: {e}")
        return []


def load_innovatorved_regex():
    """
    Load innovatorved/regex_dataset via raw JSON download
    (HuggingFace dataset loader has parsing issues)
    
    This dataset has:
    - title: short title/name
    - description: longer description (may be empty)
    - regex: the regex pattern
    """
    print("Loading innovatorved/regex_dataset from raw JSON...")
    import urllib.request
    
    try:
        url = 'https://huggingface.co/datasets/innovatorved/regex_dataset/resolve/main/data.json'
        data = json.loads(urllib.request.urlopen(url, timeout=30).read().decode('utf-8'))
        
        examples = []
        # Keywords that suggest a good natural language description
        good_keywords = ['match', 'find', 'validate', 'extract', 'check', 'detect', 
                        'pattern', 'number', 'email', 'phone', 'url', 'date', 'time',
                        'address', 'name', 'format', 'string', 'text', 'word', 'digit']
        
        for entry in data:
            # Prefer description, fall back to title
            desc = entry.get('description', '') or ''
            title = entry.get('title', '') or ''
            regex = entry.get('regex', '')
            
            # Use description if it's longer and meaningful, else use title
            nl_desc = desc if len(desc) > len(title) and len(desc) > 10 else title
            
            # Filter for quality: must have regex and reasonable description
            if regex and nl_desc and len(nl_desc) >= 12:
                # Skip entries where description IS the regex (self-referential)
                if nl_desc.strip() == regex.strip():
                    continue
                # Skip descriptions that don't look like natural language:
                # - Must have at least 2 words
                # - Must contain mostly ASCII letters
                words = nl_desc.split()
                if len(words) < 2:
                    continue
                # Check for reasonable letter ratio (filter gibberish)
                letters = sum(1 for c in nl_desc if c.isalpha())
                if letters < len(nl_desc) * 0.4:  # At least 40% letters
                    continue
                # Skip non-English entries (basic check)
                if not nl_desc.isascii():
                    continue
                    
                examples.append({
                    "instruction": "Convert this natural language description into a regular expression.",
                    "input": nl_desc.strip(),
                    "output": regex.strip()
                })
        
        print(f"  Loaded {len(examples)} examples from innovatorved/regex_dataset")
        return examples
    except Exception as e:
        print(f"  Warning: Could not load innovatorved/regex_dataset: {e}")
        return []


def load_softregex_data():
    """
    Load NL-RX data from SoftRegex repository (alternative source)
    Contains NL-RX-Turk dataset
    """
    print("Loading SoftRegex NL-RX data from GitHub...")
    import urllib.request
    
    examples = []
    base_url = "https://raw.githubusercontent.com/jacger2/softregex/master/data/NL-RX-Turk/"
    
    try:
        for split in ["train", "val", "test"]:
            nl_url = f"{base_url}src-{split}.txt"
            regex_url = f"{base_url}tgt-{split}.txt"
            
            try:
                with urllib.request.urlopen(nl_url, timeout=10) as response:
                    nl_lines = response.read().decode('utf-8').strip().split('\n')
                with urllib.request.urlopen(regex_url, timeout=10) as response:
                    regex_lines = response.read().decode('utf-8').strip().split('\n')
                
                for nl, regex in zip(nl_lines, regex_lines):
                    nl = nl.strip()
                    regex = regex.strip()
                    if nl and regex:
                        examples.append({
                            "instruction": "Convert this natural language description into a regular expression.",
                            "input": nl,
                            "output": regex
                        })
                print(f"    Loaded {split} split: {len(nl_lines)} examples")
            except Exception as e:
                print(f"    Could not load {split}: {e}")
        
        print(f"  Total from SoftRegex: {len(examples)} examples")
        return examples
    except Exception as e:
        print(f"  Warning: Could not load SoftRegex data: {e}")
        return []


def load_kb13_data():
    """
    Load KB13 dataset (smaller but high quality)
    """
    print("Loading KB13 dataset from SoftRegex GitHub...")
    import urllib.request
    
    examples = []
    base_url = "https://raw.githubusercontent.com/jacger2/softregex/master/data/KB13/"
    
    try:
        for split in ["train", "val", "test"]:
            nl_url = f"{base_url}src-{split}.txt"
            regex_url = f"{base_url}tgt-{split}.txt"
            
            try:
                with urllib.request.urlopen(nl_url, timeout=10) as response:
                    nl_lines = response.read().decode('utf-8').strip().split('\n')
                with urllib.request.urlopen(regex_url, timeout=10) as response:
                    regex_lines = response.read().decode('utf-8').strip().split('\n')
                
                for nl, regex in zip(nl_lines, regex_lines):
                    nl = nl.strip()
                    regex = regex.strip()
                    if nl and regex:
                        examples.append({
                            "instruction": "Convert this natural language description into a regular expression.",
                            "input": nl,
                            "output": regex
                        })
                print(f"    Loaded {split} split: {len(nl_lines)} examples")
            except Exception as e:
                print(f"    Could not load {split}: {e}")
        
        print(f"  Total from KB13: {len(examples)} examples")
        return examples
    except Exception as e:
        print(f"  Warning: Could not load KB13 data: {e}")
        return []


def load_nlrx_synth():
    """
    Load NL-RX-Synth (synthetic) dataset
    """
    print("Loading NL-RX-Synth from SoftRegex GitHub...")
    import urllib.request
    
    examples = []
    base_url = "https://raw.githubusercontent.com/jacger2/softregex/master/data/NL-RX-Synth/"
    
    try:
        for split in ["train", "val", "test"]:
            nl_url = f"{base_url}src-{split}.txt"
            regex_url = f"{base_url}tgt-{split}.txt"
            
            try:
                with urllib.request.urlopen(nl_url, timeout=10) as response:
                    nl_lines = response.read().decode('utf-8').strip().split('\n')
                with urllib.request.urlopen(regex_url, timeout=10) as response:
                    regex_lines = response.read().decode('utf-8').strip().split('\n')
                
                for nl, regex in zip(nl_lines, regex_lines):
                    nl = nl.strip()
                    regex = regex.strip()
                    if nl and regex:
                        examples.append({
                            "instruction": "Convert this natural language description into a regular expression.",
                            "input": nl,
                            "output": regex
                        })
                print(f"    Loaded {split} split: {len(nl_lines)} examples")
            except Exception as e:
                print(f"    Could not load {split}: {e}")
        
        print(f"  Total from NL-RX-Synth: {len(examples)} examples")
        return examples
    except Exception as e:
        print(f"  Warning: Could not load NL-RX-Synth data: {e}")
        return []


def deduplicate(examples):
    """Remove duplicate examples based on input text"""
    seen = set()
    unique = []
    for ex in examples:
        key = ex["input"].lower().strip()
        if key not in seen and key:
            seen.add(key)
            unique.append(ex)
    return unique


def format_for_training(example):
    """Format example as a chat/instruction template"""
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    }


def main():
    print("=" * 60)
    print("RegexGPT Data Preparation")
    print("=" * 60)
    
    # Load all datasets
    all_examples = []
    
    # HuggingFace datasets
    all_examples.extend(load_phongo_regex())
    all_examples.extend(load_regexeval())
    all_examples.extend(load_innovatorved_regex())
    
    # GitHub datasets (SoftRegex repository - alternative to dead deep-regex)
    all_examples.extend(load_softregex_data())
    all_examples.extend(load_kb13_data())
    all_examples.extend(load_nlrx_synth())
    
    print(f"\nTotal examples before deduplication: {len(all_examples)}")
    
    # Deduplicate
    all_examples = deduplicate(all_examples)
    print(f"Total examples after deduplication: {len(all_examples)}")
    
    # Filter out empty examples
    all_examples = [ex for ex in all_examples if ex["input"] and ex["output"]]
    print(f"Total examples after filtering empty: {len(all_examples)}")
    
    if len(all_examples) == 0:
        print("\n❌ ERROR: No examples loaded! Check network connectivity and dataset sources.")
        return
    
    # Split into train/validation (90/10)
    import random
    random.seed(42)
    random.shuffle(all_examples)
    
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    print(f"\nTrain set: {len(train_examples)} examples")
    print(f"Validation set: {len(val_examples)} examples")
    
    # Format for training
    train_formatted = [format_for_training(ex) for ex in train_examples]
    val_formatted = [format_for_training(ex) for ex in val_examples]
    
    # Save as JSONL
    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_formatted:
            f.write(json.dumps(ex) + "\n")
    
    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_formatted:
            f.write(json.dumps(ex) + "\n")
    
    print(f"\n✅ Saved training data to {train_path}")
    print(f"✅ Saved validation data to {val_path}")
    
    # Show a sample
    if train_formatted:
        print("\n" + "=" * 60)
        print("Sample training example:")
        print("=" * 60)
        print(train_formatted[0]["text"])


if __name__ == "__main__":
    main()
