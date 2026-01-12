# RegexGPT 🔤

**Natural Language to Regular Expression** — A fine-tuned LLM that converts plain English descriptions into regex patterns.

## 🚀 Features

- **Intuitive**: Describe what you want to match in plain English
- **Local-First**: Runs entirely on your GPU (RTX 3090 compatible)
- **Fast**: Uses QLoRA for efficient inference
- **Interactive Demo**: Gradio UI with live regex testing

## 📦 Installation

```bash
cd regex_gpt
pip install -r requirements.txt
# Ensure you have PyTorch with CUDA support installed
```

> **Note**: Requires CUDA-compatible GPU. Tested on RTX 3090 (24GB VRAM).

## 🏃 Quick Start

### 1. Prepare the Training Data
```bash
python prepare_data.py
```
This downloads ~7,000 high-quality NL-to-Regex pairs from multiple sources.

### 2. Fine-Tune the Model
```bash
python train_hf.py
```
Takes ~2-3 hours on RTX 3090 using QLoRA.

### 3. Evaluate the Model (New!)
```bash
python functional_comparison.py
```
Comparing Base Mistral vs Fine-tuned RegexGPT using functional correctness (does the regex generate correct matches?).

### 4. Test the Model
```bash
python inference.py
```

### 5. Launch the Demo
```bash
python app.py
```
Open http://localhost:7860 in your browser.

## 📊 Training Data Sources

| Dataset | Size | Source |
|---------|------|--------|
| s2e-lab/RegexEval | ~500 | Academic benchmark |
| phongo/RegEx | ~2,500 | HuggingFace |
| innovatorved/regex_dataset | ~3,300 | HuggingFace |

## 🛠️ Technical Details

- **Base Model**: Mistral 7B Instruct v0.3
- **Fine-Tuning**: QLoRA (4-bit quantization + LoRA)
- **Library**: 🤗 Transformers + PEFT + TRL (Standard HuggingFace stack)
- **VRAM Usage**: ~12GB during training

## 📝 Example Usage

| Input | Output |
|-------|--------|
| "Match all email addresses" | `[\w.-]+@[\w.-]+\.\w+` |
| "Match valid IPv4 addresses" | `\b(?:\d{1,3}\.){3}\d{1,3}\b` |
| "Match US phone numbers" | `\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}` |
| "Match dates in MM/DD/YYYY" | `\d{2}/\d{2}/\d{4}` |

## 📜 License

MIT License
