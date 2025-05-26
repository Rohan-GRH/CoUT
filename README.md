# Chain of Unconscious Thought (CoUT)

## Overview

Large Reasoning Models (LRMs) achieve promising performance but compromise token efficiency due to verbose reasoning processes. **Unconscious Thought Theory (UTT)** posits that complex problems can be solved more efficiently through internalized cognitive processes.

Inspired by UTT, we propose a new reasoning paradigm, termed **Chain of Unconscious Thought (CoUT)**, to improve the token efficiency of LRMs by guiding them to mimic human unconscious thought and internalize reasoning processes.

Concretely, we first prompt the model to internalize the reasoning by thinking in the hidden layer. Then, we design a bag of token-efficient strategies to further help models reduce unnecessary tokens yet preserve the performance. Our work reveals that models may possess beneficial unconscious thought, enabling improved efficiency without sacrificing performance.

**Extensive experiments demonstrate the effectiveness of CoUT. Remarkably, it surpasses CoT by reducing token usage by 47.62% while maintaining comparable accuracy.**

## 🚀 Key Features

- **Token Efficiency**: Reduces token usage by up to 47.62% compared to Chain-of-Thought (CoT)
- **Maintained Performance**: Preserves accuracy while significantly improving efficiency
- **Multiple Datasets**: Supports evaluation on 4 mathematical reasoning datasets
- **Flexible Baselines**: Easy modification of baseline prompts for comparative studies
- **Zero-Shot Approach**: Implements zero-shot methodology for practical real-world scenarios

## 📊 Supported Datasets

Our framework supports evaluation on four mathematical reasoning datasets:

1. **GSM8K**: Grade School Math 8K - Elementary mathematical word problems
2. **AQUA**: Algebraic Question Answering - Multiple choice algebraic reasoning
3. **SVAMP**: Simple Variations on Arithmetic Math word Problems
4. **MathQA**: Mathematical reasoning with multiple choice questions

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for GPT models)
- Anthropic API key (for Claude models)

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/Rohan-GRH/CoUT.git
cd chain-of-draft
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys:
```bash
# For OpenAI models
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic models  
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## 📋 Requirements

The project requires the following Python packages:

- `anthropic==0.42.0` - Anthropic Claude API client
- `datasets==3.2.0` - Hugging Face datasets library
- `names-dataset==3.1.0` - Name datasets for evaluation
- `openai==1.59.4` - OpenAI API client
- `PyYAML==6.0.2` - YAML configuration file parsing
- `tqdm==4.67.1` - Progress bars for long-running processes

## 🎯 Usage

### Basic Evaluation

Run evaluation on a specific dataset with CoUT method:

```bash
python evaluate.py --task gsm8k --model gpt-4o --prompt CoUT --shot 0
```

### Available Options

- **Tasks**: `gsm8k`, `aqua`, `svamp`, `mathqa`
- **Models**: `gpt-4o`, `sonnet`, `o3-mini`, `qwen-qwq-32b`
- **Prompting Methods**: `baseline`, `cot`, `cod`, `CoUT`, `tale_ep`
- **Shot**: Always use `0` (zero-shot approach)

### Example Commands

```bash

python evaluate.py --task gsm8k --model gpt-4o --prompt CoUT --shot 0


python evaluate.py --task gsm8k --model sonnet --prompt cot --shot 0 --max_samples 100 --api-key

python evaluate.py --task mathqa --model o3-mini --prompt tale_ep --shot 0 --max_samples max --url https://api.openai.com/v1/ --api-key



## 🔧 Customizing Prompts

You can easily modify baseline prompts by editing the configuration files in the `configs/` directory:

### Configuration Structure

Each dataset has separate configuration files for different methods:
- `{dataset}_baseline.yaml` - Basic prompting
- `{dataset}_cot.yaml` - Chain-of-Thought prompting  
- `{dataset}_cod.yaml` - Chain-of-Draft prompting
- `{dataset}_CoUT.yaml` - Our Chain of Unconscious Thought method
- `{dataset}_tale_ep.yaml` - TALE-EP method

### Example Configuration

```yaml
# configs/gsm8k_CoUT.yaml
system_prompt: |
  TOKEN CONSERVATION MODE ACTIVE. Use symbols/abbreviations when clear (e.g., &, w/, =, →). 
  Omit articles (a, an, the) when meaning remains clear. Strip all non-essential words 
  including greetings, acknowledgments, and explanations. Each saved token equals +1 
  efficiency point while each accuracy error costs -100 efficiency points. Focus 
  exclusively on maximum precision with minimum verbosity.
  Return the answer at the end of the response after a separator ####.

format: |
  Q: {question}
  A: {answer}

fewshot: 
  - question: "Sample question"
    answer: "Sample answer"
```

## 📈 Results Analysis

The framework automatically generates comprehensive results:

### Output Files

Results are saved in `./results/{date}/{task}/` directory:
- `{timestamp}_{task}-{model}-{prompt}.csv` - Summary statistics
- `{timestamp}_{task}-{model}-{prompt}_detailed.json` - Detailed results

### Metrics Reported

- **Accuracy**: Percentage of correct answers
- **Average Token Usage**: Mean tokens per query
- **Latency Statistics**: P90, P95, P99 latency percentiles
- **Detailed Per-Sample Results**: Individual question analysis

## 🎯 Zero-Shot Methodology

We adopt a **zero-shot approach** (`--shot 0`) for all evaluations because:

1. **Real-world Practicality**: Few-shot examples are rarely available in practical applications
2. **Fair Comparison**: Eliminates bias from example selection
3. **Efficiency Focus**: Aligns with our goal of reducing token usage
4. **Generalization**: Tests true model reasoning capabilities



## 📝 Project Structure

```
chain-of-draft/
├── configs/                 # Configuration files for different methods
│   ├── {dataset}_{method}.yaml
├── tasks/                   # Dataset-specific task implementations
│   ├── base.py             # Base task class
│   ├── gsm8k.py            # GSM8K dataset handler
│   ├── aqua.py             # AQUA dataset handler
│   ├── svamp.py            # SVAMP dataset handler
│   └── mathqa.py           # MathQA dataset handler
├── results/                # Generated result files
├── evaluate.py             # Main evaluation script
├── llm_client.py           # LLM API client
├── utils.py                # Utility functions
└── requirements.txt        # Python dependencies
```

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Report bugs or issues
2. Suggest new features or improvements
3. Submit pull requests with enhancements
4. Add support for new datasets or models

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.



## 🙏 Acknowledgments

- Inspired by Unconscious Thought Theory (UTT) from cognitive psychology
- Built upon the Chain-of-Thought reasoning paradigm
- Thanks to the open-source community for the foundational tools and datasets

---

**Note**: This framework is designed for research purposes and requires appropriate API keys for the language models. Please ensure you comply with the respective API usage terms and conditions.
