# 🤗 Hugging Face Model Cache Viewer

A Gradio web interface for managing and viewing your local Hugging Face model cache. This tool helps you visualize and track the disk space usage of your downloaded models.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/hf-model-manage-gradio.git
cd hf-model-manage-gradio
```

2. Install the required dependencies:
```bash
pip install gradio pandas
```

## Environment Variables
- Only tested on MacOS
- `HF_HOME`: Set this to customize your Hugging Face cache directory location. By default, it uses `~/.cache/huggingface`