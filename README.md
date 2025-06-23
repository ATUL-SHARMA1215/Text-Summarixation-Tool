# 🧠 AI Text Summarization Tool

An advanced yet user-friendly **Streamlit** app that summarizes long-form text into short, meaningful insights using Facebook’s **BART** model (`facebook/bart-large-cnn`).

![UI Preview](https://miro.medium.com/v2/resize:fit:1200/1*N_rpqtlvyepXP01EuxbGmw.jpeg)

---

## 🚀 Features

- ✍️ **Input Methods**: Type text or upload `.txt` files
- ⚙️ **Customizable Summary Length**: Set minimum and maximum limits
- 📊 **Summary Stats**: Original vs summarized word count and compression ratio
- 🎨 **Modern UI**: Beautiful dark-themed interface with animated buttons
- ⚡ **Fast & GPU-Ready**: Automatically detects CUDA for better performance

---

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) – Web framework
- [Transformers (Hugging Face)](https://huggingface.co/transformers/) – BART model
- [PyTorch](https://pytorch.org/) – Backend inference

---

## 📁 Supported Formats

- `.txt` file upload
- Direct text input through the browser

---

## 📦 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/ai-text-summarizer.git
   cd ai-text-summarizer
