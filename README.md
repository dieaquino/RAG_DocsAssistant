# Zurich Policy Manager Assistant 🤖

An advanced AI assistant for managing insurance policies, built with a **Retrieval-Augmented Generation (RAG)** architecture that runs **100% locally**.

---

## 🎯 Main Objective

This system allows users to interact with insurance policy documents (PDFs) in natural language. The assistant extracts relevant information, answers specific questions, and **always cites the original source**, ensuring reliable and auditable responses.

---

## 🚀 Installation & Execution Guide

Follow these steps to run the assistant on your local machine.

---

### 1. Prerequisites

Make sure the following are installed:

- **Conda / Miniconda**  
  For managing Python environments. Download from:  
  👉 https://docs.conda.io/projects/miniconda/en/latest/

- **Ollama**  
  To run local language models. Install from:  
  👉 https://ollama.ai

  After installation, open your terminal and run:

  ```bash
  # (Optional) Install Ollama
  curl -fsSL https://ollama.ai/install.sh | sh

  # Download the required models
  ollama pull mistral:7b
  ollama pull llama2:7b

  # Start the Ollama server
  ollama serve
  ```
### 2. Project Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/dieaquino/RAG_DocsAssistant.git
   cd zurich-policy-assistant
   ```

   Or create the folder structure and files manually if you're starting from scratch.

2. **Create the Conda environment:**

   ```bash
   conda create --name zurich-assistant python=3.11 -y
   conda activate zurich-assistant
   ```

3. **Install dependencies:**

   Make sure `requirements.txt` is in the root directory, then run:

   ```bash
   pip install -r requirements.txt
   ```

---

### 3. Prepare Your Documents

1. Ensure the following folder exists:

   ```
   data/docs/
   ```

2. Place your policy PDF file inside that folder, e.g.:

   ```
   data/docs/Group_Policy_S655.pdf
   ```

   ⚠️ If the document database is empty, the system will automatically process the first PDF found in that folder.

---

### 4. Run the Assistant

You can launch the assistant in two ways:

#### A) Web Demo (Graphical Interface)

Ideal for a full user experience via your browser:

```bash
streamlit run ui/streamlit_app.py --server.port=8501
```

This will open the assistant interface in your default web browser.  
From there, you can load policies, ask questions, and receive answers with references.

---

#### B) Console Demo (Quick Test)

Ideal for rapid testing without a GUI:

```bash
python simple_demo.py
```

This mode loads the first available PDF and allows interaction directly from the terminal using natural language.

---

## ✅ Final Checklist

Before running the assistant, ensure that:

- ✅ Ollama server is running (`ollama serve`)
- ✅ Conda environment is activated (`conda activate zurich-assistant`)
- ✅ Your documents are placed in `data/docs/`

---
