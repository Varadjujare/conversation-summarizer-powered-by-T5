# Hugging Face Text Summarizer App

A premium, modern web application that summarizes dialogue and text using a fine-tuned Hugging Face **T5 (Text-to-Text Transfer Transformer)** model, served via a high-performance **FastAPI** backend with a responsive, glassmorphic dark-theme user interface.

---

## 🚀 Features

- **Fine-tuned T5 Transformer:** Uses a specialized version of the T5 model optimized for dialogue summarization.
- **Fast & Lightweight Backend:** Built with FastAPI, utilizing async handlers and PyTorch with hardware acceleration detection (CUDA, MPS, or CPU).
- **Premium Glassmorphic UI:** A sleek, fully-responsive dark-themed user interface featuring micro-animations, loading states, and error handling.
- **Robust Text Cleaning:** Auto-cleans line breaks, multiple spaces, and HTML tags from inputs before generating summaries.
- **RESTful API Endpoint:** Exposes a clean POST endpoint for programmatic text summarization.

---

## 🛠️ Tech Stack

- **Frontend:** HTML5, CSS3 (Custom design system), Vanilla JavaScript (ES6+)
- **Backend:** FastAPI (Python), Jinja2 (HTML rendering)
- **Deep Learning:** PyTorch, Hugging Face Transformers (`transformers` library)
- **Pre-trained Model:** T5-Small (`t5-small` tokenizer with custom fine-tuned weights)

---

## 📂 Project Structure

```text
TEXTSUMMARIZERAPP/
├── saved_summary_model/       # Pre-trained fine-tuned model checkpoint
│   ├── config.json            # Model architecture configuration
│   ├── generation_config.json # Settings for summary generation (max_length, beams, etc.)
│   ├── model.safetensors      # Weights of the T5 model
│   ├── tokenizer.json         # Tokenizer vocabulary configuration
│   └── tokenizer_config.json  # Tokenizer settings
├── Text_summariser/           # Training notebooks & datasets
│   ├── Text_Summarizer_Commented.ipynb
│   ├── text_summarizer.ipynb
│   └── *.csv                  # SAMSum dataset splits (Train/Val/Test)
├── app.py                     # FastAPI backend application
├── index.html                 # Frontend user interface (styled with embedded CSS/JS)
├── README.md                  # Project documentation
└── requirements.txt           # Project dependencies
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd TEXTSUMMARIZERAPP
```

### 2. Set up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Make sure you have `pip` updated and run:
```bash
pip install fastapi uvicorn torch transformers jinja2 pydantic
```

---

## 🖥️ Running the Application

Start the FastAPI application using Uvicorn (using `python -m` ensures compatibility if Uvicorn is not in your system PATH):

```bash
python -m uvicorn app:app --reload
```

Once running, open your web browser and navigate to:
- **Web App UI:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🔌 API Documentation

### **Summarize Text / Dialogue**
Generates a condensed summary of the provided text input.

- **URL:** `/summarize/`
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
  ```json
  {
    "dialogue": "Insert your long dialogue or article content here..."
  }
  ```
- **Response Body:**
  ```json
  {
    "summary": "The summarized version of the dialogue..."
  }
  ```

---

## 🧠 Model Configuration Details

The summarization pipeline uses the following hyperparameter settings for text generation inside `app.py`:
- `max_length`: 200 (Limits the maximum summary size)
- `min_length`: 60 (Forces the model to produce meaningful, detailed sentences)
- `num_beams`: 4 (Beam search width for higher-quality generation)
- `length_penalty`: 2.0 (Encourages longer, more descriptive summaries)
- `early_stopping`: True
