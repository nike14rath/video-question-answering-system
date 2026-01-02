# 🎥 Video Question Answering System (Streamlit + Ollama)

This project implements an **end-to-end Video Question Answering (Video-QA) system** that allows users to ask questions about tutorial videos and receive **precise answers with video names and timestamps**.

The entire system works **offline**, using **Whisper** for transcription, **Ollama** for embeddings and LLM inference, and **Streamlit** for the user interface.

---

## 🚀 Features

- 🎧 Automatically convert videos to audio
- 📝 Transcribe and translate audio using Whisper
- ✂️ Chunk transcripts with precise timestamps
- 🧠 Generate semantic embeddings using **bge-m3**
- 🔍 Perform similarity search over video content
- 🤖 Answer questions using an LLM (**LLaMA**)
- 🖥️ Interactive Streamlit web interface
- 📍 Timestamp-based video guidance

---

## 📂 Project Structure

```text
.
├── app.py                                   # Streamlit web application
├── step1_process_vedio.py                   # Video → Audio conversion
├── step2_creating_chunks.py                 # Transcription & chunk creation
├── step3_saving_the_chucks_with_embeddings.py
├── step4_reading.py                         # Create embedding DataFrame
├── step5_incoming_query.py                  # CLI-based QA (optional)
│
├── Vedios/                                  # Input video files
├── audios/                                  # Extracted audio files
├── Json/                                    # Transcription JSON files
├── jsons/                                   # JSON files with embeddings
├── Dataframe_with_embeddings.joblib         # Vector store
├── prompt.txt                               # Prompt sent to LLM
└── response.txt                             # LLM response output
```

---

## 🧩 Pipeline Overview

```text
Videos
  ↓
Audio Extraction (FFmpeg)
  ↓
Transcription & Translation (Whisper)
  ↓
Timestamped Text Chunks
  ↓
Embeddings (Ollama – bge-m3)
  ↓
Similarity Search
  ↓
LLM Answer Generation
  ↓
Streamlit UI Output
```

---

## 🛠️ Step-by-Step File Explanation

### 🔹 step1_process_vedio.py
**Purpose:** Convert videos into audio files

- Reads videos from the `Vedios/` directory
- Extracts tutorial number and title
- Converts videos to `.mp3` using FFmpeg

---

### 🔹 step2_creating_chunks.py
**Purpose:** Transcribe and chunk audio

- Uses Whisper (`medium` model)
- Translates Hindi → English
- Stores text with start and end timestamps
- Outputs structured JSON files

---

### 🔹 step3_saving_the_chucks_with_embeddings.py
**Purpose:** Generate embeddings

- Uses Ollama embedding API (`bge-m3`)
- Batch embedding with fallback mechanism
- Filters invalid or very short text chunks
- Saves embeddings into JSON files

---

### 🔹 step4_reading.py
**Purpose:** Create a unified vector store

- Loads all embedded JSON files
- Combines chunks into a single Pandas DataFrame
- Saves the DataFrame as `Dataframe_with_embeddings.joblib`

---

### 🔹 step5_incoming_query.py (Optional)
**Purpose:** CLI-based question answering

- Accepts user queries from terminal
- Finds relevant video chunks
- Generates LLM-based answers
- Saves prompt and response to files

---

### 🔹 app.py (Main Application)
**Purpose:** Streamlit web interface

- Accepts user questions
- Computes semantic similarity
- Retrieves top matching video segments
- Generates LLM-based answers
- Displays results in a clean, interactive UI

---

## 📦 Requirements

### System Requirements
- Python 3.9+
- FFmpeg
- Ollama (running locally)
- GPU recommended (for Whisper)

### Python Dependencies

```bash
pip install streamlit joblib numpy pandas scikit-learn requests torch whisper
```

---

## 🧠 Ollama Setup

```bash
ollama serve
ollama pull bge-m3
ollama pull llama3.2
```

---

## ▶️ How to Run the Project

### 1️⃣ Prepare the Data (run once)

```bash
python step1_process_vedio.py
python step2_creating_chunks.py
python step3_saving_the_chucks_with_embeddings.py
python step4_reading.py
```

### 2️⃣ Launch the Streamlit App

```bash
python -m streamlit run app.py
```

Open in your browser:

```text
http://localhost:8501
```

---

## 🎯 Usage Example

```text
Question:
What are semantic HTML tags?

Output:
• Video: HTML_Tutorial_11.mp3
• Timestamp: 04:18 – 06:42
• Explanation: Semantic tags define the meaning of HTML elements...
```

---

## 🔐 Notes & Limitations

- Answers are strictly based on video content
- Unrelated questions are rejected
- Ollama must be running locally
- Embeddings must be regenerated if videos change

---

## 🌱 Future Improvements

- FAISS integration for faster search
- Clickable video timestamps
- Multi-video filtering
- Chat history & conversational memory
- Cloud deployment
- Enhanced UI & analytics

---

## 🙌 Acknowledgements

- OpenAI Whisper
- Ollama
- Streamlit
- Scikit-learn
