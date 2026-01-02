🎥 Video Question Answering System (Streamlit + Ollama)

This project builds an end-to-end Video Question Answering (Video-QA) system that allows users to ask questions about tutorial videos and receive precise answers with video names and timestamps.

The system works completely offline using Ollama, Whisper, and Streamlit.

🚀 Features

🎧 Convert videos to audio automatically

📝 Transcribe & translate audio using Whisper

✂️ Chunk transcripts with timestamps

🧠 Generate semantic embeddings (bge-m3)

🔍 Perform cosine similarity search

🤖 Answer queries using an LLM (LLaMA)

🖥️ Interactive Streamlit web interface

📍 Timestamp-based video guidance

📂 Project Structure
.
├── app.py                          # Streamlit web application
├── step1_process_vedio.py          # Video → Audio conversion
├── step2_creating_chunks.py        # Transcription & chunk creation
├── step3_saving_the_chucks_with_embeddings.py
├── step4_reading.py                # Create embedding DataFrame
├── step5_incoming_query.py         # CLI-based QA (optional)
│
├── Vedios/                         # Input video files
├── audios/                         # Extracted audio files
├── Json/                           # Transcription JSON files
├── jsons/                          # JSON files with embeddings
├── Dataframe_with_embeddings.joblib
├── prompt.txt                      # Prompt sent to LLM
└── response.txt                    # LLM response output

🧩 Pipeline Overview
Videos
  ↓
Audio Extraction (FFmpeg)
  ↓
Transcription & Translation (Whisper)
  ↓
Timestamped Text Chunks
  ↓
Embeddings (Ollama - bge-m3)
  ↓
Similarity Search
  ↓
LLM Answer Generation
  ↓
Streamlit UI Output

🛠️ Step-by-Step File Explanation
🔹 step1_process_vedio.py

Purpose: Convert videos into audio files

Reads videos from Vedios/

Extracts tutorial number & title

Converts videos to .mp3 using FFmpeg

🔹 step2_creating_chunks.py

Purpose: Transcribe and chunk audio

Uses Whisper (medium model)

Translates Hindi → English

Stores text with start & end timestamps

Outputs structured JSON files

🔹 step3_saving_the_chucks_with_embeddings.py

Purpose: Generate embeddings

Uses Ollama embedding API (bge-m3)

Batch embedding with fallback mechanism

Filters invalid or very short text

Saves embeddings into JSON

🔹 step4_reading.py

Purpose: Create a unified vector store

Loads all embedded JSON files

Combines chunks into a Pandas DataFrame

Saves as Dataframe_with_embeddings.joblib

🔹 step5_incoming_query.py (Optional)

Purpose: CLI-based question answering

Takes user input

Finds relevant video chunks

Generates LLM answer

Saves prompt & response to files

🔹 app.py (Main Application)

Purpose: Streamlit web interface

Accepts user questions

Computes semantic similarity

Retrieves top video matches

Generates LLM-based answers

Displays results in a clean UI

📦 Requirements
System

Python 3.9+

FFmpeg

Ollama (running locally)

GPU recommended (for Whisper)

Python Dependencies
pip install streamlit joblib numpy pandas scikit-learn requests torch whisper

🧠 Ollama Setup

Start Ollama:

ollama serve


Pull required models:

ollama pull bge-m3
ollama pull llama3.2

▶️ How to Run the Project
1️⃣ Prepare the Data (run once)
python step1_process_vedio.py
python step2_creating_chunks.py
python step3_saving_the_chucks_with_embeddings.py
python step4_reading.py

2️⃣ Launch the Streamlit App
python -m streamlit run app.py


Open in browser:

http://localhost:8501

🎯 Usage Example

Enter a question like:

What are semantic HTML tags?


The app will:

Search related video segments

Identify exact timestamps

Generate a clear, guided answer

🔐 Notes & Limitations

Answers are strictly based on video content

Unrelated questions are rejected

Requires Ollama to be running

Embeddings must be regenerated if videos change

🌱 Future Improvements

FAISS for faster search

Clickable video timestamps

Multi-video filtering

Chat history memory

Deployment on cloud / Hugging Face

Better UI & analytics

🙌 Acknowledgements

OpenAI Whisper

Ollama

Streamlit

Scikit-learn

