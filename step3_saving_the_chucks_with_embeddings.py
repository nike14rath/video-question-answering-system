# this is going to create embeddings related to the each text line of each file and store it into the jsons

import requests
import os
import json
import time
import pandas as pd


# CONFIG

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "bge-m3"
BATCH_SIZE = 16


# 1. Validate text

# reducing the length of the text so that
# only a optimun length sentence remain in the list of the sentences 
# to avoid crashing of the model 


def is_valid_text(text, min_len=15, max_len=1800):
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < min_len:
        return False
    if len(text) > max_len:
        return False
    return True



# 2. Embed SINGLE text (fallback-safe)

def embed_single(text):
    r = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "input": text},
        timeout=120
    )
    data = r.json()
    if "error" in data:
        return None
    return data["embeddings"][0]



# 3. Embed chunks with batch + fallback

def embed_chunks_safe(chunks):
    embedded_chunks = []

    # Keep only valid chunks
    valid_chunks = [c for c in chunks if is_valid_text(c.get("text", ""))]

    for i in range(0, len(valid_chunks), BATCH_SIZE):
        batch_chunks = valid_chunks[i:i + BATCH_SIZE]
        batch_texts = [c["text"].strip() for c in batch_chunks]

        r = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "input": batch_texts},
            timeout=120
        )

        data = r.json()

        # ✅ Batch success
        if "embeddings" in data:
            for chunk, emb in zip(batch_chunks, data["embeddings"]):
                ordered_chunk = {
                    "Video": chunk.get("Video"),
                    "start": chunk.get("start"),
                    "end": chunk.get("end"),
                    "text": chunk.get("text"),
                    "embedding": emb
                }
                embedded_chunks.append(ordered_chunk)
            continue

        # ❌ Batch failed then fallback to individusal chunk
        print("⚠️ Batch failed, retrying individually...")

        for chunk in batch_chunks:
            emb = embed_single(chunk["text"])
            if emb is None:
                print("❌ Skipping problematic chunk")
                continue

            ordered_chunk = {
                "Video": chunk.get("Video"),
                "start": chunk.get("start"),
                "end": chunk.get("end"),
                "text": chunk.get("text"),
                "embedding": emb
            }
            embedded_chunks.append(ordered_chunk)

        time.sleep(0.1)

    return embedded_chunks



# 4. Process all JSON files

json_folder = "json"
json_files = os.listdir(json_folder)
print(json_files)
my_dicts = []

for file in json_files:
    file_path = os.path.join(json_folder, file)

    with open(file_path, "r", encoding="utf-8") as f:
    # with open('json/sample.mp3.json', "r", encoding="utf-8") as f:
        content = json.load(f)

    print(f"\nCreating embeddings for: {file}")
    chunks = content["chunks"]

    print(f"  Raw chunks      : {len(chunks)}")

    embedded_chunks = embed_chunks_safe(chunks)

    print(f"  Embedded chunks : {len(embedded_chunks)}")

    # Save embeddings back into the same JSON
    content["chunks"] = embedded_chunks

    with open(f'jsons/{file}', "w", encoding="utf-8") as f:
        json.dump(content, f)

    print("  ✅ Embeddings saved successfully")


