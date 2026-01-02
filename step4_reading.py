# Reading all the embeddings of the files as dataframe and saving it as joblib
import os
import json
import pandas as pd
import joblib

all_chunks = []

json_folder = "jsons"
json_files = os.listdir(json_folder)

for file in json_files:
    file_path = os.path.join(json_folder, file)

    with open(file_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    for chunk in content["chunks"]:
        # Optional: keep track of source file
        # chunk["source_file"] = file
        all_chunks.append(chunk)

# Create ONE DataFrame
df = pd.DataFrame.from_records(all_chunks)

# saving the the df
joblib.dump(df, "Dataframe_with_embeddings.joblib")
print("✅ File Created Sucessfully!!")

print(df.tail())
print("\nTotal chunks:", len(df))






