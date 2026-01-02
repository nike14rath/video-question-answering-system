import os
import whisper
import torch
import time 
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

start = time.time()

model = whisper.load_model("medium").to(device)

audios = os.listdir("audios")
# print(audios)



# for acessing all the files from the directory
for audio in audios:
    

    result = model.transcribe(audio= f"audios/{audio}",
    # result = model.transcribe(audio= "audios/sample.mp3",
                          language = "hi",
                          task = "translate",
                          fp16=True)

    # print(result["text"])

    # let's  dump the file in the file here

    # the problem is that we want only the start time, end time, and text that's all 
    chunks = []
    for segment in result['segments']:
        chunks.append({"Video": audio, "start" : segment['start'], 'end': segment['end'], 'text': segment['text']})

    
    # let's also dump the complete text here
    
    chunks_with_meta_data = {"chunks" : chunks, 
                            "text": result['text']}
    # let's not dump the whole file content in one while
    with open(f"Json/{audio}.json", 'w') as f:
        json.dump(chunks_with_meta_data, f)
        
    print(f"File {audio} has been sucessfully Created.")
    

    
end = time.time()

print("Total time it took:", (end - start) / 60)



# ffmpeg -i "10_Video, Audio & Media in HTML .mp3" -t 10 sample.mp3
