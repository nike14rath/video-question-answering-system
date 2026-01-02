import subprocess  # learn about subprocess module 
import os

files = os.listdir("Vedios")

for file in files:
    # print(file)
    turtorial_number = file.split(".")[0].split("#")[1]  # extracted the tutorial number here 
    
    file_name = file.split("_")[0]
    print(turtorial_number, file_name)

    # converting video to audio
    subprocess.run(["ffmpeg", "-i", f'Vedios/{file}', f"audios/{turtorial_number}_{file_name}.mp3"])