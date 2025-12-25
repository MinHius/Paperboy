import sys
sys.path.append("D:/DH/Senior/Paperboy") 
from src.audio.utils import concat_wavs
from src.api.chat.utils import json_to_paragraph_audio
from src.database.parade.database import load_stories
from src.audio.neutts import generate_audio
import soundfile as sf
import os
from src.audio.neutts import NeuTTSAir


def audio_generation(story):
    content = json_to_paragraph_audio(story['story'])
    chunks = chunk_text(content, max_chars=180)

    print(f"Story split into {len(chunks)} chunks")

    wav_paths = []
    
    # Load model ONCE
    TTS = NeuTTSAir(
        backbone_repo="neuphonic/neutts-air",
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu",
    )

    # Load reference ONCE
    REF_AUDIO = "src/audio/dave.wav"
    REF_TEXT = open("src/audio/dave.txt").read().strip()
    REF_CODES = TTS.encode_reference(REF_AUDIO)

    for i, chunk in enumerate(chunks):
        out_path = f"output_{i}.wav"
        # print(f"Generating chunk {i + 1}/{len(chunks)}...")
        # wav = TTS.infer(chunks, REF_CODES, REF_TEXT)
        # sf.write(out_path, wav, 24000)
        wav_paths.append(out_path)
        generate_audio(i, chunk, out_path)

    final_path = f"output/test_final.wav"
    concat_wavs(wav_paths, final_path)

    print("Done:", final_path)
    return final_path
    

def chunk_text(text: str, max_chars=180):  
    chunks = []
    current = []

    for word in text.split():
        if sum(len(x) + 1 for x in current) + len(word) > max_chars:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(" ".join(current))

    return chunks

        
        
stories = load_stories()
for i, story in enumerate(stories):
    audio_generation(story)
    wav_paths = []
    final_path = f"output/story_{i}_audio.wav"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    for j in range(0,13):
        out_path = f"output_{j}.wav"
        wav_paths.append(out_path)
    concat_wavs(wav_paths, final_path)
    print(f"Concatenated wav file {j}.")









