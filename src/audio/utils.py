import wave

def concat_wavs(wav_paths: str, output_path: str) -> None:
    with wave.open(wav_paths[0], 'rb') as first:
        params = first.getparams()

    with wave.open(output_path, 'wb') as out:
        out.setparams(params)

        for wav in wav_paths:
            with wave.open(wav, 'rb') as w:
                out.writeframes(w.readframes(w.getnframes()))
