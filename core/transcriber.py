import os
from faster_whisper import WhisperModel
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.audio_processing import convert_to_chunks, convert_to_wav, download_audio

model = None


def load_model(model_size="small", device="cpu", compute_type="int8"):
    global model

    if model is None:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

    return model


def transcribe_file(audio_path, language=None, translate=False):
    model = load_model()

    if translate:
        task = "translate"
    else:
        task = "transcribe"

    segments, info = model.transcribe(
        audio_path,
        language=language,
        task=task,
        beam_size=1,
        best_of=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    results = []

    for seg in segments:
        results.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})

    return results, info


def transcribe_chunks(chunk_paths, language="en", translate=False):
    all_segments = []
    offset = 0.0

    for i, path in enumerate(chunk_paths, 1):
        print(f"Transcribing chunk {i}/{len(chunk_paths)}: {os.path.basename(path)}")
        segs, info = transcribe_file(path, language=language, translate=translate)

        for s in segs:
            all_segments.append(
                {
                    "start": s["start"] + offset,
                    "end": s["end"] + offset,
                    "text": s["text"],
                }
            )

        offset += info.duration

    full_text = " ".join(s["text"] for s in all_segments)

    return full_text


def transcribe_source(source, language="en", translate=False):
    if os.path.isfile(source):
        print("Using local file...")
        raw_path = source
    else:
        print("Downloading audio...")
        raw_path = download_audio(source)

    wav_path = convert_to_wav(raw_path)

    chunks = convert_to_chunks(wav_path)

    return transcribe_chunks(chunks, language=language, translate=translate)
