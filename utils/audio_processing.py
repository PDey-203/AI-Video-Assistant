import os
import subprocess

import yt_dlp
from yt_dlp.utils import match_filter_func

TEMP_FOLDER = "DOWNLOADS"


def download_audio(url):
    os.makedirs(TEMP_FOLDER, exist_ok=True)

    options = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(TEMP_FOLDER, "extracted_audio.%(ext)s"),
        "noplaylist": True,
        "match_filter": match_filter_func("!is_live"),
        "restrictfilenames": True,
        "quiet": True,
        "no_warnings": False,
        "noprogress": True,
        "retries": 3,
        "fragment_retries": 3,
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as error:
        raise Exception(f"Could not download this link. Details: {error}")

    downloads = info.get("requested_downloads") if info else None

    file_path = downloads[0]["filepath"]

    return file_path


def convert_to_wav(input_path):
    output_path = os.path.join(TEMP_FOLDER, "converted_audio.wav")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        output_path,
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception("FFmpeg could not convert the file:\n" + result.stderr[-300:])

    return output_path


def convert_to_chunks(input_path, CHUNK_SIZE=300):
    chunk_folder = os.path.join(TEMP_FOLDER, "chunks")
    os.makedirs(chunk_folder, exist_ok=True)

    chunks = []

    output_pattern = os.path.join(chunk_folder, "chunk_audio_%03d.wav")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-f",
        "segment",
        "-segment_time",
        str(CHUNK_SIZE),
        "-c",
        "copy",
        "-reset_timestamps",
        "1",
        output_pattern,
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception("FFmpeg could not convert the file:\n" + result.stderr[-300:])

    for f in sorted(os.listdir(chunk_folder)):
        if f.endswith(".wav"):
            chunks.append(os.path.join(chunk_folder, f))

    return chunks
