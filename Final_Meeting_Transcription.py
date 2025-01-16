import os
import re
import wave
import pyaudio
import threading
import whisper
from pyannote.audio import Pipeline
from datetime import datetime
from transformers import pipeline as summarization_pipeline
from tkinter import Tk, Label, Button, Text, Toplevel, END, messagebox

# Retrieve the Hugging Face token from environment variable
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Initialize PyAudio and Whisper model
p = pyaudio.PyAudio()
whisper_model = whisper.load_model("base")  # Change model size as needed
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token= hf_token)




# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
MIC_FILENAME = "microphone_audio.wav"
SYS_FILENAME = "system_audio.wav"



# Global variables
recording = False
transcription_text = ""  # Store transcription globally



# Function to record audio
def record_audio(device_index, filename, device_name):
    global recording
    print(f"Recording started on {device_name}...")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)
    frames = []
    while recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    stream.stop_stream()
    stream.close()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    print(f"Recording saved as {filename}")



# Function to transcribe audio using Whisper
def transcribe_audio(filename):
    print(f"Transcribing {filename}...")
    try:
        return whisper_model.transcribe(filename)
    except Exception as e:
        print(f"Error during transcription: {e}")
        return {"segments": []}



# Function to diarize audio
def diarize_audio(filename):
    print(f"Diarizing {filename}...")
    try:
        diarization_result = pipeline(filename)
        segments = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        return segments
    except Exception as e:
        print(f"Error during diarization: {e}")
        return []



# Function to merge transcriptions with diarization (cleaned and ordered)
def merge_transcriptions(mic_segments, sys_segments, diarization_segments):
    """
    Combines microphone and system audio transcriptions with diarization segments.
    Ensures:
    - Microphone audio is labeled as 'You'.
    - System audio is labeled with Speaker IDs (Speaker 1, Speaker 2, etc.).
    - No duplicated or out-of-sequence data.
    """
    merged_results = []
    speaker_map = {}
    speaker_counter = 1

    # Label microphone audio as "You"
    for segment in mic_segments:
        merged_results.append({
            "start": segment.get("start"),
            "end": segment.get("end"),
            "speaker": "You",
            "text": segment.get("text", "").strip()
        })

    # Map system audio speakers to Speaker 1, Speaker 2, etc.
    for diarized in diarization_segments:
        speaker = diarized.get("speaker")
        if speaker not in speaker_map:
            speaker_map[speaker] = f"Speaker {speaker_counter}"
            speaker_counter += 1

        start, end = diarized.get("start", 0), diarized.get("end", 0)
        relevant_text = []
        for segment in sys_segments:
            if segment.get("start", 0) < end and segment.get("end", 0) > start:
                relevant_text.append(segment.get("text", "").strip())
        
        combined_text = " ".join(relevant_text).strip()
        if combined_text:
            merged_results.append({
                "start": start,
                "end": end,
                "speaker": speaker_map[speaker],
                "text": combined_text
            })

    # Sort by start time to maintain sequence
    merged_results = sorted(merged_results, key=lambda x: x["start"])
    return merged_results


import re
from transformers import pipeline as summarization_pipeline

# Enhanced function to summarize transcription with dates and commitments
def summarize_transcription(transcription_text):
    """
    Summarizes the transcription with focus on discussion topics, key points, and commitments.
    Removes speaker labels, timestamps from the summary, and extracts dates for commitments.
    """
    # Initialize the summarizer (using BART model for summarization)
    summarizer = summarization_pipeline("summarization", model="facebook/bart-large-cnn")

    # Step 1: Clean the transcription text
    # Remove speaker labels and timestamps
    cleaned_text = re.sub(r"\[\d+\.\d+s - \d+\.\d+s\] Speaker .*?:", "", transcription_text)
    cleaned_text = re.sub(r"\[\d+\.\d+s - \d+\.\d+s\]", "", cleaned_text)
    cleaned_text = cleaned_text.strip()

    # Step 2: Generate the summary using the LLM-based summarizer
    summary_parts = summarizer(cleaned_text, max_length=250, min_length=100, do_sample=False)
    summary_text = " ".join([part["summary_text"] for part in summary_parts])

    # Step 3: Extract commitments and related dates from the cleaned text
    commitment_keywords = [
        "will", "shall", "commit", "promise", "agree", "deadline", "task", "must", "need to", "complete by", "due date"
    ]
    date_pattern = r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}(?:, \d{4})?)\b"

    # Extract commitments and related dates
    commitments = []
    lines = cleaned_text.splitlines()

    for line in lines:
        if any(keyword in line.lower() for keyword in commitment_keywords):
            dates = re.findall(date_pattern, line)
            if dates:
                commitments.append(f"{line.strip()} (Dates: {', '.join(dates)})")
            else:
                commitments.append(line.strip())

    # Step 4: Format the final detailed summary
    commitments_section = "\n".join([f"{idx + 1}. {commitment}" for idx, commitment in enumerate(commitments)]) or "No commitments found."

    # Final enhanced summary
    detailed_summary = (
        f"### Summary ###\n{summary_text}\n\n"
        f"### Commitments ###\n{commitments_section}"
    )

    return detailed_summary


# Save transcription and summary
def save_results(transcription_text, summary_text):
    date_str = datetime.now().strftime("%Y-%m-%d")
    with open(f"transcription_{date_str}.txt", "w", encoding="utf-8") as tf:
        tf.write(transcription_text)
    with open(f"summary_{date_str}.txt", "w", encoding="utf-8") as sf:
        sf.write(summary_text)
    print("Results saved.")



# GUI
def start_recording():
    global recording
    recording = True
    threading.Thread(target=record_audio, args=(1, SYS_FILENAME, "System Audio")).start()
    threading.Thread(target=record_audio, args=(2, MIC_FILENAME, "Microphone Audio")).start()

def stop_recording():
    global recording
    recording = False

def process_audio():
    global transcription_text
    try:
        mic_segments = transcribe_audio(MIC_FILENAME).get("segments", [])
        sys_segments = transcribe_audio(SYS_FILENAME).get("segments", [])
        diarization_segments = diarize_audio(SYS_FILENAME)
        merged_results = merge_transcriptions(mic_segments, sys_segments, diarization_segments)
        transcription_text = "\n".join([
            f"[{res['start']:.2f}s - {res['end']:.2f}s] Speaker {res['speaker']}: {res['text']}"
            for res in merged_results
        ])
        summary_text = summarize_transcription(transcription_text)
        save_results(transcription_text, summary_text)
        messagebox.showinfo("Processing Complete", "Audio processing and summarization completed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def open_summary_view():
    def display_summary():
        if transcription_text:
            summary_text = summarize_transcription(transcription_text)
            summary_box.delete("1.0", END)
            summary_box.insert("1.0", summary_text)
        else:
            messagebox.showinfo("Info", "No transcription available for summarization.")

    summary_window = Toplevel()
    summary_window.title("Summary View")
    summary_window.geometry("600x500")
    Label(summary_window, text="Summary", font=("Arial", 14, "bold"), pady=10).pack()
    summary_box = Text(summary_window, wrap="word", font=("Arial", 12), padx=10, pady=10)
    summary_box.pack(expand=True, fill="both", pady=10, padx=10)
    Button(summary_window, text="Generate Summary", command=display_summary, bg="blue", fg="white", font=("Arial", 10, "bold"), padx=10, pady=5).pack(pady=10)

root = Tk()
root.title("Meeting Assistant")
root.geometry("400x400")
Label(root, text="Meeting Transcription Tool", font=("Arial", 16, "bold"), pady=10).pack()
Button(root, text="Start Recording", command=start_recording, bg="green", fg="white", font=("Arial", 12, "bold"), padx=20, pady=10).pack(pady=10)
Button(root, text="Stop Recording", command=stop_recording, bg="red", fg="white", font=("Arial", 12, "bold"), padx=20, pady=10).pack(pady=10)
Button(root, text="Process Audio", command=process_audio, bg="blue", fg="white", font=("Arial", 12, "bold"), padx=20, pady=10).pack(pady=10)
Button(root, text="View Summary", command=open_summary_view, bg="orange", fg="white", font=("Arial", 12, "bold"), padx=20, pady=10).pack(pady=10)
root.mainloop()
p.terminate()





































