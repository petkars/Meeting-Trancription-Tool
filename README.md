# Meeting Transcription Tool

A Python-based tool to record, transcribe, diarize, and summarize meeting audio using advanced AI models like Whisper and Hugging Face pipelines.

## Features

- Records audio from both system and microphone devices simultaneously.
- Transcribes audio using OpenAI's Whisper model.
- Diarizes speakers using PyAnnote.
- Generates detailed summaries with focus on key discussion points and commitments.
- Interactive GUI built with Tkinter for ease of use.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/meeting-transcription-tool.git
cd meeting-transcription-tool
```

### 2. Install dependencies

Ensure you have Python 3.7+ installed. Then, run:

```bash
pip install -r requirements.txt
```

### 3. Set up Hugging Face token

Export your Hugging Face token as an environment variable:

```bash
export HUGGINGFACE_TOKEN=your_huggingface_token
```

### 4. Run the tool

```bash
python main.py
```

## Usage

### 1. Start Recording

Click on the **Start Recording** button to begin recording audio from the system and microphone devices.

### 2. Stop Recording

Click on the **Stop Recording** button to stop recording.

### 3. Process Audio

Click on the **Process Audio** button to transcribe, diarize, and summarize the audio. The results will be saved as text files in the current directory.

### 4. View Summary

Click on the **View Summary** button to open a new window showing the detailed summary of the meeting.

## File Structure

```
meeting-transcription-tool/
|-- main.py
|-- requirements.txt
|-- images/
    |-- start_recording.png
    |-- stop_recording.png
    |-- process_audio.png
    |-- view_summary.png
|-- transcription_YYYY-MM-DD.txt
|-- summary_YYYY-MM-DD.txt
```

## Requirements

- Python 3.7 or higher
- PyAudio
- OpenAI Whisper
- PyAnnote.audio
- Transformers
- Tkinter (comes pre-installed with Python)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

Shubham Petkar (mail to:petkars907@gmail.com)
