# Holmes Shorts Factory — AI-Powered Science of Mind YouTube Shorts Pipeline

## Project Overview

An automated content pipeline that transforms Ernest Holmes' public domain writings into monetizable YouTube Shorts. The system uses a local LLM (Ollama/Animus) to extract compelling passages, generates voiceover and visuals, and assembles finished Shorts ready for upload.

### Why This Works

- **Ernest Holmes' pre-1929 works are public domain** — zero copyright risk
- **Massive source library** — hundreds of Shorts worth of material
- **Evergreen spiritual content** — not trend-dependent, compounds over time
- **Fully automatable** — batch produce 30+ Shorts per session
- **Low/no ongoing cost** — all local tools, no API fees required

### Monetization Strategy

YouTube Shorts ad revenue alone won't pay the bills (~$0.01-0.07/1K views). The Shorts are a **top-of-funnel acquisition engine**:

| Layer | Vehicle | Revenue |
|-------|---------|---------|
| Free | YouTube Shorts (daily) | Ad revenue + audience growth |
| Free | YouTube long-form (weekly) | Higher RPM ad revenue |
| Nurture | Email list / Discord community | Relationship building |
| Paid | Digital course on Science of Mind practices | $47-197 one-time |
| Paid | Monthly membership (Skool/Patreon) | $9-29/mo recurring |
| Paid | Affiliate links (Holmes books, courses, retreats) | Commission |
| Paid | Coaching / consultation | $100-300/hr |
| Scale | License the pipeline to other creators | SaaS or done-for-you |

---

## Source Material (Public Domain)

### Confirmed Public Domain Works

| Title | Year | Status | Notes |
|-------|------|--------|-------|
| Creative Mind | 1919 | ✅ Public Domain | Short, punchy — great for Shorts |
| Creative Mind and Success | 1919 | ✅ Public Domain | Success-focused — broad appeal |
| The Science of Mind (1st ed.) | 1926 | ✅ Public Domain | The motherlode — hundreds of passages |

### Where to Source Texts

- [Project Gutenberg](https://www.gutenberg.org/) — search "Ernest Holmes"
- [Internet Archive](https://archive.org/) — full scans and OCR text
- [HathiTrust](https://www.hathitrust.org/) — academic digital library
- Direct OCR from scanned editions if needed

### Organizing the Source Library

```
holmes-shorts/
├── sources/
│   ├── creative_mind_1919.txt
│   ├── creative_mind_and_success_1919.txt
│   └── science_of_mind_1926.txt
```

**Preprocessing step**: Clean OCR artifacts, normalize formatting, split into chapters/sections for easier LLM processing.

```python
# scripts/preprocess_source.py
"""
Clean and chunk public domain texts for LLM processing.
"""
import re
from pathlib import Path

def clean_text(raw: str) -> str:
    """Remove OCR artifacts and normalize whitespace."""
    text = re.sub(r'[^\x00-\x7F]+', '', raw)  # strip non-ASCII OCR noise
    text = re.sub(r'\n{3,}', '\n\n', text)      # collapse excessive newlines
    text = re.sub(r' {2,}', ' ', text)           # collapse multiple spaces
    return text.strip()

def chunk_by_section(text: str, max_chars: int = 3000) -> list[str]:
    """Split text into chunks roughly by paragraph groups."""
    paragraphs = text.split('\n\n')
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) > max_chars and current:
            chunks.append(current.strip())
            current = p
        else:
            current += "\n\n" + p
    if current.strip():
        chunks.append(current.strip())
    return chunks

if __name__ == "__main__":
    source_dir = Path("sources")
    output_dir = Path("chunks")
    output_dir.mkdir(exist_ok=True)

    for txt_file in source_dir.glob("*.txt"):
        raw = txt_file.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_text(raw)
        chunks = chunk_by_section(cleaned)
        for i, chunk in enumerate(chunks):
            out_path = output_dir / f"{txt_file.stem}_chunk_{i:03d}.txt"
            out_path.write_text(chunk)
        print(f"[✓] {txt_file.name} → {len(chunks)} chunks")
```

---

## Architecture

### Pipeline Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Source Text │────▶│ Ollama/Animus│────▶│  Script +   │
│  (PD works)  │     │  (extract &  │     │  Hook Text  │
└─────────────┘     │  write hook) │     └──────┬──────┘
                    └──────────────┘            │
                                                ▼
                    ┌──────────────┐     ┌─────────────┐
                    │   Whisper    │◀────│  TTS Engine  │
                    │  (captions)  │     │ (voiceover)  │
                    └──────┬───────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐     ┌─────────────┐
                    │   FFmpeg     │◀────│   Visuals    │
                    │  (assembly)  │     │ (SD / stock) │
                    └──────┬───────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Final MP4   │
                    │  (1080x1920) │
                    └──────────────┘
```

### Tech Stack

| Component | Tool | Local/Cloud | Cost |
|-----------|------|-------------|------|
| LLM | Ollama (Animus) | Local | Free |
| TTS | Piper TTS | Local | Free |
| TTS (upgrade) | ElevenLabs | Cloud | ~$5/mo |
| Visuals (MVP) | Stock bg + text overlays | Local | Free |
| Visuals (v2) | Stable Diffusion | Local | Free (needs GPU) |
| Captions | Whisper | Local | Free |
| Assembly | FFmpeg | Local | Free |
| Orchestration | Python | Local | Free |

---

## System Requirements

### Minimum (MVP — text overlay Shorts)

- Any modern computer (no GPU needed for MVP)
- 8GB RAM (16GB preferred for Ollama)
- Ollama installed with your preferred model
- Python 3.10+
- FFmpeg installed

### Recommended (Full pipeline with AI visuals)

- NVIDIA GPU with 8GB+ VRAM (for Stable Diffusion)
- 32GB RAM
- 50GB+ free disk space

### Installation

```bash
# 1. System dependencies
sudo apt update && sudo apt install -y ffmpeg imagemagick

# 2. Python environment
python3 -m venv venv
source venv/bin/activate

# 3. Python packages
pip install requests piper-tts openai-whisper Pillow

# 4. Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# 5. Pull your model (adjust to whatever Animus runs)
ollama pull llama3.1:8b

# 6. Piper TTS voice (calm, warm male voice)
mkdir -p models/piper
cd models/piper
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en-us-lessac-medium.onnx
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en-us-lessac-medium.onnx.json
cd ../..

# 7. (Optional) Whisper for caption generation
pip install openai-whisper
```

---

## Directory Structure

```
holmes-shorts/
├── README.md
├── requirements.txt
├── config.yaml                  # Global settings
├── sources/                     # Public domain source texts
│   ├── creative_mind_1919.txt
│   ├── creative_mind_and_success_1919.txt
│   └── science_of_mind_1926.txt
├── chunks/                      # Preprocessed text chunks
├── scripts/
│   ├── preprocess_source.py     # Clean & chunk source texts
│   ├── extract_passage.py       # Ollama passage extraction
│   ├── generate_voiceover.py    # TTS generation
│   ├── generate_visuals.py      # Background/overlay creation
│   ├── generate_captions.py     # Whisper caption generation
│   ├── assemble_short.py        # FFmpeg final assembly
│   └── pipeline.py              # Full orchestration
├── assets/
│   ├── backgrounds/             # Background images/videos
│   ├── fonts/                   # Custom fonts
│   ├── music/                   # Royalty-free ambient tracks
│   └── overlays/                # Logo, watermark, etc.
├── output/
│   ├── scripts/                 # Generated scripts (JSON)
│   ├── audio/                   # Generated voiceovers
│   ├── captions/                # Generated SRT files
│   └── shorts/                  # Final MP4 files
└── models/
    └── piper/                   # TTS voice models
```

---

## Core Scripts

### 1. Passage Extraction — `scripts/extract_passage.py`

This is the brain of the operation. Ollama reads a chunk of Holmes text and returns a structured Short script.

```python
#!/usr/bin/env python3
"""
Extract compelling passages from Ernest Holmes texts using Ollama.
Outputs structured JSON scripts for YouTube Shorts production.
"""
import json
import requests
import sys
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"  # Replace with your Animus model

SYSTEM_PROMPT = """You are a content producer specializing in spiritual and 
philosophical YouTube Shorts. Your job is to extract the most powerful, 
compelling passages from Ernest Holmes' writings and transform them into 
30-45 second YouTube Short scripts.

RULES:
1. The passage must stand alone — no context needed
2. It must evoke emotion, insight, or a paradigm shift
3. Write a scroll-stopping hook (first 3 seconds) that is NOT from the source text
4. The hook should create curiosity or challenge a belief
5. Keep the total spoken word count between 75-120 words
6. End with a thought-provoking closing line (can be original)
7. Tag with relevant themes for visual matching

Respond ONLY in this JSON format:
{
    "hook": "The scroll-stopping opening line (your original words)",
    "passage": "The extracted/adapted Holmes passage (public domain text, lightly adapted for spoken delivery)",
    "closing": "A punchy closing thought (can be original or from Holmes)",
    "themes": ["theme1", "theme2", "theme3"],
    "source_work": "Title of the source work",
    "mood": "contemplative|empowering|mystical|practical|challenging",
    "estimated_duration_seconds": 35,
    "word_count": 95
}"""

USER_PROMPT_TEMPLATE = """Here is a passage from Ernest Holmes' "{work_title}". 
Extract the most compelling segment and create a YouTube Short script from it.

---
{chunk_text}
---

Remember: JSON only. Hook must grab attention in 3 seconds. 
75-120 words total spoken content."""


def extract_passage(chunk_text: str, work_title: str) -> dict:
    """Send chunk to Ollama and get structured Short script back."""
    payload = {
        "model": MODEL,
        "prompt": USER_PROMPT_TEMPLATE.format(
            work_title=work_title,
            chunk_text=chunk_text
        ),
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
        }
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    raw = response.json()["response"]

    # Parse JSON from response (handle markdown code blocks)
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]

    return json.loads(raw.strip())


def process_chunks(chunks_dir: str, output_dir: str, work_title: str):
    """Process all chunks in a directory."""
    chunks_path = Path(chunks_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for chunk_file in sorted(chunks_path.glob("*.txt")):
        print(f"[→] Processing: {chunk_file.name}")
        chunk_text = chunk_file.read_text()

        try:
            script = extract_passage(chunk_text, work_title)
            out_file = output_path / f"{chunk_file.stem}_script.json"
            out_file.write_text(json.dumps(script, indent=2))
            print(f"  [✓] Script generated: {out_file.name}")
            print(f"      Hook: {script['hook'][:60]}...")
            print(f"      Mood: {script['mood']} | Words: {script.get('word_count', '?')}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [✗] Failed to parse: {e}")
        except requests.RequestException as e:
            print(f"  [✗] Ollama error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_passage.py <chunks_dir> <work_title>")
        print("Example: python extract_passage.py chunks/ 'The Science of Mind'")
        sys.exit(1)

    process_chunks(
        chunks_dir=sys.argv[1],
        output_dir="output/scripts",
        work_title=sys.argv[2]
    )
```

### 2. Voiceover Generation — `scripts/generate_voiceover.py`

```python
#!/usr/bin/env python3
"""
Generate voiceover audio from Short scripts using Piper TTS.
"""
import json
import subprocess
import sys
from pathlib import Path

PIPER_MODEL = "models/piper/voice-en-us-lessac-medium.onnx"

# Alternate: use ElevenLabs API for higher quality
USE_ELEVENLABS = False
ELEVENLABS_API_KEY = ""  # Set if using ElevenLabs
ELEVENLABS_VOICE_ID = ""  # Choose a calm, authoritative voice


def generate_with_piper(text: str, output_path: str):
    """Generate audio using local Piper TTS."""
    cmd = [
        "piper",
        "--model", PIPER_MODEL,
        "--output_file", output_path,
    ]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate(input=text.encode("utf-8"))
    if process.returncode != 0:
        raise RuntimeError(f"Piper failed: {stderr.decode()}")


def generate_with_elevenlabs(text: str, output_path: str):
    """Generate audio using ElevenLabs API (higher quality, costs money)."""
    import requests

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.6,
            "similarity_boost": 0.75,
            "style": 0.3,  # Slight expressiveness
        }
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)


def build_spoken_text(script: dict) -> str:
    """Combine hook + passage + closing into natural spoken text."""
    parts = []
    if script.get("hook"):
        parts.append(script["hook"])
    if script.get("passage"):
        parts.append(script["passage"])
    if script.get("closing"):
        parts.append(script["closing"])
    # Add pauses between sections
    return " ... ".join(parts)


def process_scripts(scripts_dir: str, output_dir: str):
    """Generate voiceovers for all script JSON files."""
    scripts_path = Path(scripts_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generate_fn = generate_with_elevenlabs if USE_ELEVENLABS else generate_with_piper

    for script_file in sorted(scripts_path.glob("*_script.json")):
        print(f"[→] Generating voiceover: {script_file.name}")
        script = json.loads(script_file.read_text())
        spoken_text = build_spoken_text(script)

        out_file = output_path / f"{script_file.stem.replace('_script', '')}.wav"
        try:
            generate_fn(spoken_text, str(out_file))
            print(f"  [✓] Audio saved: {out_file.name}")
        except Exception as e:
            print(f"  [✗] TTS failed: {e}")


if __name__ == "__main__":
    process_scripts(
        scripts_dir=sys.argv[1] if len(sys.argv) > 1 else "output/scripts",
        output_dir="output/audio"
    )
```

### 3. Visual Generation — `scripts/generate_visuals.py`

```python
#!/usr/bin/env python3
"""
Generate background visuals for Shorts.

MVP: Gradient backgrounds with animated text overlays.
V2: Stable Diffusion generated imagery based on themes.
"""
import json
import subprocess
import sys
from pathlib import Path

# Visual style presets
STYLES = {
    "cosmic": {
        "gradient_start": "#0a0a2e",
        "gradient_end": "#1a0533",
        "text_color": "#e8d5b7",
        "accent_color": "#7b68ee",
        "font": "assets/fonts/Cormorant-SemiBold.ttf",
    },
    "vintage": {
        "gradient_start": "#2c1810",
        "gradient_end": "#1a1008",
        "text_color": "#d4a574",
        "accent_color": "#8b6914",
        "font": "assets/fonts/Cormorant-SemiBold.ttf",
    },
    "ethereal": {
        "gradient_start": "#0d1b2a",
        "gradient_end": "#1b263b",
        "text_color": "#e0e1dd",
        "accent_color": "#778da9",
        "font": "assets/fonts/Cormorant-SemiBold.ttf",
    },
    "empowering": {
        "gradient_start": "#1a1a2e",
        "gradient_end": "#16213e",
        "text_color": "#f4d03f",
        "accent_color": "#e74c3c",
        "font": "assets/fonts/Cormorant-SemiBold.ttf",
    },
}

# Map moods to visual styles
MOOD_STYLE_MAP = {
    "contemplative": "ethereal",
    "empowering": "empowering",
    "mystical": "cosmic",
    "practical": "vintage",
    "challenging": "empowering",
}


def create_gradient_background(width: int, height: int, style: dict, output_path: str):
    """Create a gradient background image using ImageMagick."""
    cmd = [
        "magick", "-size", f"{width}x{height}",
        f"gradient:{style['gradient_start']}-{style['gradient_end']}",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def create_background_video(
    duration: float,
    style: dict,
    output_path: str,
    width: int = 1080,
    height: int = 1920,
):
    """Create a simple animated gradient background video."""
    # Generate static gradient, then create video from it
    bg_img = "/tmp/holmes_bg.png"
    create_gradient_background(width, height, style, bg_img)

    # Create video from static image with subtle zoom animation
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", bg_img,
        "-t", str(duration),
        "-vf", (
            f"scale=1200:2133,zoompan=z='min(zoom+0.0005,1.1)':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={int(duration*30)}:s={width}x{height}:fps=30"
        ),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def process_scripts(scripts_dir: str, output_dir: str):
    """Generate background videos for all scripts."""
    scripts_path = Path(scripts_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for script_file in sorted(scripts_path.glob("*_script.json")):
        print(f"[→] Generating visual: {script_file.name}")
        script = json.loads(script_file.read_text())

        mood = script.get("mood", "contemplative")
        style_name = MOOD_STYLE_MAP.get(mood, "ethereal")
        style = STYLES[style_name]
        duration = script.get("estimated_duration_seconds", 35) + 3  # padding

        out_file = output_path / f"{script_file.stem.replace('_script', '')}_bg.mp4"
        try:
            create_background_video(duration, style, str(out_file))
            print(f"  [✓] Visual saved: {out_file.name} ({style_name})")
        except Exception as e:
            print(f"  [✗] Visual generation failed: {e}")


if __name__ == "__main__":
    process_scripts(
        scripts_dir=sys.argv[1] if len(sys.argv) > 1 else "output/scripts",
        output_dir="output/visuals"
    )
```

### 4. Caption Generation — `scripts/generate_captions.py`

```python
#!/usr/bin/env python3
"""
Generate word-level captions using Whisper for burned-in subtitles.
Critical for Shorts engagement — most viewers watch without sound.
"""
import json
import subprocess
import sys
from pathlib import Path


def generate_srt(audio_path: str, output_path: str):
    """Generate SRT captions from audio using Whisper."""
    import whisper

    model = whisper.load_model("base")  # Use "small" or "medium" for better accuracy
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en",
    )

    # Build SRT from word-level timestamps
    # Group into 3-5 word phrases for animated caption style
    words = []
    for segment in result["segments"]:
        if "words" in segment:
            words.extend(segment["words"])

    srt_entries = []
    idx = 1
    group_size = 4  # words per caption group
    for i in range(0, len(words), group_size):
        group = words[i:i + group_size]
        start = group[0]["start"]
        end = group[-1]["end"]
        text = " ".join(w["word"].strip() for w in group)

        srt_entries.append(
            f"{idx}\n"
            f"{format_time(start)} --> {format_time(end)}\n"
            f"{text}\n"
        )
        idx += 1

    Path(output_path).write_text("\n".join(srt_entries))


def format_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def process_audio(audio_dir: str, output_dir: str):
    """Generate captions for all audio files."""
    audio_path = Path(audio_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for audio_file in sorted(audio_path.glob("*.wav")):
        print(f"[→] Generating captions: {audio_file.name}")
        out_file = output_path / f"{audio_file.stem}.srt"
        try:
            generate_srt(str(audio_file), str(out_file))
            print(f"  [✓] Captions saved: {out_file.name}")
        except Exception as e:
            print(f"  [✗] Caption generation failed: {e}")


if __name__ == "__main__":
    process_audio(
        audio_dir=sys.argv[1] if len(sys.argv) > 1 else "output/audio",
        output_dir="output/captions"
    )
```

### 5. Final Assembly — `scripts/assemble_short.py`

```python
#!/usr/bin/env python3
"""
Assemble final YouTube Short from components using FFmpeg.
Combines: background video + voiceover + burned-in captions + optional music.
Output: 1080x1920 MP4 ready for YouTube upload.
"""
import json
import subprocess
import sys
from pathlib import Path

# Caption styling — big, bold, centered text (TikTok/Shorts style)
CAPTION_STYLE = (
    "FontName=Impact,"
    "FontSize=22,"
    "PrimaryColour=&H00FFFFFF,"    # White text
    "OutlineColour=&H00000000,"    # Black outline
    "BorderStyle=3,"               # Opaque box behind text
    "BackColour=&H80000000,"       # Semi-transparent black bg
    "Outline=2,"
    "Shadow=0,"
    "MarginV=120,"                 # Positioned in lower third
    "Alignment=2"                  # Bottom center
)

# Optional: ambient background music volume (0.0 to 1.0)
MUSIC_VOLUME = 0.08  # Very quiet — just atmosphere


def assemble_short(
    background_video: str,
    audio_file: str,
    caption_file: str,
    output_file: str,
    music_file: str = None,
):
    """Assemble all components into a final Short."""

    # Get audio duration to trim video
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_file,
    ]
    duration = float(subprocess.check_output(probe_cmd).decode().strip())
    duration += 1.5  # Add 1.5s padding at end

    # Build FFmpeg command
    inputs = [
        "-i", background_video,
        "-i", audio_file,
    ]

    filter_parts = []
    audio_mix = "[1:a]"

    # Add background music if provided
    if music_file and Path(music_file).exists():
        inputs.extend(["-i", music_file])
        filter_parts.append(
            f"[2:a]volume={MUSIC_VOLUME}[music];"
            f"[1:a][music]amix=inputs=2:duration=first[mixed]"
        )
        audio_mix = "[mixed]"

    # Subtitle filter
    sub_filter = (
        f"subtitles={caption_file}:force_style='{CAPTION_STYLE}'"
    )

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-t", str(duration),
        "-filter_complex",
        (
            f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
            f"{sub_filter}[vout];"
            + ("".join(filter_parts) if filter_parts else "")
        ).rstrip(";"),
        "-map", "[vout]",
        "-map", audio_mix,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-movflags", "+faststart",  # Optimized for streaming
        "-pix_fmt", "yuv420p",
        output_file,
    ]

    print(f"[→] Assembling: {output_file}")
    subprocess.run(cmd, check=True)
    print(f"[✓] Done! Duration: {duration:.1f}s")


def process_all(
    scripts_dir: str = "output/scripts",
    audio_dir: str = "output/audio",
    visuals_dir: str = "output/visuals",
    captions_dir: str = "output/captions",
    output_dir: str = "output/shorts",
    music_file: str = None,
):
    """Assemble all Shorts from generated components."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for script_file in sorted(Path(scripts_dir).glob("*_script.json")):
        stem = script_file.stem.replace("_script", "")
        audio = Path(audio_dir) / f"{stem}.wav"
        visual = Path(visuals_dir) / f"{stem}_bg.mp4"
        caption = Path(captions_dir) / f"{stem}.srt"
        output = output_path / f"{stem}_short.mp4"

        if not audio.exists():
            print(f"  [!] Missing audio: {audio}")
            continue
        if not visual.exists():
            print(f"  [!] Missing visual: {visual}")
            continue
        if not caption.exists():
            print(f"  [!] Missing caption: {caption}")
            continue

        try:
            assemble_short(
                str(visual), str(audio), str(caption), str(output),
                music_file=music_file,
            )
        except Exception as e:
            print(f"  [✗] Assembly failed for {stem}: {e}")


if __name__ == "__main__":
    music = sys.argv[1] if len(sys.argv) > 1 else None
    process_all(music_file=music)
```

### 6. Full Pipeline Orchestrator — `scripts/pipeline.py`

```python
#!/usr/bin/env python3
"""
Holmes Shorts Factory — Full Pipeline Orchestrator

Run the entire pipeline from source text to finished YouTube Shorts.

Usage:
    python scripts/pipeline.py                    # Process all chunks
    python scripts/pipeline.py --chunk 005        # Process single chunk
    python scripts/pipeline.py --batch 10         # Process N chunks
    python scripts/pipeline.py --step extract     # Run single step
"""
import argparse
import json
import sys
from pathlib import Path

# Import pipeline stages
from extract_passage import extract_passage
from generate_voiceover import build_spoken_text, generate_with_piper
from generate_visuals import create_background_video, STYLES, MOOD_STYLE_MAP
from generate_captions import generate_srt
from assemble_short import assemble_short


def run_pipeline(
    chunk_file: Path,
    work_title: str,
    music_file: str = None,
    steps: list[str] = None,
):
    """Run full pipeline for a single chunk."""
    stem = chunk_file.stem
    all_steps = ["extract", "voiceover", "visuals", "captions", "assemble"]
    steps = steps or all_steps

    print(f"\n{'='*60}")
    print(f"Processing: {chunk_file.name}")
    print(f"{'='*60}")

    script_path = Path(f"output/scripts/{stem}_script.json")
    audio_path = Path(f"output/audio/{stem}.wav")
    visual_path = Path(f"output/visuals/{stem}_bg.mp4")
    caption_path = Path(f"output/captions/{stem}.srt")
    output_path = Path(f"output/shorts/{stem}_short.mp4")

    # Ensure output dirs exist
    for p in [script_path, audio_path, visual_path, caption_path, output_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract passage
    if "extract" in steps:
        print("\n[1/5] Extracting passage...")
        chunk_text = chunk_file.read_text()
        try:
            script = extract_passage(chunk_text, work_title)
            script_path.write_text(json.dumps(script, indent=2))
            print(f"  Hook: {script['hook'][:80]}...")
        except Exception as e:
            print(f"  [✗] Extraction failed: {e}")
            return False

    # Load script for remaining steps
    if not script_path.exists():
        print(f"  [!] No script found at {script_path}")
        return False
    script = json.loads(script_path.read_text())

    # Step 2: Generate voiceover
    if "voiceover" in steps:
        print("\n[2/5] Generating voiceover...")
        try:
            spoken_text = build_spoken_text(script)
            generate_with_piper(spoken_text, str(audio_path))
            print(f"  [✓] Audio: {audio_path}")
        except Exception as e:
            print(f"  [✗] TTS failed: {e}")
            return False

    # Step 3: Generate visuals
    if "visuals" in steps:
        print("\n[3/5] Generating visuals...")
        try:
            mood = script.get("mood", "contemplative")
            style_name = MOOD_STYLE_MAP.get(mood, "ethereal")
            style = STYLES[style_name]
            duration = script.get("estimated_duration_seconds", 35) + 3
            create_background_video(duration, style, str(visual_path))
            print(f"  [✓] Visual: {visual_path} ({style_name})")
        except Exception as e:
            print(f"  [✗] Visual generation failed: {e}")
            return False

    # Step 4: Generate captions
    if "captions" in steps:
        print("\n[4/5] Generating captions...")
        try:
            generate_srt(str(audio_path), str(caption_path))
            print(f"  [✓] Captions: {caption_path}")
        except Exception as e:
            print(f"  [✗] Caption generation failed: {e}")
            return False

    # Step 5: Assemble final Short
    if "assemble" in steps:
        print("\n[5/5] Assembling final Short...")
        try:
            assemble_short(
                str(visual_path),
                str(audio_path),
                str(caption_path),
                str(output_path),
                music_file=music_file,
            )
            print(f"\n  ✅ DONE: {output_path}")
            print(f"  📏 Check file: ffprobe -v error -show_format {output_path}")
        except Exception as e:
            print(f"  [✗] Assembly failed: {e}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Holmes Shorts Factory")
    parser.add_argument("--chunks-dir", default="chunks", help="Chunks directory")
    parser.add_argument("--work-title", default="The Science of Mind", help="Source work title")
    parser.add_argument("--chunk", help="Process specific chunk number (e.g., 005)")
    parser.add_argument("--batch", type=int, help="Process N chunks")
    parser.add_argument("--step", choices=["extract", "voiceover", "visuals", "captions", "assemble"],
                        help="Run single step only")
    parser.add_argument("--music", help="Path to background music file")
    args = parser.parse_args()

    chunks = sorted(Path(args.chunks_dir).glob("*.txt"))

    if args.chunk:
        chunks = [c for c in chunks if args.chunk in c.stem]
        if not chunks:
            print(f"No chunk matching '{args.chunk}' found")
            sys.exit(1)

    if args.batch:
        chunks = chunks[:args.batch]

    steps = [args.step] if args.step else None

    print(f"Holmes Shorts Factory")
    print(f"Chunks to process: {len(chunks)}")
    print(f"Steps: {steps or 'all'}")
    print(f"Work: {args.work_title}")

    success = 0
    for chunk in chunks:
        if run_pipeline(chunk, args.work_title, args.music, steps):
            success += 1

    print(f"\n{'='*60}")
    print(f"Complete: {success}/{len(chunks)} Shorts generated")
    print(f"Output: output/shorts/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
```

---

## Ollama Prompt Engineering Guide

### Model Selection for Animus

For this pipeline, the model needs to be good at:
- Understanding philosophical/spiritual text
- Extracting emotionally resonant passages
- Writing attention-grabbing hooks
- Outputting clean JSON

**Recommended models (ranked)**:
1. `llama3.1:8b` — Best balance of quality and speed
2. `mistral:7b` — Good creative output, fast
3. `llama3.1:70b` — Best quality if you have the VRAM
4. `phi3:medium` — Surprisingly good for philosophical content

### System Prompt Tuning

The system prompt in `extract_passage.py` is your most important lever. Here are variations to experiment with:

#### For Maximum Engagement (Algorithm-Optimized)

```
You are a viral content strategist specializing in spiritual wisdom. 
Your hooks must create an OPEN LOOP — a question the viewer MUST stay 
to answer. Use pattern interrupts: challenge assumptions, state 
counterintuitive truths, or create urgency.

Hook formulas that work:
- "Most people get [concept] completely wrong..."
- "What if everything you believed about [topic] was backwards?"
- "Ernest Holmes discovered something in 1926 that science just confirmed..."
- "[Bold claim] — here's why..."
```

#### For Depth & Authority (Brand-Building)

```
You are a scholar and teacher of New Thought philosophy. Your role is 
to make Ernest Holmes' profound insights accessible to modern seekers. 
Your hooks should promise transformation. Your tone should be warm but 
authoritative — a wise mentor, not a salesperson.
```

#### For Broad Appeal (Growth Phase)

```
You are translating century-old wisdom into language that resonates 
with people who've never heard of Science of Mind. Avoid jargon. 
Connect Holmes' ideas to modern psychology, neuroscience, and 
self-improvement. Frame insights as practical tools, not abstract 
philosophy.
```

### Testing Your Prompts

```bash
# Quick test — process a single chunk and inspect the output
echo "There is a Power in the universe that responds to our thought. 
This Power is the Law of Mind in action. It is Intelligence acting 
upon our intelligence. It will always respond by corresponding." | \
ollama run llama3.1:8b "$(cat <<'EOF'
[paste your system prompt here]

Extract a YouTube Short script from this Holmes passage. JSON only.
EOF
)"
```

---

## Configuration

### `config.yaml`

```yaml
# Holmes Shorts Factory Configuration

# Ollama settings
ollama:
  url: "http://localhost:11434/api/generate"
  model: "llama3.1:8b"
  temperature: 0.7
  top_p: 0.9

# TTS settings
tts:
  engine: "piper"  # "piper" or "elevenlabs"
  piper:
    model: "models/piper/voice-en-us-lessac-medium.onnx"
  elevenlabs:
    api_key: ""
    voice_id: ""

# Visual settings
visuals:
  default_style: "ethereal"
  width: 1080
  height: 1920
  fps: 30

# Caption settings
captions:
  whisper_model: "base"  # base, small, medium, large
  words_per_group: 4
  style: "bold_centered"  # For Shorts-style animated text

# Output settings
output:
  format: "mp4"
  video_codec: "libx264"
  audio_codec: "aac"
  crf: 23  # Quality (lower = better, bigger file)
  
# Background music
music:
  enabled: false
  file: "assets/music/ambient.mp3"
  volume: 0.08

# Content settings
content:
  min_words: 75
  max_words: 120
  target_duration_seconds: 35
```

### `requirements.txt`

```
requests>=2.31.0
Pillow>=10.0.0
openai-whisper>=20230918
pyyaml>=6.0.1
piper-tts>=1.2.0
```

---

## Getting Started — Step by Step

### Phase 1: Setup (Day 1)

```bash
# Clone or create project directory
mkdir holmes-shorts && cd holmes-shorts

# Create directory structure
mkdir -p sources chunks scripts assets/{backgrounds,fonts,music,overlays} \
         output/{scripts,audio,visuals,captions,shorts} models/piper

# Install dependencies (see Installation section above)

# Download a free serif font for overlays
# Cormorant Garamond works great for spiritual content
# Download from: https://fonts.google.com/specimen/Cormorant+Garamond
```

### Phase 2: Source Material (Day 1)

```bash
# Download public domain texts from Internet Archive
# Search: https://archive.org/search?query=ernest+holmes

# Place cleaned .txt files in sources/
# Run preprocessor
python scripts/preprocess_source.py
```

### Phase 3: First Short (Day 2)

```bash
# Make sure Ollama is running
ollama serve &

# Process a single chunk end-to-end
python scripts/pipeline.py --chunk 001 --work-title "The Science of Mind"

# Review the output
ffplay output/shorts/*001*_short.mp4
```

### Phase 4: Batch Production (Day 3+)

```bash
# Generate a batch of 10 Shorts
python scripts/pipeline.py --batch 10

# Review all generated scripts before producing
ls output/scripts/

# Run just the assembly step after reviewing
python scripts/pipeline.py --batch 10 --step assemble
```

### Phase 5: Upload & Iterate

- Upload first 5 Shorts to YouTube
- Monitor analytics for 7 days
- Identify which hooks/moods perform best
- Adjust prompts accordingly
- Scale to daily uploads

---

## Content Calendar Strategy

### Weekly Cadence

| Day | Content Type | Mood |
|-----|-------------|------|
| Monday | Empowering affirmation | empowering |
| Tuesday | Deep teaching excerpt | contemplative |
| Wednesday | Practical application | practical |
| Thursday | Mystical/spiritual insight | mystical |
| Friday | Challenge a common belief | challenging |
| Saturday | Success & abundance focus | empowering |
| Sunday | Contemplative meditation prompt | contemplative |

### Hashtag Strategy

```
Primary (always use):
#ScienceOfMind #ErnestHolmes #NewThought #SpiritualWisdom

Secondary (rotate):
#LawOfAttraction #Manifestation #SpiritualGrowth #MindPower
#Consciousness #Metaphysics #DivineIntelligence #SpiritualAwakening

Trending (research weekly):
#SpiritualTikTok #WisdomShorts #MindsetShift #InnerPeace
```

### Title/Description Templates

```
Title: "[HOOK FIRST 5 WORDS]..." — Ernest Holmes (1926)
Description: "[Full hook + brief context]. From Ernest Holmes' 
'The Science of Mind' (1926), public domain. 

📚 Free: Read the original at [link]
🔔 Follow for daily wisdom from Science of Mind
```

---

## Future Upgrades (V2+)

### Stable Diffusion Visuals

Replace gradient backgrounds with AI-generated contemplative imagery:

```python
# Rough integration sketch — use ComfyUI or A1111 API
def generate_sd_visual(themes: list[str], mood: str) -> str:
    prompt = f"ethereal {mood} spiritual landscape, {', '.join(themes)}, " \
             f"deep colors, mystical atmosphere, meditation, cosmic, " \
             f"no text, no people, abstract"
    negative = "text, words, letters, people, faces, cartoon, anime"
    # Call SD API...
```

### Talking Avatar (V3)

Use SadTalker or similar to generate a speaking face synced to voiceover:
- Create a consistent AI avatar as the "channel host"
- Much higher engagement than text-only Shorts
- Requires more compute but dramatically increases watch time

### Auto-Upload

Use the YouTube Data API to schedule and upload directly:

```python
# youtube_upload.py — sketch
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

def upload_short(video_path, title, description, tags):
    youtube = build('youtube', 'v3', credentials=creds)
    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": "27",  # Education
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False,
                "shorts": {"shortsEnabled": True},
            }
        },
        media_body=video_path,
    )
    response = request.execute()
```

### Analytics Feedback Loop

Track which Shorts perform best and feed that data back to the LLM:

```python
# Concept: Use YouTube Analytics API to identify top performers,
# then instruct Ollama to generate more content matching those
# themes, moods, and hook styles.
```

---

## Legal Notes

### Public Domain Confirmation

- Works published before 1928 in the US are in the public domain
- Ernest Holmes' "Creative Mind" (1919), "Creative Mind and Success" (1919), and "The Science of Mind" (1926, 1st edition) all qualify
- **The 1938 revised edition of "The Science of Mind" is NOT public domain** — use only the 1926 first edition
- When in doubt, verify publication dates against original copyright records

### What You Own

- Your hooks, closings, and original writing: **fully yours**
- The extracted Holmes passages: **public domain, free to use**
- AI-generated voiceover and visuals: **yours** (created by you using tools)
- The assembled Shorts: **fully yours, monetizable**

### Music Licensing

If using background music, ensure it's:
- Royalty-free (check license terms carefully)
- YouTube Audio Library (free, pre-cleared)
- Creative Commons (check attribution requirements)
- Original composition (use AI music generators like Suno or Udio)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Ollama returns garbled JSON | Lower temperature to 0.5, add "Respond ONLY in valid JSON" to prompt |
| Piper voice sounds robotic | Try different voice models, add "..." pauses in text |
| FFmpeg subtitle filter fails | Escape special characters in SRT, check file paths have no spaces |
| Captions are misaligned | Use Whisper "small" model instead of "base" for better timestamps |
| Video looks stretched | Ensure all components target 1080x1920 (9:16 ratio) |
| Gradient backgrounds are boring | Add particle overlay or use stock video backgrounds |
| Audio clipping/distortion | Normalize audio: `ffmpeg -i input.wav -af loudnorm output.wav` |

---

## Quick Reference Commands

```bash
# Full pipeline — all chunks
python scripts/pipeline.py

# Single chunk
python scripts/pipeline.py --chunk 005

# Batch of 10
python scripts/pipeline.py --batch 10

# Just extract scripts (for review before producing)
python scripts/pipeline.py --step extract --batch 20

# Just assemble (after reviewing scripts)
python scripts/pipeline.py --step assemble

# Preview a generated Short
ffplay output/shorts/science_of_mind_1926_chunk_005_short.mp4

# Check video specs
ffprobe -v error -show_format -show_streams output/shorts/*.mp4
```

---

*Built by ARETE — Turning century-old wisdom into modern reach.*
