# New Eden Whispers — AI-Powered EVE Chronicles YouTube Shorts Pipeline

## Project Overview

An automated content pipeline that transforms CCP's EVE Online Chronicles into cinematic YouTube Shorts. The system uses a local LLM (Ollama/Animus) to extract the most dramatic moments from EVE's fiction library, generates epic narration, pairs it with in-game visuals, and assembles finished Shorts ready for upload.

### Why This Works

- **200+ Chronicles spanning 20+ years** — massive content library, barely touched in Shorts format
- **CCP explicitly allows monetization** via passive ad revenue (YouTube Shorts qualifies)
- **Built-in audience** — millions of current/former EVE players hungry for lore content
- **Existing narrators do long-form only** — the Shorts niche is wide open
- **Your deep EVE knowledge** is a massive competitive moat
- **Cross-promotes your EVE projects** — Gatekeeper, Rebellion, and future builds
- **Path to CCP Partnership Program** — game time, exclusive SKINs, dev access

### Competitive Landscape

| Existing Creator | Format | Gap You Fill |
|-----------------|--------|-------------|
| The Eve Reader (Zendane) | Long-form podcast (30-60 min) | No Shorts, inactive periods |
| Chronicles of New Eden (Dame du Nord) | Podcast + YouTube long-form | No Shorts format |
| Various forum readers | Text posts, Soundcloud | No video, no production value |

**Nobody is doing cinematic EVE lore Shorts with AI-generated narration and in-game visuals.** The field is open.

### CCP Content Creation Terms — What's Allowed

Per [CCP's Content Creation Terms of Use](https://support.eveonline.com/hc/en-us/articles/8563917741084):

**✅ ALLOWED:**
- Use CCP storylines, characters, assets, in-game footage
- Monetize via passive advertisement (YouTube ad revenue, pre-roll ads, sponsor overlays)
- Create derivative content (narrations, analysis, dramatizations)

**❌ NOT ALLOWED:**
- Block behind paywall or premium subscription
- Receive direct profits from sales of the content
- Claim CCP affiliation or endorsement

**🎯 GOAL:** Build channel → Apply to [EVE Partnership Program](https://www.eveonline.com/partners) → Get official CCP support, game time, exclusive SKINs, dev access.

---

## Monetization Strategy

### Revenue Layers

| Layer | Vehicle | Revenue | Timeline |
|-------|---------|---------|----------|
| Free | YouTube Shorts (daily) | Ad rev + growth | Month 1+ |
| Free | YouTube long-form (weekly deep dives) | Higher RPM | Month 2+ |
| Free | Podcast version (Spotify/Apple) | Ad rev + reach | Month 2+ |
| Partnership | CCP Partnership Program | Game time, SKINs, promo | 5K+ subs |
| Affiliate | EVE starter packs, PLEX referral links | Commission | Month 3+ |
| Merch | EVE-inspired designs (original art only) | Direct sales | 10K+ subs |
| Cross-promo | Drive traffic to EVE_Gatekeeper, EVE_Rebellion | Portfolio value | Ongoing |
| Consulting | "I built an AI content pipeline" — sell the capability | $$$$ | Anytime |

### The Real Play

The channel itself generates modest revenue. The **strategic value** is:

1. **CCP Partnership** — official recognition, dev access, exclusive content
2. **Portfolio piece** — "autonomous AI content pipeline" for job interviews
3. **Community authority** — become the go-to EVE lore voice
4. **Cross-promotion engine** — funnel viewers to your EVE development projects
5. **Pipeline-as-product** — license the system to other gaming communities

---

## Content Library — The Source Material

### EVE Fiction Portal: https://universe.eveonline.com/

| Content Type | Quantity | URL | Best For |
|-------------|----------|-----|----------|
| Chronicles | 200+ stories | universe.eveonline.com/chronicles | Core Shorts content |
| Short Stories | 20+ extended fiction | universe.eveonline.com/short-stories | Multi-part series |
| Lore Articles | 1000+ articles | universe.eveonline.com/lore | Explainer Shorts |
| News Articles | Ongoing | universe.eveonline.com/news | Current events tie-ins |
| Scientific Articles | Dozens | universe.eveonline.com/scientific-articles | Deep dive content |

### Chronicles by Era (Content Planning)

```
YC102 (41 chronicles) — Early New Eden, foundation stories
YC103 (24)  — Empire conflicts, faction origins  
YC105 (8)   — Expansion era
YC106 (10)  — Pirate factions, underworld
YC107 (5)   — Capsuleer emergence
YC108 (9)   — Technology & society
YC109 (27)  — Major conflicts
YC110 (33)  — War & politics
YC111 (26)  — Intrigue & espionage
YC112 (29)  — Empire machinations
YC113 (13)  — Modern era
YC114 (4)   — Recent events
YC115 (4)   — Drifter emergence
YC116 (5)   — Escalation
YC117 (2)   — Current tensions
YC119 (4)   — Latest chronicles
YC120 (2)   — Recent
YC126 (1)   — Most recent
```

**That's 200+ chronicles = 200+ potential Shorts minimum**, plus many chronicles are long enough for multiple Shorts.

### High-Value Chronicles for Launch

These are the most dramatic, engaging chronicles — perfect for your first batch:

| Chronicle | Why It's Great for Shorts |
|-----------|--------------------------|
| "The Jovian Wetgrave" | Body horror, mystery, the Jove enigma |
| "Theodicy" | Amarr religious extremism, moral complexity |
| "Hands of a Killer" | Assassination, betrayal, visceral action |
| "The Breakout" | Prison escape, Minmatar rebellion |
| "All These Lives Are Fit to Ruin" | Dark, emotional, capsuleer psychology |
| "Xenocracy" | Capsuleer god-complex, planetary bombardment threat |
| "The Plague Years" | Biological warfare, desperation |
| "Innocent Faces" | Slavery, moral horror, emotional gut-punch |
| "The Speaker of Truths" | Minmatar spirituality, identity |
| "Anoikis" | Wormhole space discovery, cosmic mystery |

---

## Architecture

### Pipeline Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  EVE Fiction  │────▶│ Ollama/Animus│────▶│   Script:    │
│  Portal Text  │     │  (extract    │     │ Hook + Scene │
│  (scraper)    │     │   dramatic   │     │ + Closing    │
└──────────────┘     │   moments)   │     └──────┬───────┘
                     └──────────────┘            │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  EVE Visual  │────▶│   FFmpeg     │◀────│  TTS Engine  │
│  Library     │     │  (assembly)  │     │ (epic voice) │
│ (screenshots,│     └──────┬───────┘     └──────────────┘
│  footage,    │            │
│  trailers)   │            ▼
└──────────────┘     ┌──────────────┐
                     │  Whisper     │
                     │  (captions)  │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Final MP4   │
                     │  1080x1920   │
                     │  < 60 sec    │
                     └──────────────┘
```

### Tech Stack

| Component | Tool | Cost | Notes |
|-----------|------|------|-------|
| LLM | Ollama (Animus) | Free | Passage extraction + hooks |
| Web Scraper | Python (requests + BeautifulSoup) | Free | Pull chronicles from fiction portal |
| TTS (MVP) | Piper TTS | Free | Local, decent quality |
| TTS (Upgrade) | ElevenLabs | ~$5-22/mo | Dramatically better for dramatic narration |
| TTS (Alt) | Coqui XTTS | Free | Local, voice cloning capable |
| Visuals | EVE screenshots + trailer clips | Free | CCP allows in-game footage |
| Visuals (enhance) | Stable Diffusion | Free (GPU) | Generate supplementary sci-fi imagery |
| Captions | Whisper | Free | Word-level timestamps |
| Assembly | FFmpeg | Free | Final compositing |
| Music | EVE soundtracks / royalty-free | Free | Ambient atmosphere |
| Orchestration | Python | Free | Pipeline automation |

---

## Directory Structure

```
new-eden-whispers/
├── README.md
├── requirements.txt
├── config.yaml
├── scripts/
│   ├── scrape_chronicles.py      # Pull chronicle texts from EVE fiction portal
│   ├── extract_scene.py          # Ollama dramatic moment extraction
│   ├── generate_narration.py     # TTS voiceover generation
│   ├── generate_captions.py      # Whisper caption generation  
│   ├── select_visuals.py         # Match visuals to scene themes
│   ├── assemble_short.py         # FFmpeg final assembly
│   ├── pipeline.py               # Full orchestration
│   └── upload.py                 # YouTube API upload (future)
├── sources/
│   ├── chronicles/               # Scraped chronicle texts (JSON)
│   ├── lore_articles/            # Scraped lore articles
│   └── chronicle_index.json      # Master index of all chronicles
├── assets/
│   ├── visuals/
│   │   ├── amarr/                # Amarr-themed screenshots/footage
│   │   ├── caldari/              # Caldari-themed
│   │   ├── gallente/             # Gallente-themed
│   │   ├── minmatar/            # Minmatar-themed
│   │   ├── jove/                 # Jove/Sleeper/Drifter
│   │   ├── pirate/               # Angel Cartel, Serpentis, Guristas, etc.
│   │   ├── wormhole/             # Anoikis, wormhole space
│   │   ├── combat/               # Battle scenes
│   │   ├── stations/             # Station interiors, citadels
│   │   └── generic_space/        # Nebulae, gates, general space
│   ├── fonts/
│   ├── music/                    # EVE-style ambient tracks
│   └── overlays/                 # Channel branding, watermarks
├── output/
│   ├── scripts/                  # Generated Short scripts (JSON)
│   ├── audio/                    # Voiceover files
│   ├── captions/                 # SRT subtitle files
│   └── shorts/                   # Final MP4 files
└── models/
    └── piper/                    # TTS voice models
```

---

## Core Scripts

### 1. Chronicle Scraper — `scripts/scrape_chronicles.py`

```python
#!/usr/bin/env python3
"""
Scrape EVE Online Chronicles from the fiction portal.
Stores each chronicle as a JSON file with metadata.

Source: https://universe.eveonline.com/chronicles
"""
import json
import re
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

BASE_URL = "https://universe.eveonline.com"
CHRONICLES_URL = f"{BASE_URL}/chronicles"
OUTPUT_DIR = Path("sources/chronicles")
INDEX_FILE = Path("sources/chronicle_index.json")

# Respectful scraping — don't hammer CCP's servers
REQUEST_DELAY = 2  # seconds between requests
HEADERS = {
    "User-Agent": "NewEdenWhispers-ContentBot/1.0 (EVE Chronicle Research)"
}


def get_chronicle_list() -> list[dict]:
    """Fetch the master list of all chronicles."""
    print("[→] Fetching chronicle index...")
    response = requests.get(CHRONICLES_URL, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    chronicles = []
    # Parse chronicle links — adjust selectors based on actual page structure
    for link in soup.select("a[href*='/chronicles/']"):
        href = link.get("href", "")
        if href and href != "/chronicles/":
            title = link.get_text(strip=True)
            if title:
                chronicles.append({
                    "title": title,
                    "url": f"{BASE_URL}{href}" if href.startswith("/") else href,
                    "slug": href.split("/")[-1],
                })

    print(f"[✓] Found {len(chronicles)} chronicles")
    return chronicles


def scrape_chronicle(chronicle: dict) -> dict:
    """Scrape a single chronicle's full text and metadata."""
    print(f"  [→] Scraping: {chronicle['title']}")
    time.sleep(REQUEST_DELAY)

    response = requests.get(chronicle["url"], headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the main content — adjust selectors to match actual page structure
    content_div = soup.select_one("article, .chronicle-content, .content, main")
    
    if content_div:
        # Get clean text, preserving paragraph breaks
        paragraphs = content_div.find_all("p")
        text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    else:
        text = ""

    # Try to extract metadata
    meta = {
        "title": chronicle["title"],
        "slug": chronicle["slug"],
        "url": chronicle["url"],
        "text": text,
        "word_count": len(text.split()),
        "paragraph_count": len(text.split("\n\n")),
    }

    # Extract era/date if present
    date_elem = soup.select_one(".date, .chronicle-date, time")
    if date_elem:
        meta["era"] = date_elem.get_text(strip=True)

    # Extract faction tags if present
    tags = [tag.get_text(strip=True) for tag in soup.select(".tag, .faction-tag, .category")]
    if tags:
        meta["tags"] = tags

    return meta


def build_index(chronicles: list[dict]) -> None:
    """Build and save the master index."""
    index = {
        "total_chronicles": len(chronicles),
        "total_words": sum(c.get("word_count", 0) for c in chronicles),
        "chronicles": chronicles,
    }
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index, indent=2))
    print(f"\n[✓] Index saved: {INDEX_FILE}")
    print(f"    Total chronicles: {index['total_chronicles']}")
    print(f"    Total words: {index['total_words']:,}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get list of all chronicles
    chronicle_list = get_chronicle_list()

    # Scrape each chronicle
    full_chronicles = []
    for i, chronicle in enumerate(chronicle_list):
        try:
            full = scrape_chronicle(chronicle)
            if full["text"]:
                # Save individual chronicle
                out_file = OUTPUT_DIR / f"{full['slug']}.json"
                out_file.write_text(json.dumps(full, indent=2))
                full_chronicles.append(full)
                print(f"  [✓] Saved ({full['word_count']} words)")
            else:
                print(f"  [!] No text found — skipping")
        except Exception as e:
            print(f"  [✗] Failed: {e}")

        # Progress
        if (i + 1) % 10 == 0:
            print(f"\n--- Progress: {i+1}/{len(chronicle_list)} ---\n")

    # Build master index
    build_index(full_chronicles)


if __name__ == "__main__":
    main()
```

### 2. Scene Extraction — `scripts/extract_scene.py`

This is the critical differentiator. The LLM needs to understand EVE's tone and find the most **cinematic** moments.

```python
#!/usr/bin/env python3
"""
Extract the most dramatic, cinematic moments from EVE Chronicles
using Ollama. Produces structured Short scripts optimized for
60-second YouTube Shorts with epic sci-fi narration.
"""
import json
import requests
import sys
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"  # Replace with your Animus model

SYSTEM_PROMPT = """You are a cinematic narrator for EVE Online's universe — New Eden. 
Your voice channels the gravitas of a sci-fi documentary narrator crossed with 
a war correspondent. Think: the opening crawl of a space opera, the tone of 
"World War II in Color" but set 20,000 years in the future.

Your job is to extract the single most DRAMATIC, CINEMATIC moment from an EVE 
Chronicle and transform it into a 40-55 second YouTube Short script.

RULES:
1. The scene must stand alone — a viewer who has NEVER played EVE must understand it
2. Open with a HOOK that creates immediate tension, mystery, or awe (first 3 seconds)
3. The hook should be YOUR words — not from the chronicle directly
4. Provide brief context if needed (1-2 sentences max) to set the scene
5. The core passage should be adapted for SPOKEN delivery — tighten prose, add dramatic pauses
6. End with a gut-punch closing line or cliffhanger
7. Tag with factions, themes, and visual cues for video assembly
8. 100-140 words total spoken content (aiming for 40-55 seconds at narration pace)
9. If the chronicle has dialogue, include the most powerful line
10. Lean into EVE's themes: mortality, power, identity, empire, rebellion, cosmic horror

FACTION CONTEXT (for accurate tagging):
- Amarr Empire: Religious zealots, slavers, golden ships, scripture
- Caldari State: Corporate militarists, efficiency, gray/blue aesthetic
- Gallente Federation: Democratic idealists, freedom, culture, green/blue
- Minmatar Republic: Former slaves, tribal, rust-and-fire aesthetic, rebellion
- Jove Empire: Mysterious, advanced, dying race, body modification
- Drifters/Sleepers: Ancient, terrifying, wormhole entities
- Pirate factions: Serpentis, Guristas, Angel Cartel, Blood Raiders, Sansha's Nation

Respond ONLY in this JSON format:
{
    "hook": "The scroll-stopping opening line (your original dramatic words)",
    "context": "1-2 sentences of setup if needed for non-EVE viewers (or null)",
    "narration": "The core dramatic passage, adapted for spoken delivery with [pause] markers",
    "closing": "A punchy closing line — cliffhanger, revelation, or emotional hit",
    "chronicle_title": "Original chronicle title",
    "factions": ["faction1", "faction2"],
    "themes": ["theme1", "theme2", "theme3"],
    "visual_cues": ["specific visual suggestion 1", "visual 2", "visual 3"],
    "mood": "epic|dark|mysterious|tragic|brutal|contemplative|horrifying",
    "has_dialogue": true/false,
    "estimated_duration_seconds": 45,
    "word_count": 120
}"""

USER_PROMPT_TEMPLATE = """Here is an EVE Online Chronicle titled "{title}".

Extract the single most cinematic, dramatic moment and create a YouTube Short 
script from it. Remember: this needs to hook someone scrolling in 3 seconds 
and hold them for 45-55 seconds.

---
{text}
---

JSON only. Make it epic."""


def extract_scene(chronicle: dict) -> dict:
    """Send chronicle to Ollama and get structured Short script back."""
    # Truncate very long chronicles to fit context window
    text = chronicle["text"]
    if len(text) > 8000:
        text = text[:8000] + "\n\n[Chronicle continues...]"

    payload = {
        "model": MODEL,
        "prompt": USER_PROMPT_TEMPLATE.format(
            title=chronicle["title"],
            text=text,
        ),
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_ctx": 8192,
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    raw = response.json()["response"]

    # Parse JSON from response
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]

    result = json.loads(raw.strip())
    result["source_url"] = chronicle.get("url", "")
    return result


def process_chronicles(chronicles_dir: str, output_dir: str, limit: int = None):
    """Process all scraped chronicles."""
    chron_path = Path(chronicles_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(chron_path.glob("*.json"))
    if limit:
        files = files[:limit]

    for chron_file in files:
        print(f"[→] Processing: {chron_file.stem}")
        chronicle = json.loads(chron_file.read_text())

        if chronicle.get("word_count", 0) < 100:
            print(f"  [!] Too short ({chronicle.get('word_count', 0)} words) — skipping")
            continue

        try:
            script = extract_scene(chronicle)
            out_file = out_path / f"{chron_file.stem}_script.json"
            out_file.write_text(json.dumps(script, indent=2))
            print(f"  [✓] {script.get('mood', '?')} | {script.get('word_count', '?')} words")
            print(f"      Hook: {script['hook'][:70]}...")
            print(f"      Factions: {', '.join(script.get('factions', []))}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [✗] Parse failed: {e}")
        except requests.RequestException as e:
            print(f"  [✗] Ollama error: {e}")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    process_chronicles(
        chronicles_dir="sources/chronicles",
        output_dir="output/scripts",
        limit=limit,
    )
```

### 3. Narration Generation — `scripts/generate_narration.py`

```python
#!/usr/bin/env python3
"""
Generate epic narration voiceover from Short scripts.

For EVE content, voice quality matters MORE than for most niches.
The narration needs to feel like a sci-fi documentary or audiobook.

MVP: Piper TTS (free, local, decent)
Recommended: ElevenLabs (paid, dramatically better for dramatic content)
Alternative: Coqui XTTS (free, local, voice cloning)
"""
import json
import subprocess
import sys
from pathlib import Path

# ── Configuration ──────────────────────────────────────────
ENGINE = "piper"  # "piper", "elevenlabs", or "coqui"

# Piper config
PIPER_MODEL = "models/piper/voice-en-us-lessac-medium.onnx"

# ElevenLabs config (recommended for production)
ELEVENLABS_API_KEY = ""       # Set via env var or config
ELEVENLABS_VOICE_ID = ""      # Pick a deep, dramatic male voice
# Recommended voices: "Adam" (deep narrator), "Antoni" (warm authority)
# Or clone your own voice for brand consistency

# Coqui XTTS config
COQUI_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
COQUI_SPEAKER_WAV = ""  # Reference audio for voice cloning


def build_narration_text(script: dict) -> str:
    """Assemble the full narration with dramatic pacing markers."""
    parts = []

    # Hook — spoken with urgency
    if script.get("hook"):
        parts.append(script["hook"])
        parts.append("...")  # Brief pause after hook

    # Context — measured, informational
    if script.get("context"):
        parts.append(script["context"])
        parts.append("...")

    # Core narration — dramatic delivery
    if script.get("narration"):
        # Replace [pause] markers with actual pause indicators
        narration = script["narration"].replace("[pause]", "...")
        parts.append(narration)

    # Closing — weighted, final
    if script.get("closing"):
        parts.append("...")  # Beat before closing
        parts.append(script["closing"])

    return " ".join(parts)


def generate_piper(text: str, output_path: str):
    """Generate with Piper TTS (local, free)."""
    cmd = [
        "piper",
        "--model", PIPER_MODEL,
        "--output_file", output_path,
        "--length_scale", "1.1",  # Slightly slower for dramatic effect
    ]
    process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate(input=text.encode("utf-8"))
    if process.returncode != 0:
        raise RuntimeError(f"Piper failed: {stderr.decode()}")


def generate_elevenlabs(text: str, output_path: str):
    """Generate with ElevenLabs API (cloud, premium quality)."""
    import requests as req

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2",
        "voice_settings": {
            "stability": 0.4,           # Lower = more expressive
            "similarity_boost": 0.8,
            "style": 0.5,               # Dramatic expressiveness
            "use_speaker_boost": True,
        }
    }
    response = req.post(url, json=payload, headers=headers)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)


def generate_coqui(text: str, output_path: str):
    """Generate with Coqui XTTS (local, voice cloning capable)."""
    from TTS.api import TTS

    tts = TTS(model_name=COQUI_MODEL, gpu=True)
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=COQUI_SPEAKER_WAV if COQUI_SPEAKER_WAV else None,
        language="en",
    )


def process_scripts(scripts_dir: str, output_dir: str):
    """Generate narration for all script files."""
    scripts_path = Path(scripts_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    engines = {
        "piper": generate_piper,
        "elevenlabs": generate_elevenlabs,
        "coqui": generate_coqui,
    }
    generate_fn = engines[ENGINE]

    for script_file in sorted(scripts_path.glob("*_script.json")):
        print(f"[→] Narrating: {script_file.stem}")
        script = json.loads(script_file.read_text())
        text = build_narration_text(script)

        out_file = out_path / f"{script_file.stem.replace('_script', '')}.wav"
        try:
            generate_fn(text, str(out_file))
            print(f"  [✓] Audio: {out_file.name}")
        except Exception as e:
            print(f"  [✗] TTS failed: {e}")


if __name__ == "__main__":
    process_scripts(
        scripts_dir=sys.argv[1] if len(sys.argv) > 1 else "output/scripts",
        output_dir="output/audio",
    )
```

### 4. Visual Selection — `scripts/select_visuals.py`

This matches script themes/factions to your visual library.

```python
#!/usr/bin/env python3
"""
Match Short scripts to appropriate visuals from the asset library.

Strategy:
1. Use faction tags to select faction-specific visuals
2. Use mood to determine visual treatment (color grading, speed)
3. Use visual_cues from the script for specific scene matching
4. Fall back to generic space visuals if no specific match

Visual sources:
- Your own EVE screenshots (take these intentionally!)
- EVE cinematic trailers (CCP allows in-game footage)
- Generic space/nebula footage for transitions
"""
import json
import random
from pathlib import Path

ASSETS_DIR = Path("assets/visuals")

# Map factions to visual directories
FACTION_DIRS = {
    "amarr": "amarr",
    "amarr empire": "amarr",
    "caldari": "caldari",
    "caldari state": "caldari",
    "gallente": "gallente",
    "gallente federation": "gallente",
    "minmatar": "minmatar",
    "minmatar republic": "minmatar",
    "jove": "jove",
    "jove empire": "jove",
    "sleeper": "jove",
    "drifter": "jove",
    "triglavian": "jove",
    "serpentis": "pirate",
    "guristas": "pirate",
    "angel cartel": "pirate",
    "blood raiders": "pirate",
    "sansha": "pirate",
    "sansha's nation": "pirate",
}

# Mood-based color grading presets (FFmpeg filter values)
MOOD_GRADING = {
    "epic": "eq=contrast=1.2:brightness=0.02:saturation=1.3",
    "dark": "eq=contrast=1.3:brightness=-0.05:saturation=0.8,colorbalance=rs=0.1:bs=0.15",
    "mysterious": "eq=contrast=1.1:saturation=0.7,colorbalance=gs=0.1:bs=0.2",
    "tragic": "eq=contrast=1.1:brightness=-0.03:saturation=0.6",
    "brutal": "eq=contrast=1.4:brightness=-0.02:saturation=1.1,colorbalance=rs=0.2",
    "contemplative": "eq=contrast=1.0:brightness=0.01:saturation=0.9",
    "horrifying": "eq=contrast=1.3:brightness=-0.08:saturation=0.5,colorbalance=bs=0.2",
}


def get_visual_files(faction_dirs: list[str]) -> list[Path]:
    """Get all visual files matching faction directories."""
    files = []
    for faction_dir in faction_dirs:
        dir_path = ASSETS_DIR / faction_dir
        if dir_path.exists():
            files.extend(dir_path.glob("*.png"))
            files.extend(dir_path.glob("*.jpg"))
            files.extend(dir_path.glob("*.mp4"))
            files.extend(dir_path.glob("*.webm"))
    return files


def select_visuals_for_script(script: dict) -> dict:
    """Select visuals and grading for a script."""
    factions = [f.lower() for f in script.get("factions", [])]
    mood = script.get("mood", "epic")
    duration = script.get("estimated_duration_seconds", 45)

    # Determine visual directories to search
    visual_dirs = []
    for faction in factions:
        if faction in FACTION_DIRS:
            visual_dirs.append(FACTION_DIRS[faction])

    # Add mood-appropriate generic directories
    if mood in ("dark", "brutal", "horrifying"):
        visual_dirs.append("combat")
    if mood in ("mysterious", "contemplative"):
        visual_dirs.append("generic_space")
    if mood in ("epic",):
        visual_dirs.extend(["combat", "generic_space"])

    # Always include generic space as fallback
    visual_dirs.append("generic_space")

    # Get available files
    available = get_visual_files(list(set(visual_dirs)))

    # Select visuals (3-5 images/clips for a 45-second Short)
    num_visuals = max(3, min(5, duration // 10))
    if len(available) >= num_visuals:
        selected = random.sample(available, num_visuals)
    elif available:
        selected = available  # Use what we have
    else:
        selected = []  # Will need generated fallback

    # Get color grading
    grading = MOOD_GRADING.get(mood, MOOD_GRADING["epic"])

    return {
        "visual_files": [str(f) for f in selected],
        "color_grading": grading,
        "mood": mood,
        "factions": factions,
        "num_selected": len(selected),
        "needs_generated_fallback": len(selected) == 0,
    }


def process_scripts(scripts_dir: str, output_dir: str):
    """Select visuals for all scripts."""
    scripts_path = Path(scripts_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for script_file in sorted(scripts_path.glob("*_script.json")):
        print(f"[→] Selecting visuals: {script_file.stem}")
        script = json.loads(script_file.read_text())

        selection = select_visuals_for_script(script)
        out_file = out_path / f"{script_file.stem.replace('_script', '_visuals')}.json"
        out_file.write_text(json.dumps(selection, indent=2))

        status = "✓" if not selection["needs_generated_fallback"] else "!"
        print(f"  [{status}] {selection['num_selected']} visuals | {selection['mood']} | grade: {selection['color_grading'][:30]}...")


if __name__ == "__main__":
    process_scripts(
        scripts_dir="output/scripts",
        output_dir="output/visual_selections",
    )
```

### 5. Caption Generation — `scripts/generate_captions.py`

```python
#!/usr/bin/env python3
"""
Generate word-level captions using Whisper.
Styled for dramatic sci-fi narration — larger text, 
strategic word grouping for emphasis.
"""
import json
import sys
from pathlib import Path


def generate_srt(audio_path: str, output_path: str, words_per_group: int = 3):
    """Generate SRT with dramatic pacing — fewer words per group for emphasis."""
    import whisper

    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True, language="en")

    words = []
    for segment in result["segments"]:
        if "words" in segment:
            words.extend(segment["words"])

    srt_entries = []
    idx = 1
    for i in range(0, len(words), words_per_group):
        group = words[i:i + words_per_group]
        start = group[0]["start"]
        end = group[-1]["end"]
        text = " ".join(w["word"].strip() for w in group).upper()  # CAPS for dramatic effect

        srt_entries.append(
            f"{idx}\n"
            f"{_fmt(start)} --> {_fmt(end)}\n"
            f"{text}\n"
        )
        idx += 1

    Path(output_path).write_text("\n".join(srt_entries))


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def process_audio(audio_dir: str, output_dir: str):
    audio_path = Path(audio_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for audio_file in sorted(audio_path.glob("*.wav")):
        print(f"[→] Captioning: {audio_file.name}")
        out_file = out_path / f"{audio_file.stem}.srt"
        try:
            generate_srt(str(audio_file), str(out_file))
            print(f"  [✓] {out_file.name}")
        except Exception as e:
            print(f"  [✗] Failed: {e}")


if __name__ == "__main__":
    process_audio(
        audio_dir=sys.argv[1] if len(sys.argv) > 1 else "output/audio",
        output_dir="output/captions",
    )
```

### 6. Final Assembly — `scripts/assemble_short.py`

```python
#!/usr/bin/env python3
"""
Assemble final EVE Chronicle YouTube Short.

Composites: background visuals (slideshow/video) + color grading +
voiceover narration + burned-in captions + ambient music + branding.

Output: 1080x1920 MP4, < 60 seconds, optimized for YouTube Shorts.
"""
import json
import subprocess
import sys
from pathlib import Path

# ── Caption Styling ──────────────────────────────────────────
# Bold, centered, sci-fi feel — white text with dark outline
CAPTION_STYLE = (
    "FontName=Impact,"
    "FontSize=24,"
    "PrimaryColour=&H00FFFFFF,"
    "SecondaryColour=&H0000FFFF,"    # Cyan accent
    "OutlineColour=&H00000000,"
    "BorderStyle=1,"
    "Outline=3,"
    "Shadow=2,"
    "ShadowColour=&H80000000,"
    "MarginV=140,"
    "Alignment=2,"
    "Bold=1"
)

MUSIC_VOLUME = 0.06  # Very subtle ambient


def get_audio_duration(audio_file: str) -> float:
    """Get duration of audio file in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_file,
    ]
    return float(subprocess.check_output(cmd).decode().strip())


def create_slideshow_video(
    images: list[str],
    duration: float,
    color_grading: str,
    output_path: str,
    width: int = 1080,
    height: int = 1920,
):
    """Create a Ken Burns slideshow from images with color grading."""
    if not images:
        # Fallback: create dark space background
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i",
            f"color=c=#0a0a1a:s={width}x{height}:d={duration}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            output_path,
        ]
        subprocess.run(cmd, check=True)
        return

    # Calculate duration per image
    dur_per = max(3, duration / len(images))

    # Build input args and filter chain
    inputs = []
    filter_parts = []
    for i, img in enumerate(images):
        inputs.extend(["-loop", "1", "-t", str(dur_per), "-i", img])
        # Ken Burns: slow zoom + pan for each image
        filter_parts.append(
            f"[{i}:v]scale={width*2}:{height*2},"
            f"zoompan=z='min(zoom+0.0008,1.15)':"
            f"x='iw/2-(iw/zoom/2)+{20 * (i % 3 - 1)}':"
            f"y='ih/2-(ih/zoom/2)':"
            f"d={int(dur_per*30)}:s={width}x{height}:fps=30,"
            f"{color_grading}[v{i}]"
        )

    # Concatenate all clips
    concat_inputs = "".join(f"[v{i}]" for i in range(len(images)))
    filter_chain = ";".join(filter_parts)
    filter_chain += f";{concat_inputs}concat=n={len(images)}:v=1:a=0[vout]"

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_chain,
        "-map", "[vout]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-t", str(duration),
        output_path,
    ]
    subprocess.run(cmd, check=True)


def assemble_short(
    visual_selection: dict,
    audio_file: str,
    caption_file: str,
    output_file: str,
    music_file: str = None,
):
    """Assemble all components into a final Short."""
    duration = get_audio_duration(audio_file) + 2.0  # Padding

    # Create background video from selected visuals
    bg_video = output_file.replace(".mp4", "_bg.mp4")
    create_slideshow_video(
        images=visual_selection.get("visual_files", []),
        duration=duration,
        color_grading=visual_selection.get("color_grading", ""),
        output_path=bg_video,
    )

    # Build final assembly command
    inputs = ["-i", bg_video, "-i", audio_file]
    audio_map = "1:a"

    # Add background music if available
    filter_extra = ""
    if music_file and Path(music_file).exists():
        inputs.extend(["-i", music_file])
        filter_extra = (
            f"[2:a]volume={MUSIC_VOLUME}[music];"
            f"[1:a][music]amix=inputs=2:duration=first[amixed];"
        )
        audio_map = "[amixed]"

    # Subtitle filter
    sub_filter = f"subtitles={caption_file}:force_style='{CAPTION_STYLE}'"

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-t", str(duration),
        "-filter_complex",
        f"{filter_extra}[0:v]{sub_filter}[vout]",
        "-map", "[vout]",
        "-map", audio_map,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "21",              # Slightly higher quality for detail
        "-c:a", "aac",
        "-b:a", "192k",           # Higher audio quality for narration
        "-ar", "44100",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_file,
    ]

    print(f"[→] Assembling: {output_file}")
    subprocess.run(cmd, check=True)

    # Cleanup temp
    Path(bg_video).unlink(missing_ok=True)
    print(f"[✓] Done! Duration: {duration:.1f}s")


def process_all(
    scripts_dir: str = "output/scripts",
    audio_dir: str = "output/audio",
    visuals_dir: str = "output/visual_selections",
    captions_dir: str = "output/captions",
    output_dir: str = "output/shorts",
    music_file: str = None,
):
    """Assemble all Shorts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for script_file in sorted(Path(scripts_dir).glob("*_script.json")):
        stem = script_file.stem.replace("_script", "")

        audio = Path(audio_dir) / f"{stem}.wav"
        vis_sel = Path(visuals_dir) / f"{stem}_visuals.json"
        caption = Path(captions_dir) / f"{stem}.srt"
        output = out_path / f"{stem}_short.mp4"

        missing = [f for f in [audio, vis_sel, caption] if not f.exists()]
        if missing:
            print(f"  [!] Missing for {stem}: {[f.name for f in missing]}")
            continue

        visual_selection = json.loads(vis_sel.read_text())

        try:
            assemble_short(visual_selection, str(audio), str(caption), str(output), music_file)
        except Exception as e:
            print(f"  [✗] Assembly failed for {stem}: {e}")


if __name__ == "__main__":
    music = sys.argv[1] if len(sys.argv) > 1 else None
    process_all(music_file=music)
```

### 7. Pipeline Orchestrator — `scripts/pipeline.py`

```python
#!/usr/bin/env python3
"""
New Eden Whispers — Full Pipeline Orchestrator

Usage:
    python scripts/pipeline.py                          # Process all chronicles
    python scripts/pipeline.py --chronicle xenocracy    # Single chronicle
    python scripts/pipeline.py --batch 10               # First N chronicles
    python scripts/pipeline.py --step extract           # Run single step
    python scripts/pipeline.py --scrape                 # Scrape chronicles first
"""
import argparse
import json
import sys
from pathlib import Path


def run_step(step: str, *args, **kwargs):
    """Dynamically run a pipeline step."""
    if step == "scrape":
        from scrape_chronicles import main as scrape_main
        scrape_main()
    elif step == "extract":
        from extract_scene import extract_scene
        return extract_scene(*args, **kwargs)
    elif step == "narrate":
        from generate_narration import build_narration_text, generate_piper
        text = build_narration_text(kwargs["script"])
        generate_piper(text, kwargs["output_path"])
    elif step == "visuals":
        from select_visuals import select_visuals_for_script
        return select_visuals_for_script(kwargs["script"])
    elif step == "captions":
        from generate_captions import generate_srt
        generate_srt(kwargs["audio_path"], kwargs["output_path"])
    elif step == "assemble":
        from assemble_short import assemble_short
        assemble_short(**kwargs)


def process_chronicle(chronicle_file: Path, music_file: str = None, steps: list = None):
    """Run full pipeline for a single chronicle."""
    stem = chronicle_file.stem
    all_steps = ["extract", "narrate", "visuals", "captions", "assemble"]
    steps = steps or all_steps

    print(f"\n{'='*60}")
    print(f"  Processing: {stem}")
    print(f"{'='*60}")

    # Output paths
    paths = {
        "script": Path(f"output/scripts/{stem}_script.json"),
        "audio": Path(f"output/audio/{stem}.wav"),
        "vis_sel": Path(f"output/visual_selections/{stem}_visuals.json"),
        "caption": Path(f"output/captions/{stem}.srt"),
        "final": Path(f"output/shorts/{stem}_short.mp4"),
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    chronicle = json.loads(chronicle_file.read_text())

    # Step 1: Extract dramatic scene
    if "extract" in steps:
        print("\n[1/5] Extracting cinematic scene...")
        try:
            script = run_step("extract", chronicle)
            paths["script"].write_text(json.dumps(script, indent=2))
            print(f"  Hook: {script['hook'][:70]}...")
            print(f"  Mood: {script['mood']} | Factions: {', '.join(script.get('factions', []))}")
        except Exception as e:
            print(f"  [✗] Extraction failed: {e}")
            return False

    # Load script
    if not paths["script"].exists():
        print(f"  [!] No script found")
        return False
    script = json.loads(paths["script"].read_text())

    # Step 2: Generate narration
    if "narrate" in steps:
        print("\n[2/5] Generating narration...")
        try:
            run_step("narrate", script=script, output_path=str(paths["audio"]))
            print(f"  [✓] Audio: {paths['audio']}")
        except Exception as e:
            print(f"  [✗] Narration failed: {e}")
            return False

    # Step 3: Select visuals
    if "visuals" in steps:
        print("\n[3/5] Selecting visuals...")
        try:
            vis = run_step("visuals", script=script)
            paths["vis_sel"].write_text(json.dumps(vis, indent=2))
            print(f"  [✓] {vis['num_selected']} visuals selected")
        except Exception as e:
            print(f"  [✗] Visual selection failed: {e}")
            return False

    # Step 4: Generate captions
    if "captions" in steps:
        print("\n[4/5] Generating captions...")
        try:
            run_step("captions", audio_path=str(paths["audio"]),
                     output_path=str(paths["caption"]))
            print(f"  [✓] Captions: {paths['caption']}")
        except Exception as e:
            print(f"  [✗] Caption generation failed: {e}")
            return False

    # Step 5: Assemble
    if "assemble" in steps:
        print("\n[5/5] Assembling Short...")
        try:
            vis = json.loads(paths["vis_sel"].read_text())
            run_step("assemble",
                     visual_selection=vis,
                     audio_file=str(paths["audio"]),
                     caption_file=str(paths["caption"]),
                     output_file=str(paths["final"]),
                     music_file=music_file)
            print(f"\n  ✅ COMPLETE: {paths['final']}")
        except Exception as e:
            print(f"  [✗] Assembly failed: {e}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="New Eden Whispers Pipeline")
    parser.add_argument("--scrape", action="store_true", help="Scrape chronicles first")
    parser.add_argument("--chronicle", help="Process specific chronicle by slug")
    parser.add_argument("--batch", type=int, help="Process N chronicles")
    parser.add_argument("--step", choices=["extract", "narrate", "visuals", "captions", "assemble"])
    parser.add_argument("--music", help="Background music file path")
    args = parser.parse_args()

    if args.scrape:
        print("Scraping chronicles from EVE fiction portal...")
        run_step("scrape")
        print()

    chronicles_dir = Path("sources/chronicles")
    files = sorted(chronicles_dir.glob("*.json"))

    if args.chronicle:
        files = [f for f in files if args.chronicle in f.stem]
        if not files:
            print(f"No chronicle matching '{args.chronicle}'")
            sys.exit(1)

    if args.batch:
        files = files[:args.batch]

    steps = [args.step] if args.step else None

    print(f"╔══════════════════════════════════════╗")
    print(f"║     NEW EDEN WHISPERS PIPELINE       ║")
    print(f"╠══════════════════════════════════════╣")
    print(f"║  Chronicles: {len(files):<23} ║")
    print(f"║  Steps: {str(steps or 'all'):<28} ║")
    print(f"╚══════════════════════════════════════╝")

    success = 0
    for f in files:
        if process_chronicle(f, music_file=args.music, steps=steps):
            success += 1

    print(f"\n{'='*60}")
    print(f"  Complete: {success}/{len(files)} Shorts generated")
    print(f"  Output:   output/shorts/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
```

---

## Ollama Prompt Engineering — EVE-Specific

### Model Selection

EVE Chronicles are darker, more complex fiction than typical content. The model needs to handle:
- Military/political intrigue
- Philosophical undertones
- Multiple factions with distinct voices
- Sci-fi terminology

**Recommended**: `llama3.1:8b` minimum — `llama3.1:70b` if you have VRAM.

### System Prompt Variants

#### Epic Cinematic (Default — best for dramatic chronicles)
```
You are the narrator of New Eden's darkest chapters. Your voice carries 
the weight of civilizations spanning 20,000 years. You find the moment 
in each chronicle where the reader's breath catches — the betrayal, 
the revelation, the horror — and you frame it for maximum impact.

Think: the narration style of Warhammer 40K lore videos meets 
the pacing of a movie trailer.
```

#### Lore Explainer (Best for articles/world-building content)
```
You are New Eden's historian. You make the complex political web of 
EVE's universe accessible to outsiders while keeping veterans engaged 
with details they might have missed. Your hooks frame familiar EVE 
concepts in ways that make non-players care.

Hook formula: "[Relatable human concept] — except it's 20,000 years 
in the future and the stakes are [specific EVE stakes]."
```

#### Horror/Mystery (Best for Jove, Sleeper, Drifter content)
```
You are a voice from the void. The chronicles you narrate deal with 
things humanity was never meant to find — ancient technologies, 
dying races, cosmic horrors lurking in wormhole space. Your tone is 
quiet dread building to revelation. Leave the viewer unsettled.
```

### Testing Prompts

```bash
# Quick single-chronicle test
cat sources/chronicles/xenocracy.json | jq -r .text | head -c 3000 | \
ollama run llama3.1:8b "$(cat scripts/prompts/epic_cinematic.txt)

Extract a YouTube Short script from this EVE Chronicle. JSON only."
```

---

## Visual Asset Strategy

### Building Your Visual Library

This is ongoing work that compounds over time. Start with these sources:

#### 1. Your Own EVE Screenshots (Highest Value)

```
Launch EVE → Ctrl+F9 (hide UI) → F12 (screenshot)

Priority shots to take:
□ Each empire's space (Amarr gold nebulae, Caldari blue-gray, etc.)
□ Fleet fights / battle wreckage
□ Station interiors and citadels
□ Gate activations
□ Wormhole space (eerie, alien)
□ Planet surfaces from orbit
□ Ship close-ups (faction-specific)
□ Explosions and weapon effects
□ Nebulae and cosmic phenomena
□ Abandoned structures
```

#### 2. CCP's Media Library

CCP provides official assets for community use:
- EVE trailer stills
- Concept art (check licensing)
- Promotional screenshots

#### 3. EVE Cinematic Trailers (Video Clips)

CCP's trailers are cinematically stunning. Under their content policy, you can use in-game footage. Cut relevant 3-5 second clips for backgrounds.

#### 4. Stable Diffusion Supplementary Visuals (V2)

For scenes that need specific imagery you don't have:

```python
# Example prompts for EVE-style SD generation
prompts = {
    "amarr": "golden cathedral spaceship interior, stained glass, divine light, sci-fi, 8k",
    "caldari": "sterile corporate military command bridge, holographic displays, blue-gray, sci-fi",
    "minmatar": "rust and fire tribal spaceship, welded metal, glowing engines, gritty sci-fi",
    "gallente": "elegant democratic senate chamber in space, green and silver, futuristic",
    "wormhole": "alien void, impossible geometry, cosmic horror, dark purple nebula, unsettling",
    "combat": "massive space battle, explosions, laser beams, debris field, epic scale",
}
```

### Organizing Visuals

```bash
# After a screenshot session, sort into faction folders
assets/visuals/
├── amarr/           # 20+ golden empire shots
├── caldari/         # 20+ corporate state shots  
├── gallente/        # 20+ federation shots
├── minmatar/        # 20+ republic shots
├── jove/            # Sleeper sites, Drifter encounters, wormholes
├── pirate/          # Pirate faction bases, ships
├── combat/          # Battles, explosions, wreckage
├── stations/        # Citadels, stations, interiors
├── wormhole/        # W-space, Thera, cosmic anomalies
└── generic_space/   # Nebulae, gates, jump effects, stars
```

**Target: 10-20 images per category before launching.** You can build this over a few play sessions.

---

## Configuration

### `config.yaml`

```yaml
# New Eden Whispers Configuration

project:
  name: "New Eden Whispers"
  tagline: "The untold stories of New Eden"
  
# Ollama
ollama:
  url: "http://localhost:11434/api/generate"
  model: "llama3.1:8b"
  temperature: 0.7
  top_p: 0.9
  num_ctx: 8192

# TTS
tts:
  engine: "piper"
  piper:
    model: "models/piper/voice-en-us-lessac-medium.onnx"
    length_scale: 1.1  # Slightly slower for dramatic delivery
  elevenlabs:
    api_key: ""
    voice_id: ""  # Choose: Adam, Antoni, or custom clone

# Visuals
visuals:
  width: 1080
  height: 1920
  fps: 30
  ken_burns_zoom_speed: 0.0008

# Captions
captions:
  whisper_model: "base"
  words_per_group: 3  # Fewer = more dramatic pacing
  style: "dramatic_caps"  # ALL CAPS for impact

# Output
output:
  format: "mp4"
  video_codec: "libx264"
  crf: 21
  audio_codec: "aac"
  audio_bitrate: "192k"

# Music
music:
  enabled: false
  file: "assets/music/ambient_space.mp3"
  volume: 0.06

# Scraping
scraping:
  base_url: "https://universe.eveonline.com"
  request_delay: 2
  user_agent: "NewEdenWhispers-Bot/1.0"

# Content constraints
content:
  min_words: 100
  max_words: 140
  target_duration: 45
  max_duration: 58  # Must stay under 60s for Shorts
```

### `requirements.txt`

```
requests>=2.31.0
beautifulsoup4>=4.12.0
Pillow>=10.0.0
openai-whisper>=20230918
pyyaml>=6.0.1
piper-tts>=1.2.0
```

---

## Getting Started — Step by Step

### Phase 1: Setup (Day 1)

```bash
mkdir new-eden-whispers && cd new-eden-whispers
mkdir -p sources/{chronicles,lore_articles} scripts \
         assets/visuals/{amarr,caldari,gallente,minmatar,jove,pirate,combat,stations,wormhole,generic_space} \
         assets/{fonts,music,overlays} \
         output/{scripts,audio,visual_selections,captions,shorts} \
         models/piper

# Install dependencies
pip install requests beautifulsoup4 Pillow openai-whisper pyyaml piper-tts
sudo apt install -y ffmpeg imagemagick

# Ollama
ollama pull llama3.1:8b
```

### Phase 2: Scrape Chronicles (Day 1)

```bash
python scripts/scrape_chronicles.py
# Review: sources/chronicle_index.json
```

### Phase 3: Build Visual Library (Day 1-2)

```bash
# Play EVE for 1-2 hours with UI hidden (Ctrl+F9)
# Take 100+ screenshots across factions/environments
# Sort into assets/visuals/ subdirectories
```

### Phase 4: First Short (Day 2)

```bash
ollama serve &
python scripts/pipeline.py --chronicle xenocracy
ffplay output/shorts/xenocracy_short.mp4
```

### Phase 5: First Batch (Day 3)

```bash
python scripts/pipeline.py --batch 10
# Review output/scripts/ — curate the best ones
# Re-run assembly on approved scripts
```

### Phase 6: Launch (Day 4+)

- Upload first 5-7 Shorts
- Post in r/Eve, EVE Forums, EVE Discord
- Monitor analytics for 7 days
- Double down on what performs

---

## Content Calendar

### Weekly Cadence

| Day | Content Theme | Mood | Target Faction |
|-----|--------------|------|----------------|
| Mon | Empire politics & betrayal | dark | Amarr/Caldari |
| Tue | Minmatar rebellion & freedom | epic | Minmatar |
| Wed | Cosmic mystery & horror | mysterious | Jove/Drifter |
| Thu | Pirate underworld | brutal | Pirate factions |
| Fri | Capsuleer psychology | contemplative | Any |
| Sat | Epic battles & warfare | epic | Any |
| Sun | Lore deep dive (article-based) | contemplative | Rotating |

### Hashtag Strategy

```
Primary (every Short):
#EVEOnline #NewEden #EVELore #Capsuleer

Secondary (rotate):
#SpaceLore #SciFi #GamingLore #MMO #EVEChronicles
#AmarrEmpire #CaldariState #MinmatarRepublic #GallenteFederation

Trending gaming tags:
#GamingShorts #GameLore #SpaceGames #MMORPG
```

### Title Templates

```
"[HOOK FIRST 5 WORDS]..." | EVE Online Lore
"The Darkest Secret in New Eden" | EVE Chronicles
"Why the Amarr Enslave Millions" | EVE Online
"This Is How Capsuleers ACTUALLY Die" | EVE Lore
```

### Description Template

```
[Full hook text]

From the EVE Online Chronicles — the untold stories of New Eden.

📖 Read the full chronicle: [link to fiction portal]
🚀 Play EVE Online free: [affiliate/referral link]
🔔 Subscribe for daily New Eden lore

#EVEOnline #EVELore #SciFi #GamingLore

---
EVE Online content created under CCP's Content Creation Terms of Use.
EVE Online and all associated logos and designs are the intellectual
property of CCP hf.
```

---

## Growth Milestones & CCP Partnership Path

| Milestone | Action | Timeline |
|-----------|--------|----------|
| 0-100 subs | Daily Shorts, engage in EVE subreddit/forums | Month 1-2 |
| 500 subs | Start weekly long-form lore deep dives | Month 2-3 |
| 1K subs | Launch podcast version (Spotify/Apple) | Month 3-4 |
| 2K subs | Add community polls for which chronicle to cover next | Month 4-5 |
| 5K subs | Apply for CCP Partnership Program | Month 5-7 |
| 10K subs | Partnership tier perks, potential Fanfest coverage | Month 8-12 |
| 25K subs | Sponsorship opportunities, established authority | Year 1-2 |

---

## Cross-Promotion with Your EVE Projects

| Your Project | Cross-Promo Angle |
|-------------|------------------|
| EVE_Gatekeeper (starmap app) | "Built a tool to explore the very star systems in this chronicle" |
| EVE_Rebellion (arcade shooter) | "Inspired by chronicles like this — play it free" |
| Future EVE tools | Premiere announcements to your lore audience |
| Portfolio / Job search | "I built an AI content pipeline for this channel" |

---

## Future Upgrades

### V2: Multi-Voice Narration
Use ElevenLabs or Coqui to create distinct voices for:
- Narrator (deep, authoritative)
- Amarr characters (formal, imperial)
- Minmatar characters (passionate, defiant)
- Caldari characters (clipped, efficient)

### V3: Full Dramatization
- Sound effects (explosions, alarms, ship engines)
- Multiple voice actors
- Dynamic camera movements on visuals
- Music scoring per scene

### V4: Interactive / AI-Powered
- Viewer-voted chronicle selection
- AI-generated "what if" alternate histories
- Live Q&A about lore powered by Ollama

### V5: Long-Form Expansion
- 10-15 minute YouTube videos covering full chronicles
- Multi-part series for longer stories (Black Mountain)
- Documentary-style faction deep dives

---

## Legal Checklist

- [x] CCP Content Creation Terms allow YouTube monetization via ad revenue
- [x] CCP allows use of storylines, characters, and in-game assets
- [ ] Include CCP attribution in every video description
- [ ] Never claim CCP endorsement or affiliation
- [ ] Never paywall the content
- [ ] Never sell the content directly
- [ ] Background music must be royalty-free or properly licensed
- [ ] Apply to Partnership Program once eligible

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Scraper gets blocked | Increase REQUEST_DELAY, check if CCP added rate limiting |
| Ollama produces bad hooks | Tighten system prompt, lower temperature, try different model |
| Narration too robotic | Switch to ElevenLabs, adjust Piper length_scale |
| Visuals look repetitive | Take more diverse screenshots, add Stable Diffusion supplementary images |
| Captions misaligned | Use Whisper "small" model, reduce words_per_group to 2 |
| Short exceeds 60 seconds | Reduce max_words in config, trim padding in assembly |
| Color grading too dark | Adjust brightness values in MOOD_GRADING presets |

---

## Quick Reference

```bash
# Scrape all chronicles
python scripts/scrape_chronicles.py

# Full pipeline — all chronicles
python scripts/pipeline.py

# Single chronicle
python scripts/pipeline.py --chronicle xenocracy

# Batch of 10
python scripts/pipeline.py --batch 10

# Extract scripts only (for review)
python scripts/pipeline.py --step extract --batch 20

# Assemble only (after review)
python scripts/pipeline.py --step assemble

# Preview
ffplay output/shorts/xenocracy_short.mp4

# Check specs
ffprobe -v error -show_format output/shorts/*.mp4
```

---

*Built by ARETE — Channeling New Eden's darkest stories into 60 seconds of impact.*
*"They say capsuleers are immortal. They never mention what immortality costs."*
