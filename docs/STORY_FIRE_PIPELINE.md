# Story Fire — AI Folklore Shorts in the Spirit of The Storyteller

*"When people told themselves their past with stories, explained their present with stories, foretold the future with stories, the best place by the fire was kept for the storyteller."*

---

## Project Overview

An AI-powered content pipeline that brings world folklore and fairy tales to life as YouTube Shorts, channeling the narrative spirit of Jim Henson's *The Storyteller* — warm, conspiratorial, darkly whimsical, and profoundly human. Powered by the **Gorgon multi-agent orchestration framework**.

This is not a children's channel. Like Henson's original, this is **timeless storytelling for all ages** — tales that are "beautifully, lyrically written" with "a touch of whimsy and a dash of darkness." The audience is everyone who still has a sense of wonder.

### The Storyteller DNA

Jim Henson and Anthony Minghella created something specific and replicable in *The Storyteller*. Understanding **why** it worked is essential to getting the AI narration right:

**The Voice:**
- The Storyteller doesn't lecture — he *confides*. He leans in. He's telling you a secret by the fire.
- He's slightly irreverent, occasionally surprised by his own story. "And THEN — oh, you won't believe this part..."
- He talks in the oral tradition — repetitions, rhythms, run-on sentences that build momentum. "And he walked and he walked and he walked until his feet were raw and his heart was sore."
- He addresses the listener directly. You're *there* with him.
- There's warmth even in dark moments. Horror is delivered with a wry shake of the head, not a scream.

**The Dog:**
- The Dog is the audience surrogate — skeptical, easily frightened, asking the obvious questions
- "Did it end well?" "Was there a princess?" "I don't like this bit."
- Creates natural dialogue breaks in the narration — essential for Shorts pacing
- The Dog keeps the Storyteller honest and grounded

**The Stories:**
- Obscure European folk tales, not the Disney versions
- Darker, more complex, morally ambiguous
- Happy endings were earned, not guaranteed
- Every story had weight — consequences mattered
- The narration wove *through* the story, not just around it

**The Aesthetic:**
- Firelit warmth — ambers, deep shadows, golden light
- Medieval European texture — stone, wood, tapestry, candlelight
- Creatures that are beautiful and unsettling simultaneously
- A world that feels ancient and lived-in

### Why This Works as a YouTube Channel

| Factor | Advantage |
|--------|-----------|
| **Not "Made for Kids"** | General audience = full ad personalization, higher RPM ($2-8) |
| **Infinite public domain content** | Every culture's folklore, mythology, fables |
| **Built-in narrative hooks** | These stories survived thousands of years *because* they hook listeners |
| **Nostalgia factor** | Millennials who grew up on Henson's work are now the core YouTube demo |
| **Universal appeal** | Every culture has folklore — global audience, multi-language potential |
| **Distinctive voice** | Nobody is doing Storyteller-style narration on Shorts |
| **Rewatchability** | Good stories get replayed — especially by families |

---

## The Storyteller Voice — Prompt Engineering Bible

This is the most important section of this entire document. The narration style IS the product. Get this wrong and you have another generic fairy tale channel.

### Core System Prompt — The Storyteller

```
You are The Storyteller — an ancient, warm, slightly mischievous keeper 
of tales from every corner of the world. You sit by a crackling fire, 
a skeptical but lovable Dog at your feet, and you tell stories the way 
they were meant to be told: out loud, by the fire, with the shadows 
dancing on the walls.

YOUR VOICE:
- You CONFIDE in the listener. You lean in. You whisper the scary parts 
  and beam through the wonderful parts.
- You use the oral tradition: rhythmic repetitions, run-on momentum, 
  direct address. "And he walked and he walked and he walked..."
- You are occasionally surprised by your own story: "And THEN — oh, 
  now this is the part — then she..."
- You find dark things wryly amusing, not horrifying. A witch eating 
  children gets a raised eyebrow, not a scream.
- You love these stories. Every one. You've told them a hundred times 
  and they still delight you.
- You speak in short, punchy sentences mixed with long, rolling ones.
  Rhythm matters. Sound matters. These words are meant to be HEARD.
- You never explain a moral. The story IS the moral. Trust the listener.

THE DOG (optional interjections):
- Skeptical: "That doesn't sound right."
- Frightened: "I don't like this bit. Skip ahead."
- Impatient: "But did it END well?"
- Loyal: "Tell it again."
- The Dog's voice is distinct — shorter sentences, blunter, more modern.
  A counterweight to the Storyteller's lyricism.

WHAT YOU NEVER DO:
- Never sound like a documentary narrator
- Never explain or moralize — the story speaks for itself
- Never use modern slang (the Dog can, sparingly)
- Never rush — pauses are part of the music
- Never make it safe or sanitized — these are REAL folk tales with 
  real teeth. Darkness is part of the deal.
- Never reference the source culture academically — you don't say 
  "In Norse mythology..." You say "In the frozen north, where the 
  gods themselves walked among the ice..."

FOR YOUTUBE SHORTS (40-55 seconds):
- Open with a hook that creates an OPEN LOOP — a question, a danger,
  a wonder the listener MUST stay to resolve
- The hook should feel like catching the Storyteller mid-tale:
  "Now THIS one... this one keeps me up at night."
- Build to a single dramatic peak or revelation
- End with either: a satisfying closing ("And that is how..."), a 
  cliffhanger ("But the thing about wishes..."), or a Storyteller 
  aside ("The Dog doesn't like that ending. Neither do I, if I'm honest.")
- 100-140 spoken words total
```

### Variant Prompts by Culture

#### European Folk Tales (Default — closest to Henson's original)

```
[Add to base prompt]
You are telling tales from the old countries — Germany, Russia, Ireland, 
Scandinavia, the Balkans. Stone castles and dark forests. Witches in 
the woods and kings with terrible secrets. The stories your grandmother's 
grandmother told by candlelight. 

Setting cues: firelight, stone walls, thick forests, snow, candlelit 
chambers, crossroads at midnight, thatched cottages.
```

#### Japanese Folklore (Yōkai and Shinto Tales)

```
[Add to base prompt]
Tonight, the tales come from far across the sea — from the land of 
mist and mountains, where the spirits wear a thousand faces. Fox 
women and snow ghosts and temples where the dead still pray. These 
are stories that smell of incense and rain on bamboo.

Setting cues: paper lanterns, torii gates, mountain mist, bamboo 
forests, moonlit bridges, shrine bells, autumn leaves.
```

#### African Folklore (Anansi and Creation Tales)

```
[Add to base prompt]
Ah, NOW we go to where the stories began. Where Anansi the Spider — 
clever, terrible, wonderful Anansi — tricked even the Sky God himself. 
These are stories that beat like a drum and laugh like a river.

Setting cues: baobab trees, red earth, starlit savanna, river 
crossings, village fires, drum circles, vast skies.
```

#### Norse Mythology (Eddas)

```
[Add to base prompt]
The fire burns low and the wind howls. Good. These are stories for 
dark nights. Stories of gods who knew they would die, and fought 
anyway. Of a world built on a giant's bones. Of a wolf that would 
swallow the sun.

Setting cues: aurora borealis, frozen seas, longships, Yggdrasil, 
forge-fire, rune stones, ravens, mountain halls.
```

#### Arabian Nights (1001 Nights)

```
[Add to base prompt]
Close your eyes. Smell the spices? Hear the fountain? We are in the 
palace of a thousand and one nights, where a woman saved her own life 
by never, ever finishing the story. Clever, that. The cleverest trick 
of all.

Setting cues: desert stars, palace courtyards, market bazaars, 
flying carpets, oil lamps, mosaic tiles, oasis moonlight.
```

### Script Output Format

```json
{
    "hook": "The Storyteller's opening — mid-thought, conspiratorial, an open loop",
    "dog_reaction": "The Dog's response to the hook (or null if not used)",
    "narration": "The core tale — oral tradition style with [pause] markers and [whisper] / [louder] vocal cues",
    "dog_interjection": "Optional mid-story Dog comment (or null)",
    "closing": "The Storyteller's closing — satisfying, wry, or haunting",
    "dog_closing": "Dog's final word (or null) — often the punchline",
    "tale_title": "The name of the folk tale",
    "culture": "Origin culture/region",
    "themes": ["theme1", "theme2", "theme3"],
    "visual_cues": ["specific scene descriptions for image generation"],
    "mood": "warm_dark|whimsical|haunting|triumphant|bittersweet|terrifying_but_wry",
    "palette": "firelit_gold|moonlit_blue|forest_green|desert_amber|frost_silver",
    "estimated_duration_seconds": 48,
    "word_count": 125,
    "has_dog": true
}
```

---

## Architecture — Gorgon Multi-Agent Pipeline

### Agent Map

```
┌──────────────────────────────────────────────────────────────┐
│                     GORGON ORCHESTRATOR                       │
│                  "The Story Fire Engine"                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │
│  │ BARD      │  │ PAINTER   │  │ VOICE     │  │ SCRIBE    │ │
│  │ Agent     │  │ Agent     │  │ Agent     │  │ Agent     │ │
│  │           │  │           │  │           │  │           │ │
│  │ Ollama    │  │ Stable    │  │ TTS       │  │ Whisper   │ │
│  │ Script    │  │ Diffusion │  │ Narration │  │ Captions  │ │
│  │ Extract   │  │ Visuals   │  │ + Dog     │  │           │ │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘ │
│        │              │              │              │         │
│        ▼              ▼              ▼              ▼         │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                   SHARED STATE                            │ │
│  │  culture | mood | palette | tale | current_scene          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                │
│  │ WEAVER    │  │ KEEPER    │  │ HERALD    │                │
│  │ Agent     │  │ Agent     │  │ Agent     │                │
│  │           │  │           │  │           │                │
│  │ FFmpeg    │  │ Curate &  │  │ Upload &  │                │
│  │ Assembly  │  │ Schedule  │  │ Metadata  │                │
│  └───────────┘  └───────────┘  └───────────┘                │
└──────────────────────────────────────────────────────────────┘
```

### Agent Specifications

| Agent | Named After | Role | Input | Output |
|-------|-------------|------|-------|--------|
| **BARD** | The oral tradition | Extracts dramatic moments from folk tales via Ollama, writes Storyteller scripts | Source text | Script JSON |
| **PAINTER** | Illuminated manuscripts | Generates visual scenes via Stable Diffusion matching the tale's culture and mood | Script JSON | Images/video |
| **VOICE** | The spoken word | Generates two-voice narration (Storyteller + Dog) via TTS | Script JSON | Audio WAV |
| **SCRIBE** | The written record | Generates word-level captions via Whisper | Audio WAV | SRT file |
| **WEAVER** | The tapestry maker | Composites all elements via FFmpeg into final Short | All outputs | MP4 |
| **KEEPER** | The archive keeper | Curates content calendar, selects tales, schedules production | Config | Task queue |
| **HERALD** | The town crier | Manages YouTube upload, metadata, thumbnails, scheduling | Final MP4 | Published Short |

---

## The Two-Voice System

This is the killer differentiator. No other AI folklore channel has a **two-character narration** — the warm Storyteller and his skeptical Dog. This creates natural dialogue, pacing variation, and personality.

### Voice Assignment

| Character | Voice Quality | TTS Approach |
|-----------|--------------|-------------|
| **The Storyteller** | Warm, aged, expressive, slight rasp — think John Hurt's cadence | ElevenLabs "Adam" or custom clone with age/warmth settings |
| **The Dog** | Slightly nasal, matter-of-fact, endearing grumpiness | ElevenLabs "Charlie" or a distinctly different voice model |

### Voice Agent — Dual Character Generation

```python
"""
VOICE Agent — Two-character narration system.

Generates Storyteller and Dog voices separately, then composites 
with appropriate pauses and ambient (fire crackle, wind).
"""

class VoiceAgent:
    # Voice configurations
    STORYTELLER_VOICE = {
        "voice_id": "adam",       # Deep, warm, narrative
        "stability": 0.4,         # Lower = more expressive
        "similarity": 0.75,
        "style": 0.6,             # Dramatic range
        "speed": 0.85,            # Slightly slower than normal
    }
    
    DOG_VOICE = {
        "voice_id": "charlie",    # Distinct, slightly nasal
        "stability": 0.5,
        "similarity": 0.8,
        "style": 0.4,             # More conversational
        "speed": 1.0,             # Normal speed — blunter delivery
    }
    
    # Ambient layers
    AMBIENT = {
        "fire_crackle": "assets/audio/fire_crackle_loop.wav",
        "wind_gentle": "assets/audio/wind_gentle_loop.wav",
        "rain_soft": "assets/audio/rain_soft_loop.wav",
        "night_forest": "assets/audio/night_forest_loop.wav",
    }
    AMBIENT_VOLUME = 0.08  # Very subtle — atmosphere, not distraction

    def build_audio_sequence(self, script: dict) -> list[AudioSegment]:
        """Build the narration sequence from script."""
        sequence = []
        
        # Opening fire ambience (1 second)
        sequence.append(AudioSegment(
            type="ambient",
            source=self.AMBIENT["fire_crackle"],
            duration=1.0,
        ))
        
        # Hook (Storyteller)
        if script.get("hook"):
            sequence.append(AudioSegment(
                type="speech",
                voice="storyteller",
                text=script["hook"],
                config=self.STORYTELLER_VOICE,
            ))
        
        # Dog reaction
        if script.get("dog_reaction"):
            sequence.append(AudioSegment(type="pause", duration=0.4))
            sequence.append(AudioSegment(
                type="speech",
                voice="dog",
                text=script["dog_reaction"],
                config=self.DOG_VOICE,
            ))
            sequence.append(AudioSegment(type="pause", duration=0.3))
        
        # Core narration (Storyteller)
        if script.get("narration"):
            narration = script["narration"]
            # Process vocal cues
            segments = self._parse_vocal_cues(narration)
            for seg in segments:
                sequence.append(AudioSegment(
                    type="speech",
                    voice="storyteller",
                    text=seg["text"],
                    config={
                        **self.STORYTELLER_VOICE,
                        "style": seg.get("intensity", 0.6),
                    },
                ))
                if seg.get("pause"):
                    sequence.append(AudioSegment(
                        type="pause",
                        duration=seg["pause"],
                    ))
        
        # Dog interjection (mid-story)
        if script.get("dog_interjection"):
            sequence.append(AudioSegment(type="pause", duration=0.3))
            sequence.append(AudioSegment(
                type="speech",
                voice="dog",
                text=script["dog_interjection"],
                config=self.DOG_VOICE,
            ))
            sequence.append(AudioSegment(type="pause", duration=0.3))
        
        # Closing (Storyteller)
        if script.get("closing"):
            sequence.append(AudioSegment(type="pause", duration=0.5))
            sequence.append(AudioSegment(
                type="speech",
                voice="storyteller",
                text=script["closing"],
                config={
                    **self.STORYTELLER_VOICE,
                    "speed": 0.8,  # Slower for the closing
                },
            ))
        
        # Dog's last word
        if script.get("dog_closing"):
            sequence.append(AudioSegment(type="pause", duration=0.4))
            sequence.append(AudioSegment(
                type="speech",
                voice="dog",
                text=script["dog_closing"],
                config=self.DOG_VOICE,
            ))
        
        return sequence

    def _parse_vocal_cues(self, text: str) -> list[dict]:
        """Parse [pause], [whisper], [louder] markers from narration."""
        segments = []
        current = {"text": "", "intensity": 0.6}
        
        for part in re.split(r'(\[.*?\])', text):
            if part == "[pause]":
                if current["text"].strip():
                    segments.append(current)
                segments.append({"text": "", "pause": 0.6})
                current = {"text": "", "intensity": 0.6}
            elif part == "[whisper]":
                if current["text"].strip():
                    segments.append(current)
                current = {"text": "", "intensity": 0.2}
            elif part == "[louder]":
                if current["text"].strip():
                    segments.append(current)
                current = {"text": "", "intensity": 0.9}
            else:
                current["text"] += part
        
        if current["text"].strip():
            segments.append(current)
        
        return segments
```

---

## Visual System — The Illuminated Manuscript

The visual style should evoke **firelit illustrations come to life** — think illuminated manuscripts, Henson's practical effects aesthetic, and the warm darkness of a story told by the fire.

### Visual Style Guide

| Element | Approach |
|---------|----------|
| **Base aesthetic** | Golden-hour warmth, painterly, slightly textured — NOT photorealistic |
| **Lighting** | Always firelit or candlelit — deep shadows, warm highlights |
| **Borders** | Optional ornamental borders evoking illuminated manuscripts |
| **Movement** | Slow Ken Burns on still images, subtle particle effects (embers, dust motes) |
| **Text overlay** | Tale title in serif/uncial font, minimal — let the narration carry |
| **Color palettes** | Culture-specific (see below) |

### Culture-Specific Palettes

```python
CULTURE_PALETTES = {
    "european": {
        "sd_style": "oil painting, medieval illuminated manuscript style, "
                    "golden candlelight, rich shadows, warm amber tones, "
                    "detailed brushwork, fairy tale illustration, "
                    "Arthur Rackham inspired, Brian Froud inspired",
        "colors": ["#8B6914", "#2C1810", "#D4A574", "#1A1008", "#C9956B"],
        "ambient": "fire_crackle",
    },
    "norse": {
        "sd_style": "oil painting, viking age art style, carved wood texture, "
                    "frost and firelight, deep blue and amber, "
                    "runic borders, aurora borealis glow, stark and powerful",
        "colors": ["#1B3A4B", "#8B6914", "#C0C0C0", "#0A1628", "#4A6741"],
        "ambient": "wind_gentle",
    },
    "japanese": {
        "sd_style": "ukiyo-e woodblock print style, atmospheric mist, "
                    "ink wash painting, soft moonlight, cherry blossom pink, "
                    "paper lantern glow, delicate linework, wabi-sabi aesthetic",
        "colors": ["#2D1B2E", "#D4A574", "#8B4513", "#F5F5DC", "#9B2335"],
        "ambient": "rain_soft",
    },
    "african": {
        "sd_style": "warm earth tones, bold geometric patterns, "
                    "sunset savanna light, rich ochre and sienna, "
                    "carved wood texture, starlit sky, tribal art inspired, "
                    "powerful silhouettes against golden sky",
        "colors": ["#8B4513", "#D2691E", "#DAA520", "#2F1810", "#CD853F"],
        "ambient": "night_forest",
    },
    "arabian": {
        "sd_style": "persian miniature painting style, intricate geometric patterns, "
                    "lapis lazuli blue and gold leaf, oil lamp warmth, "
                    "palace courtyard moonlight, arabesque borders, "
                    "rich jewel tones, desert night sky",
        "colors": ["#1A237E", "#DAA520", "#800020", "#0D1117", "#C0965C"],
        "ambient": "fire_crackle",
    },
    "celtic": {
        "sd_style": "celtic knotwork borders, misty green hills, "
                    "ancient stone circles, bog and heather, "
                    "moonlit standing stones, watercolor style, "
                    "fairy ring glow, manuscript illumination",
        "colors": ["#2E5339", "#8B6914", "#4A6741", "#1A1008", "#A0785A"],
        "ambient": "wind_gentle",
    },
    "slavic": {
        "sd_style": "russian lacquer box art style, birch forest, "
                    "snow and firelight, matryoshka colors, "
                    "onion dome silhouettes, rich reds and golds, "
                    "Palekh miniature painting inspired",
        "colors": ["#8B0000", "#DAA520", "#F5F5DC", "#1A1A2E", "#228B22"],
        "ambient": "wind_gentle",
    },
    "greek": {
        "sd_style": "ancient greek pottery art style, black figure and red figure, "
                    "mediterranean light, marble and olive groves, "
                    "heroic poses, laurel wreaths, temple columns, "
                    "Aegean blue and terracotta",
        "colors": ["#C2452D", "#1A1A2E", "#DAA520", "#F5F5DC", "#4169E1"],
        "ambient": "fire_crackle",
    },
    "egyptian": {
        "sd_style": "ancient egyptian art style, papyrus texture, "
                    "gold and lapis lazuli, hieroglyphic borders, "
                    "temple torchlight, Nile moonlight, "
                    "profile perspective, sacred geometry",
        "colors": ["#DAA520", "#1A237E", "#8B4513", "#F5F5DC", "#228B22"],
        "ambient": "night_forest",
    },
    "indian": {
        "sd_style": "Mughal miniature painting style, vibrant jewel tones, "
                    "intricate floral borders, palace scenes, "
                    "lotus and peacock motifs, warm golden light, "
                    "Rajasthani art inspired, rich detail",
        "colors": ["#FF6347", "#DAA520", "#800080", "#006400", "#F5F5DC"],
        "ambient": "rain_soft",
    },
}
```

### Stable Diffusion Prompt Template

```python
def build_visual_prompt(scene: str, culture: str, mood: str) -> str:
    """Build SD prompt for a specific scene in the tale."""
    palette = CULTURE_PALETTES.get(culture, CULTURE_PALETTES["european"])
    
    base = palette["sd_style"]
    
    mood_modifiers = {
        "warm_dark": "intimate firelight, deep rich shadows, warm and foreboding",
        "whimsical": "magical sparkle, enchanted glow, playful light",
        "haunting": "eerie mist, pale moonlight, unsettling beauty",
        "triumphant": "golden dawn light, heroic composition, soaring feeling",
        "bittersweet": "fading autumn light, beautiful melancholy, gentle sadness",
        "terrifying_but_wry": "dark humor, grotesque beauty, candlelit horror",
    }
    
    mood_mod = mood_modifiers.get(mood, "warm firelight")
    
    prompt = (
        f"{scene}, {base}, {mood_mod}, "
        f"storytelling illustration, painterly, atmospheric, "
        f"no text, no words, no letters, no UI, "
        f"masterpiece, best quality, highly detailed"
    )
    
    negative = (
        "text, words, letters, watermark, signature, "
        "photorealistic, photograph, 3d render, "
        "modern, contemporary, cartoon, anime, chibi, "
        "blurry, low quality, deformed, ugly"
    )
    
    return prompt, negative
```

---

## Source Material Library

### Tier 1: Classic Collections (Public Domain)

| Collection | Culture | Stories | Source |
|-----------|---------|---------|--------|
| Grimm's Fairy Tales | German | 200+ | Project Gutenberg |
| Aesop's Fables | Greek | 600+ | Project Gutenberg |
| 1001 Arabian Nights | Persian/Arabic | 200+ | Project Gutenberg |
| Norse Eddas (Prose & Poetic) | Scandinavian | 30+ major tales | Sacred Texts |
| Metamorphoses (Ovid) | Roman | 250+ myths | Project Gutenberg |
| Irish Fairy Tales (Stephens) | Celtic | 10+ | Project Gutenberg |
| Japanese Fairy Tales (Ozaki) | Japanese | 22 | Project Gutenberg |
| Russian Fairy Tales (Afanasyev) | Slavic | 600+ | Various |
| Jataka Tales | Indian/Buddhist | 550+ | Sacred Texts |
| Anansi Stories | West African | 40+ | Various archives |
| Native American Legends | Various tribes | Hundreds | Sacred Texts |
| Egyptian Book of the Dead | Egyptian | Numerous | Sacred Texts |
| Kalevala | Finnish | 50 cantos | Project Gutenberg |
| Mabinogion | Welsh | 11 tales | Project Gutenberg |
| Panchatantra | Indian | 70+ | Various |

**Conservative estimate: 3,000+ distinct stories.** At one Short per day, that's **8+ years** of content before repeating.

### Where to Source

```
gutenberg.org          — Grimm, Aesop, Andersen, 1001 Nights, Ovid
sacred-texts.com       — Norse Eddas, Egyptian, Native American, Hindu
archive.org            — Everything else, scanned original editions
worldoftales.com       — Curated folk tale collections
fairytalez.com         — Categorized by culture
```

### Story Selection Criteria (KEEPER Agent)

The KEEPER agent selects tales based on:

```python
SELECTION_CRITERIA = {
    "has_clear_dramatic_peak": True,      # Essential for Shorts
    "can_stand_alone_in_excerpt": True,    # No excessive context needed
    "has_moral_or_twist": True,            # Satisfying in 45 seconds
    "avoids_problematic_content": True,    # No outdated stereotypes
    "balances_weekly_cultures": True,      # Rotate cultures for variety
    "balances_weekly_moods": True,         # Mix light and dark
    "not_recently_covered": True,          # No repeats within 6 months
    "obscure_preferred": True,             # Like Henson — surprise them
}
```

---

## Content Calendar

### Weekly Rotation

| Day | Culture Focus | Mood Target | Example Tales |
|-----|--------------|-------------|--------------|
| Mon | European (Grimm, Slavic) | warm_dark | The Juniper Tree, Vasilisa the Beautiful |
| Tue | Asian (Japanese, Chinese, Indian) | haunting | Yuki-Onna, The Painted Skin, Naga tales |
| Wed | African / Caribbean | whimsical | Anansi stories, Why-tales, trickster legends |
| Thu | Norse / Celtic | bittersweet | Baldur's Death, Children of Lir |
| Fri | Arabian / Persian | whimsical | Aladdin (original), Sinbad, clever judges |
| Sat | Greek / Roman / Egyptian | triumphant | Perseus, Isis and Osiris, Odysseus |
| Sun | "The Dog Picks" (viewer voted) | varies | Community choice — engagement driver |

### Series Within the Channel

| Series Name | Concept | Frequency |
|-------------|---------|-----------|
| **"The Fire Tales"** | Core Shorts — one tale per day | Daily |
| **"The Dog's Favorites"** | Community-voted tales | Weekly |
| **"Trickster Tuesdays"** | Trickster gods: Anansi, Loki, Coyote, Hermes | Weekly |
| **"Dark Fridays"** | The genuinely terrifying folk tales | Weekly |
| **"The Same Story"** | Same tale told across cultures (e.g., Cinderella variations worldwide) | Monthly |

### YouTube Metadata

**Title Template:**
```
"[HOOK PHRASE]" — [Culture] Folk Tale | Story Fire
"The girl who married Death" — Russian Folk Tale | Story Fire  
"Why Anansi has thin legs" — West African Tale | Story Fire
"A wolf at the world's end" — Norse Mythology | Story Fire
```

**Description Template:**
```
[Storyteller's hook line from the Short]

A [culture] folk tale, told the old way — by the fire, with a skeptical 
Dog for company.

🔥 Story Fire — folk tales and mythology from every corner of the world.
Subscribe for a new tale every day.

📖 Source: [Public domain source and link]
🎨 Visuals generated with AI, inspired by [culture] artistic traditions
🎙️ In the spirit of Jim Henson's The Storyteller

#Folklore #FairyTales #Mythology #[CultureTag] #StoryFire #Storytelling

---
All source material is in the public domain. Visual and audio elements 
are original AI-generated content by Story Fire.
```

---

## Production Pipeline — Step by Step

### Phase 1: Setup (Day 1-2)

```bash
mkdir story-fire && cd story-fire
mkdir -p sources/{european,norse,japanese,african,arabian,celtic,slavic,greek,egyptian,indian}
mkdir -p scripts assets/{audio,fonts,overlays,borders}
mkdir -p output/{scripts,audio,visuals,captions,shorts}
mkdir -p models/{piper,sd}

# Install dependencies
pip install requests beautifulsoup4 Pillow openai-whisper pyyaml
pip install diffusers transformers accelerate  # For Stable Diffusion
sudo apt install -y ffmpeg imagemagick

# Ollama
ollama pull llama3.1:8b
```

### Phase 2: Source Library (Day 2-3)

```bash
# Download key collections from Project Gutenberg
# Grimm's: https://www.gutenberg.org/ebooks/2591
# Aesop's: https://www.gutenberg.org/ebooks/11339
# 1001 Nights: https://www.gutenberg.org/ebooks/3435
# Japanese: https://www.gutenberg.org/ebooks/4018

# Preprocess into individual stories
python scripts/preprocess_sources.py
```

### Phase 3: First Short (Day 3)

```bash
# Run the full pipeline on one tale
python scripts/pipeline.py --tale "the_juniper_tree" --culture european

# Preview
ffplay output/shorts/the_juniper_tree_short.mp4
```

### Phase 4: Batch & Launch (Day 4+)

```bash
# Generate a week's worth
python scripts/pipeline.py --batch 7 --schedule weekly_rotation

# Review and curate
ls output/scripts/*.json  # Check Storyteller scripts
ls output/shorts/*.mp4    # Watch final Shorts
```

---

## Gorgon Integration

### Config for Story Fire

```yaml
# gorgon_story_fire.yaml

orchestrator:
  name: "StoryFire"
  mode: "batch"  # "batch" for Shorts, "live" for future livestream
  health_check_interval: 30

agents:
  bard:
    model: "llama3.1:8b"
    ollama_url: "http://localhost:11434/api/generate"
    temperature: 0.75
    system_prompt_path: "prompts/storyteller_base.txt"
    
  painter:
    model: "stable-diffusion-xl"
    width: 1080
    height: 1920
    steps: 25
    guidance_scale: 7.5
    images_per_scene: 4
    
  voice:
    engine: "elevenlabs"  # or "piper" for free
    storyteller:
      voice_id: "adam"
      stability: 0.4
      style: 0.6
      speed: 0.85
    dog:
      voice_id: "charlie"
      stability: 0.5
      style: 0.4
      speed: 1.0
    ambient_volume: 0.08
    
  scribe:
    whisper_model: "small"
    words_per_group: 3
    style: "warm_serif"  # Not aggressive caps — warm, readable
    
  weaver:
    output_format: "mp4"
    resolution: "1080x1920"
    fps: 30
    crf: 21
    
  keeper:
    schedule: "daily"
    cultures_rotation: ["european", "japanese", "african", "norse", "arabian", "greek", "celtic"]
    mood_balance: true
    avoid_recent_days: 180
    
  herald:
    platform: "youtube"
    auto_upload: false  # Manual review first, auto later
    schedule_time: "08:00"
    timezone: "America/Los_Angeles"

shared_state:
  current_culture: "european"
  current_mood: "warm_dark"
  current_palette: "firelit_gold"
  tales_told: []
  tales_queue: []
```

### Running via Gorgon

```bash
# Generate today's Short
gorgon run story_fire --mode single

# Generate a week batch
gorgon run story_fire --mode batch --count 7

# Generate with specific culture
gorgon run story_fire --mode single --override culture=japanese

# Dry run — generate script only, no media
gorgon run story_fire --mode single --step bard

# Health check
gorgon status story_fire
```

---

## Expansion: Story Fire Livestream (V2)

Once the Shorts channel is established, add a 24/7 **"Story Fire Radio"** livestream using the same Gorgon architecture:

```
┌────────────────────────────────────┐
│      STORY FIRE RADIO (24/7)       │
│                                    │
│  🔥 Crackling fire ambience        │
│  📖 Periodic tale narrations       │
│  🎨 Slowly morphing folk art       │
│  🎵 Culture-specific ambient music │
│  💬 Chat votes on next culture     │
│                                    │
│  Same Gorgon agents, live mode     │
└────────────────────────────────────┘
```

This reuses BARD, PAINTER, VOICE from Shorts but adds:
- MUSE agent (ambient music generation matching current culture)
- IRIS agent (slow visual morphing between generated folk art)
- HERMES agent (chat interaction — culture voting, tale requests)
- CHRONOS agent (time-of-day mood shifts, scheduled cultural hours)

**One Gorgon config, two output modes: batch Shorts + live radio.**

---

## The Three-Channel Gorgon Ecosystem

Story Fire sits alongside your other channels, all powered by the same framework:

```
GORGON
├── Story Fire        (Folklore — broadest audience)
│   ├── Shorts pipeline (batch)
│   └── Story Fire Radio (live)
│
├── New Eden Whispers (EVE lore — niche, passionate)
│   ├── Shorts pipeline (batch)
│   └── New Eden Radio (live)
│
└── Holmes Wisdom     (Science of Mind — spiritual)
    ├── Shorts pipeline (batch)
    └── Contemplation Radio (live)
```

All three share:
- Same BARD agent (different system prompts per channel)
- Same PAINTER agent (different style configs)
- Same VOICE agent (different voice profiles)
- Same WEAVER agent (same assembly pipeline)
- Same KEEPER/HERALD agents (different schedules/metadata)

**One framework. Three channels. Six content streams.**

---

## Branding

### Channel Name: **Story Fire**

Why it works:
- Evokes the firelit storytelling tradition
- Short, memorable, searchable
- Works across cultures (every culture has fire + stories)
- Implies warmth, community, ancient tradition
- Doesn't lock you into one culture or age group

### Channel Tagline Options

- *"Tales told the old way."*
- *"Pull up a chair. The fire's burning."*
- *"Every culture. Every age. One fire."*
- *"The Dog doesn't like this one."*

### Thumbnail Style

- Painterly illustration from the Short (SD-generated)
- Warm golden/amber border evoking firelight
- Tale title in serif font (Cormorant Garamond or EB Garamond)
- Culture icon in corner (small, subtle)
- Consistent visual identity across all cultures

---

## Monetization Roadmap

| Milestone | Action | Revenue Stream | Timeline |
|-----------|--------|---------------|----------|
| 0-1K subs | Daily Shorts, build backlog | YouTube ad rev | Month 1-3 |
| 1K subs | Launch podcast (Spotify/Apple) | Podcast ad rev | Month 3 |
| 5K subs | Merch — "Story Fire" branded items | Direct sales | Month 4-6 |
| 10K subs | Illustrated ebook of "The Dog's Favorites" | Digital product ($9.99) | Month 6-8 |
| 25K subs | Story Fire Radio livestream launch | Stream ad rev + Super Chats | Month 8-12 |
| 50K subs | Brand partnerships (book publishers, etc.) | Sponsorships | Year 1-2 |
| 100K subs | Licensing — Story Fire as a format | Licensing deals | Year 2+ |
| ??? | Pitch to streaming platforms as original series | Production deal | Dream goal |

### The Merch Play

"The Dog" becomes your mascot. Merch ideas:
- *"I don't like this bit."* — The Dog (mug, shirt)
- *"Did it end well?"* — The Dog (sticker)
- *"Pull up a chair."* — Story Fire logo (hoodie)
- Illustrated tale prints (culture-specific folk art from SD)

### The Book Play

Compile the best Storyteller scripts into an illustrated book:
- *"Story Fire: Tales Told the Old Way"*
- AI-generated illustrations for each tale
- Sold as digital ($9.99) and print-on-demand ($24.99)
- Cross-promotes the channel, the channel cross-promotes the book

---

## Legal Notes

### Copyright

- **All source folk tales are public domain** — no restrictions
- Specific modern *retellings* may be copyrighted — always go back to original sources
- Your Storyteller scripts, narrations, and visuals are **your original work**
- AI-generated visuals and audio are owned by you (check TTS platform terms)

### Not "Made for Kids"

**Critical:** Do NOT mark this channel as "Made for Kids." The content is:
- Family-friendly but NOT targeted at children under 13
- Includes dark themes (death, trickery, consequences) appropriate to folklore
- Comparable to Henson's original Storyteller — rated TV-Y7, but watched by all ages
- YouTube's algorithm treats general audience content better for monetization

The disclaimer in your About section:
```
Story Fire shares folk tales and mythology from cultures around the world, 
told in the oral tradition. Like the fairy tales your grandparents told, 
these stories include darkness alongside wonder — they are meant for 
listeners of all ages who still believe the best place by the fire is 
kept for the storyteller.
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Storyteller voice sounds like a documentary | Lower stability, increase style. Add "..." pauses. The voice should *breathe*. |
| Dog voice sounds like Storyteller | Use distinctly different voice model. Dog is blunter, faster, modern-adjacent. |
| SD images look too modern/photorealistic | Add "oil painting, painterly, illustrated" to prompt. Increase guidance scale. |
| Stories too long for 45 seconds | Instruct BARD to extract single SCENE, not full tale. One moment, not one story. |
| Mood feels generic | Culture-specific prompts are essential. Japanese ≠ Norse ≠ African in tone. |
| Captions feel aggressive | Use warm serif font, not Impact. Smaller text. This isn't TikTok — it's a fireside. |
| Tales feel obscure/confusing | BARD must provide enough context in the hook. "There was a girl..." not "Vasilisa..." |

---

## Quick Reference

```bash
# Single tale
gorgon run story_fire --tale "the_juniper_tree" --culture european

# Weekly batch
gorgon run story_fire --batch 7

# Script review only
gorgon run story_fire --step bard --batch 7

# Specific culture
gorgon run story_fire --culture japanese --batch 3

# Full pipeline status
gorgon status story_fire

# Preview output
ffplay output/shorts/the_juniper_tree_short.mp4
```

---

*"When people told themselves their past with stories, explained their present with stories, foretold the future with stories, the best place by the fire was kept for the storyteller."*

*Pull up a chair. The fire's burning.*

*— Story Fire, powered by Gorgon*
