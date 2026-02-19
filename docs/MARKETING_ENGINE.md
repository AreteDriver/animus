# Animus Forge: Marketing Engine

> Autonomous content creation, batch approval, and cross-platform posting to drive traffic to projects and media channels.

---

## Overview

A Forge workflow that generates platform-optimized marketing posts for two content streams, queues them for weekly batch approval, and publishes approved posts on schedule. All three target platforms (Twitter/X, LinkedIn, Reddit) have posting APIs — no browser automation required.

---

## Content Streams

### Stream 1: Project Marketing
Drives traffic to GitHub repos, launches, and updates.

| Project | Content Angles |
|---------|---------------|
| BenchGoblins | Launch announcements, feature highlights, fantasy sports hot takes, NFL Draft season timing |
| Animus | Architecture deep-dives, technical blog posts, "building in public" updates |
| LinuxTools | Tool releases, OCR demos (LikX), Steam Deck compatibility (SteamProtonHelper) |
| EVE_Collection | Game dev progress, Argus updates, EVE community engagement |
| CLAUDE.md Generator | Usage tips, developer productivity angle |

### Stream 2: Media Engine
Drives traffic to YouTube channels.

| Channel | Content Angles |
|---------|---------------|
| Story Fire | Episode teasers, folklore facts, mythology threads |
| New Eden Whispers | EVE lore snippets, in-universe teasers, community engagement |
| Holmes Wisdom | Deduction puzzles, wisdom quotes, episode highlights |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  WEEKLY CYCLE                     │
│                                                   │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │ Research  │ → │  Draft   │ → │  Format  │     │
│  │  Agent   │   │  Agent   │   │  Agent   │     │
│  └──────────┘   └──────────┘   └──────────┘     │
│       ↓              ↓              ↓             │
│  Trending topics  Raw posts    Platform-specific  │
│  Project updates  per content  versions (X, LI,   │
│  Competitor intel stream      Reddit formatting)  │
│                                                   │
│              ┌──────────────┐                     │
│              │  Queue Agent │                     │
│              └──────┬───────┘                     │
│                     ↓                             │
│         ┌─────────────────────┐                   │
│         │   APPROVAL QUEUE    │                   │
│         │   (Weekly Batch)    │                   │
│         │                     │                   │
│         │  Core presents to   │                   │
│         │  you for review.    │                   │
│         │  Approve/edit/reject│                   │
│         │  per post.          │                   │
│         └──────────┬──────────┘                   │
│                    ↓                              │
│         ┌─────────────────────┐                   │
│         │   SCHEDULER AGENT   │                   │
│         │                     │                   │
│         │  Posts approved      │                   │
│         │  content at optimal  │                   │
│         │  times per platform  │                   │
│         └──────────┬──────────┘                   │
│                    ↓                              │
│     ┌──────────┬──────────┬──────────┬──────────┬──────────┐ │
│     │ X API    │ LinkedIn │ Reddit   │ YouTube  │ TikTok   │ │
│     │ Post     │ API Post │ API Post │ API Up   │ API Post │ │
│     └──────────┴──────────┴──────────┴──────────┴──────────┘ │
└─────────────────────────────────────────────────┘
```

---

## Agent Definitions

### Research Agent
**Input:** Project repos, Media Engine output, platform trending topics
**Output:** Content brief per post (topic, angle, target audience, relevant links)
**Sources:**
- GitHub activity (new commits, releases, stars)
- YouTube analytics (new episodes, view counts)
- Platform trends (X trending topics, Reddit hot posts in target subreddits, LinkedIn feed trends)
- Competitor activity (similar tools/projects)

### Draft Agent
**Input:** Content briefs from Research Agent
**Output:** Raw post text per content stream
**Rules:**
- Each post has a clear CTA (link to repo, video, or landing page)
- Tone matches the content stream (technical for Animus, casual for EVE, practical for LinuxTools)
- No generic AI slop ("excited to announce", "game-changer", "leveraging AI")
- Include hooks: questions, bold claims, or contrarian takes that invite engagement

### Format Agent
**Input:** Raw post text
**Output:** Platform-specific versions of each post
**Platform rules:**

| Platform | Constraints | Best Practices |
|----------|------------|----------------|
| **Twitter/X** | 280 chars (or long-form with Blue) | Thread format for deep content. Lead with hook. Hashtags sparingly (1-2 max). Images boost engagement 2-3x. |
| **LinkedIn** | 3,000 chars | Professional tone. First line is the hook (before "see more" fold). No hashtag spam. Tag relevant people/companies when genuine. |
| **Reddit** | Varies by subreddit | Match subreddit culture exactly. Self-promotion rules vary — some subs ban it, others allow with context. Never post the same content to multiple subs simultaneously. Include genuine value, not just a link. |
| **YouTube** | Title: 100 chars. Description: 5,000 chars. Tags: 500 chars total. | Front-load keywords in title. First 2 lines of description are visible before fold — put CTA and links there. Use chapters (timestamps). Custom thumbnail is mandatory for CTR. |
| **TikTok** | Caption: 2,200 chars. Video: 15s-3min. Vertical 9:16 only. | Hook in first 1-2 seconds or you're dead. Trending sounds boost reach. Hashtags matter more here than other platforms (3-5 relevant). Text overlays on video for silent viewers. No repurposed horizontal video — must be native vertical. |

### Queue Agent
**Input:** Formatted posts from Format Agent
**Output:** Structured weekly queue (JSON) organized by day/time/platform
**Logic:**
- Distributes posts across the week (not all on Monday)
- Avoids posting the same content stream back-to-back
- Assigns optimal posting windows per platform:
  - X: 8-10 AM and 5-7 PM (user's timezone)
  - LinkedIn: Tuesday-Thursday, 8-10 AM
  - Reddit: Morning hours, varies by subreddit activity

### Scheduler Agent
**Input:** Approved posts from queue
**Output:** Published posts via platform APIs
**Behavior:**
- Runs on schedule (cron or persistent daemon)
- Posts at the assigned time
- Logs success/failure per post
- Retries failed posts once, then flags for manual review
- Captures post URLs and engagement metrics for feedback loop

---

## Weekly Workflow

### Monday: Generation
Forge runs the Research → Draft → Format → Queue pipeline. Takes ~5-10 minutes of compute.

### Monday evening: Approval
Core presents the weekly queue to you. For each post you can:
- **Approve** — goes to scheduler as-is
- **Edit** — modify text, Core saves your version
- **Reject** — removed from queue, optionally replaced
- **Reschedule** — move to a different day/time

Approval interface: Core presents posts in a structured format (could be CLI, could be a simple web UI, could be a notification). You review the batch in one session.

### Tuesday–Sunday: Posting
Scheduler Agent publishes approved content at assigned times. You don't touch anything unless a post fails.

### Sunday: Analytics
Research Agent pulls engagement metrics from the past week:
- Impressions, clicks, replies, shares per post
- Which content streams performed best
- Which platforms drove the most traffic
- Feeds into next Monday's Research Agent for optimization

---

## Platform API Requirements

### Twitter/X
- **API:** Twitter API v2 (Free tier: 1,500 tweets/month write. Basic $100/mo: 3,000 tweets/month + read access)
- **Auth:** OAuth 2.0 with PKCE
- **Endpoints:** `POST /2/tweets`, `GET /2/tweets/:id` (for metrics)
- **Media:** Upload API for images/video
- **Free tier is sufficient** for your volume (~15-20 posts/week)

### LinkedIn
- **API:** LinkedIn Marketing API or Community Management API
- **Auth:** OAuth 2.0 (3-legged)
- **Endpoints:** `POST /ugcPosts` for sharing, `GET /socialActions` for metrics
- **Requirement:** Must create a LinkedIn app and get review for posting permissions
- **Note:** LinkedIn API access requires app review — can take 1-2 weeks

### Reddit
- **API:** Reddit API (free for non-commercial, rate limited)
- **Auth:** OAuth 2.0
- **Endpoints:** `POST /api/submit` for new posts, `POST /api/comment` for comments
- **Rate limit:** 60 requests/minute
- **Critical:** Reddit detects and punishes automated/spammy posting. Posts must be genuine and subreddit-appropriate. The Draft Agent needs subreddit-specific context.

### YouTube
- **API:** YouTube Data API v3 (free quota: 10,000 units/day — uploads cost 1,600 units each, so ~6 uploads/day max)
- **Auth:** OAuth 2.0
- **Endpoints:** `POST /videos` (upload), `PUT /videos` (update metadata), `POST /thumbnails/set`, `GET /videos` (analytics)
- **Media:** Resumable upload protocol for videos, direct upload for thumbnails
- **Two use cases:**
  - **Media Engine:** Full video uploads (Story Fire, New Eden Whispers, Holmes Wisdom). Forge's Media Engine pipeline already produces the video — this agent handles upload, title, description, tags, thumbnail, playlist assignment, and scheduling.
  - **Project marketing:** YouTube Shorts (< 60s) for quick demos, tool walkthroughs, and teasers that link to full repos or longer content.
- **Metadata matters:** Title, description, tags, thumbnail, and category directly impact discovery. The Format Agent needs YouTube SEO awareness — front-load keywords in titles, write descriptions with timestamps and links, use relevant tags without stuffing.
- **Scheduling:** YouTube API supports setting `publishAt` for scheduled publishing. Videos upload as private, go public at the scheduled time.

### TikTok
- **API:** TikTok Content Posting API (requires developer app approval)
- **Auth:** OAuth 2.0 (Login Kit)
- **Endpoints:** `POST /v2/post/publish/video/init/` (initiate upload), `POST /v2/post/publish/status/fetch/` (check status)
- **Restrictions:**
  - Must be approved for Content Posting API access (application required)
  - Videos must be uploaded (no URL-based posting)
  - Maximum 10MB direct upload or chunked for larger files
  - Captions limited to 2,200 characters
  - No scheduling via API — posts go live immediately or as drafts
  - Rate limit: 3 posts per day per user (TikTok enforces this)
- **Content strategy:** Short-form only (15s-3min). Best for: quick tool demos, "did you know" EVE lore clips, fantasy sports hot takes, Linux tips. These should be extracted/repurposed from longer YouTube content, not created from scratch.
- **Important:** TikTok's algorithm favors native-feeling content. Reposting YouTube videos with letterboxing or visible watermarks gets suppressed. The Format Agent must output vertical (9:16) TikTok-native versions.
- **Approval timeline:** TikTok developer app review can take 2-4 weeks. Start early.

**Target subreddits by content stream:**

| Content | Subreddits |
|---------|-----------|
| BenchGoblins | r/fantasyfootball, r/DynastyFF, r/FantasyFootballers |
| Animus/AI | r/LocalLLaMA, r/MachineLearning, r/artificial |
| LinuxTools | r/linux, r/unixporn, r/linux_gaming, r/SteamDeck |
| EVE | r/Eve, r/eveonline, r/evetech |
| General dev | r/Python, r/rust, r/selfhosted |

---

---

## Video Content Pipeline

YouTube and TikTok introduce a video production dependency that text-only platforms don't have. Here's how it connects to your existing Media Engine:

### Media Engine → YouTube (Full Videos)
The Media Engine already produces complete videos for Story Fire, New Eden Whispers, and Holmes Wisdom. The Marketing Engine's YouTube agent only handles the **last mile**: upload, metadata, thumbnail, playlist, and scheduling. It does NOT produce video.

```
Media Engine (Forge pipeline) → Rendered video file
    ↓
Marketing Engine YouTube Agent → Upload + metadata + schedule
```

### YouTube → TikTok (Repurposed Clips)
TikTok content is **extracted from YouTube videos**, not created from scratch. This saves production cost and ensures content consistency.

```
YouTube full video (16:9, 5-15 min)
    ↓
Clip Extraction Agent:
  - Identifies 3-5 high-impact moments (hooks, reveals, key insights)
  - Extracts 15-60 second clips
  - Reframes to 9:16 vertical (center crop or dynamic reframe)
  - Adds text overlays for silent viewing
  - Adds trending audio if appropriate
    ↓
TikTok-native vertical clips ready for posting
```

**Tools for reframing:**
- **FFmpeg** — CLI-based crop/resize. `ffmpeg -i input.mp4 -vf "crop=ih*9/16:ih" output.mp4`
- **Auto-reframe** — AI-based subject tracking (OpenCV or cloud APIs) for dynamic cropping that follows the speaker/action

### YouTube Shorts (Project Marketing)
Quick demos and walkthroughs for project marketing. These are **net new** short-form content, not repurposed from Media Engine:

- LikX OCR demo (15s screen recording)
- BenchGoblins walkthrough (30s)
- "One thing about Animus's architecture" explainers (60s)
- SteamProtonHelper fixing a real issue (30s)

Shorts use the same vertical 9:16 format as TikTok. **Produce once, post to both YouTube Shorts and TikTok.**

### Internet Archive → Shorts (Public Domain Film Clips)

A fourth content source: extracting compelling clips from public domain films on the Internet Archive and repurposing them as Shorts/TikToks with commentary, context, or channel-relevant narration.

**Why this works:**
- Massive library of free, legal content with zero production cost
- Old footage has a distinctive aesthetic that performs well on short-form platforms (nostalgia, curiosity)
- Directly feeds Story Fire (folklore/mythology documentaries from the 40s-60s) and Holmes Wisdom (Basil Rathbone Sherlock Holmes films from 1939-1946 are public domain)
- New Eden Whispers could use retro sci-fi footage as visual backdrop for lore narration

**Licensing rules (non-negotiable):**
- **ONLY use** films marked Public Domain or CC0 (Creative Commons Zero)
- **Filter URL:** `https://archive.org/search.php?query=licenseurl:*publicdomain*&mediatype=movies`
- **NEVER use** CC BY-NC (non-commercial) content if channels are monetized with ads — ad-supported platforms count as commercial use
- **NEVER use** restored/colorized/re-scored versions — the restoration may have its own copyright even if the original film is PD
- **ALWAYS verify** individual film licensing before use. The Archive's metadata isn't always accurate. When in doubt, check the original copyright registration.
- **Prelinger Archives** collection is explicitly offered for free reuse — prioritize this collection

**Key collections to mine:**

| Collection | URL | Channel Fit |
|-----------|-----|-------------|
| Prelinger Archives | archive.org/details/prelinger | Story Fire (cultural/educational films), general B-roll |
| Feature Films | archive.org/details/feature_films | Holmes Wisdom (Rathbone-era Sherlock Holmes) |
| Sci-Fi Films | archive.org/details/SciFi_Horror | New Eden Whispers (retro space footage as visual layer) |
| Animation | archive.org/details/animation | Story Fire (classic folklore/fairy tale animations) |
| Newsreels | archive.org/details/newsandpublicaffairs | General B-roll, historical context clips |

**Pipeline:**

```
Archive Source Agent:
  - Searches Internet Archive API for PD-licensed films in target collections
  - Downloads candidate clips
  - Verifies license metadata
      ↓
Clip Extraction Agent:
  - Identifies visually compelling moments (15-60s)
  - Extracts clips via FFmpeg
      ↓
Narration Agent:
  - Generates voiceover script contextualizing the clip
  - Ties to channel theme (folklore context for Story Fire, 
    deduction lesson for Holmes, lore parallel for New Eden)
  - TTS generation via Media Engine voice pipeline
      ↓
Assembly Agent:
  - Composites: archive footage + narration + text overlays + branding
  - Outputs both 16:9 (YouTube) and 9:16 (Shorts/TikTok) versions
      ↓
Queue → Approval → Publish (same as all other content)
```

**Internet Archive API access:**
- Archive.org has a free API for searching and downloading: `https://archive.org/advancedsearch.php`
- No authentication required for public content
- Bulk downloading supported via `ia` CLI tool (`pip install internetarchive`)
- Rate limiting: be respectful — Archive.org is a nonprofit. Cache aggressively, don't hammer.

**Volume estimate:** 3-5 archive-sourced Shorts per week across channels. Low cost (no original video production), high novelty factor.

---

## Forge YAML Config

```yaml
# configs/marketing_engine/weekly_content.yaml
name: weekly_marketing_content
description: Generate, approve, and publish weekly marketing posts
schedule: "0 6 * * 1"  # Monday 6 AM

agents:
  - name: researcher
    archetype: researcher
    budget_tokens: 5000
    inputs:
      - github_activity: AreteDriver/*
      - youtube_channels: [story_fire, new_eden_whispers, holmes_wisdom]
      - platform_trends: [twitter, linkedin, reddit]
    outputs:
      - content_briefs

  - name: drafter
    archetype: writer
    budget_tokens: 8000
    inputs:
      - content_briefs: researcher.content_briefs
      - brand_voice: configs/marketing_engine/brand_voice.yaml
    outputs:
      - raw_posts

  - name: formatter
    archetype: writer
    budget_tokens: 4000
    inputs:
      - raw_posts: drafter.raw_posts
      - platform_rules: configs/marketing_engine/platform_rules.yaml
    outputs:
      - formatted_posts

  - name: queue_builder
    archetype: planner
    budget_tokens: 2000
    inputs:
      - formatted_posts: formatter.formatted_posts
      - schedule_rules: configs/marketing_engine/schedule_rules.yaml
    outputs:
      - weekly_queue

gates:
  - name: human_approval
    type: human_in_loop
    after: queue_builder
    mode: batch
    notification: core  # Core presents queue for approval

  - name: publish
    type: automated
    after: human_approval
    agent: scheduler
    archetype: publisher
    budget_tokens: 1000
    inputs:
      - approved_queue: gates.human_approval.approved
      - api_credentials: secrets/platform_credentials.yaml
    outputs:
      - post_results
      - engagement_metrics
```

---

## Brand Voice Config

```yaml
# configs/marketing_engine/brand_voice.yaml
global:
  avoid:
    - "excited to announce"
    - "game-changer"
    - "leveraging AI"
    - "deep dive" (unless genuinely technical)
    - "thrilled"
    - emoji spam
  principles:
    - Direct and specific. Say what the thing does, not how it makes you feel.
    - Technical credibility over hype. Show the architecture, not the buzzwords.
    - Genuine personality. EVE references, Toyota/TPS analogies, and real opinions are welcome.

streams:
  project_marketing:
    tone: Builder sharing work. Confident but not arrogant. Show, don't tell.
    cta_style: Direct link with one-line context.
    example: "LikX is the only Linux screenshot tool with built-in OCR. Capture → extract text → paste. No plugins, no pipelines. [link]"

  benchgoblins:
    tone: Sharp, opinionated fantasy sports voice. Not corporate.
    cta_style: Hook with a take, link to the tool.
    example: "Your league-mates are still using gut feelings. BenchGoblins runs 5 proprietary indexes on every start/sit decision. Free tier, no credit card. [link]"

  eve_content:
    tone: In-universe where possible. Community-native. Not promotional — contributory.
    cta_style: Value-first, link secondary.
    example: "The gap between DUST 514 and the Deathless is the most underexplored era in New Eden. We're telling those stories. Latest episode explores the Molden Heath dead zones. [link]"

  linux_tools:
    tone: Pragmatic. "I built this because nothing else worked."
    cta_style: Problem → solution → link.
    example: "Flameshot can't do OCR. Ksnip can't do OCR. Shutter can't do OCR. LikX can. [link]"

  technical_ai:
    tone: Architectural thinking. TPS/lean analogies. Systems perspective.
    cta_style: Insight first, project second.
    example: "Most multi-agent systems use a supervisor pattern — one agent watching all others. That's a bottleneck and a token sink. Animus Swarm uses stigmergy instead — agents read shared state and self-coordinate. Same principle as flocking birds. [link to whitepaper]"
```

---

## Data Model

```python
from pydantic import BaseModel
from enum import Enum
from datetime import datetime

class Platform(str, Enum):
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    REDDIT = "reddit"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"

class ContentStream(str, Enum):
    PROJECT_MARKETING = "project_marketing"
    BENCHGOBLINS = "benchgoblins"
    EVE_CONTENT = "eve_content"
    LINUX_TOOLS = "linux_tools"
    TECHNICAL_AI = "technical_ai"

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"

class PostDraft(BaseModel):
    id: str
    stream: ContentStream
    platform: Platform
    content: str
    media_urls: list[str] = []
    cta_url: str
    hashtags: list[str] = []
    subreddit: str | None = None  # Reddit only
    scheduled_time: datetime
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    edited_content: str | None = None  # If user edits during approval
    post_url: str | None = None  # After publishing
    metrics: dict | None = None  # After analytics collection

class WeeklyQueue(BaseModel):
    week_of: datetime
    posts: list[PostDraft]
    total_by_platform: dict[Platform, int]
    total_by_stream: dict[ContentStream, int]
```

---

## Implementation Priority

### Phase 1: Manual Pipeline (Week 1)
- Build the Research + Draft + Format agents
- Output: Markdown file with the weekly queue for manual review
- Posting: You copy-paste approved posts manually
- **Value:** Immediately saves you content creation time

### Phase 2: Approval UI (Week 2)
- Core presents queue in a structured format (CLI table or simple web page)
- Approve/edit/reject per post
- Output: Approved queue as JSON

### Phase 3: Text Platform API Integration (Week 3-4)
- Twitter API posting
- LinkedIn API posting (may need app review wait)
- Reddit API posting (careful with rate limits and subreddit rules)
- Scheduler daemon (cron or persistent process)

### Phase 4: YouTube Integration (Week 4-5)
- YouTube Data API v3 upload flow (resumable uploads for large videos)
- Metadata agent (title, description, tags, thumbnail, playlist, category)
- Scheduling via `publishAt` parameter
- Connect to Media Engine output directory for automatic pickup

### Phase 5: TikTok Integration (Week 5-6)
- TikTok Content Posting API (requires app approval — start application in Week 1)
- Clip Extraction Agent (FFmpeg-based, identifies and crops key moments)
- Vertical reframing pipeline (16:9 → 9:16)
- Text overlay generation for silent viewers
- YouTube Shorts cross-posting (same vertical content, both platforms)

### Phase 6: Analytics Feedback Loop (Week 7)
- Pull engagement metrics per post across all 5 platforms
- Feed into Research Agent for next week's optimization
- Weekly performance summary
- Platform comparison (which content performs where)

---

### API Application Timeline (Start Immediately)

| Platform | Action | Expected Wait |
|----------|--------|---------------|
| Twitter/X | Create developer app | Instant (free tier) |
| LinkedIn | Apply for Marketing API access | 1-2 weeks |
| Reddit | Create script app | Instant |
| YouTube | Enable YouTube Data API in Google Cloud Console | Instant (quota review if needed) |
| TikTok | Apply for Content Posting API | 2-4 weeks |

**Start LinkedIn and TikTok applications on Day 1** — they're the bottleneck. Everything else is instant.

---

## Estimated Volume

| Platform | Posts/Week | Monthly Total | Notes |
|----------|-----------|---------------|-------|
| Twitter/X | 7-10 | 30-40 | Text + image posts |
| LinkedIn | 3-5 | 12-20 | Professional content |
| Reddit | 3-5 | 12-20 | Subreddit-specific |
| YouTube (full) | 10-12 | 40-48 | Media Engine videos |
| YouTube (Shorts) | 5-8 | 20-32 | Project demos + archive clips |
| TikTok | 5-7 | 20-28 | Vertical clips (YouTube repurposed + archive clips) |
| **Total** | **33-47** | **134-188** |

Content sources breakdown:
- **Media Engine videos:** ~24/week (3 channels × 8 languages) — already being produced
- **Project marketing (text):** ~13-20/week across X, LinkedIn, Reddit
- **Project marketing (video):** ~3-5 Shorts/week (demos, explainers)
- **Archive-sourced clips:** ~3-5/week (public domain film excerpts with narration)
- **YouTube → TikTok repurposed:** ~5-7/week (vertical crops of existing content)

YouTube volume is primarily driven by Media Engine. Archive clips and project Shorts are additive.
TikTok is capped at 3 posts/day by API limits (21/week max). 5-7/week gives headroom.

---

## Secrets Management

Platform API credentials stored in `secrets/platform_credentials.yaml` (gitignored, encrypted at rest). Forge reads credentials at publish time only. Never logged, never included in checkpoint state.

```yaml
# secrets/platform_credentials.yaml
twitter:
  api_key: "..."
  api_secret: "..."
  access_token: "..."
  access_token_secret: "..."

linkedin:
  client_id: "..."
  client_secret: "..."
  access_token: "..."

reddit:
  client_id: "..."
  client_secret: "..."
  username: "..."
  password: "..."

youtube:
  client_id: "..."
  client_secret: "..."
  refresh_token: "..."

tiktok:
  client_key: "..."
  client_secret: "..."
  access_token: "..."
```

---

---

## ROI Tracking & Cost Measurement

Shorts and social posts are a **distribution channel, not a revenue stream.** Direct ad revenue from Shorts will not cover production costs. The value is in what they drive traffic to. Track accordingly.

### Cost Tracking

Every post and Short should log its production cost:

```python
class PostCost(BaseModel):
    post_id: str
    costs: list[CostItem]
    total_cost: float

class CostItem(BaseModel):
    category: str        # "ai_video", "ai_text", "api_call", "tts", "compute"
    provider: str        # "kling", "sora", "claude", "elevenlabs"
    amount: float
    unit: str            # "seconds", "tokens", "credits"
    quantity: float
```

Weekly cost summary by category:

| Category | Provider | Weekly Budget | Track |
|----------|----------|--------------|-------|
| AI video generation | Kling / Sora | $5-30 | Per-clip cost, resolution, duration |
| LLM inference | Claude API / Ollama | $5-15 | Tokens per draft/format cycle |
| TTS | Media Engine voice pipeline | $0-10 | Per-minute generated |
| Platform API calls | Twitter/LinkedIn/Reddit/YouTube/TikTok | $0 (free tiers) | Rate limit headroom |
| **Total weekly** | | **$10-55** | |
| **Total monthly** | | **$40-220** | |

### Revenue Tracking — Direct

Track direct monetization even though it won't be the primary ROI:

| Source | Metric | How to Track |
|--------|--------|-------------|
| YouTube Shorts RPM | $/1000 views | YouTube Studio analytics |
| YouTube long-form RPM | $/1000 views | YouTube Studio analytics |
| TikTok Creator Program | $/qualified views | TikTok analytics (requires 10K followers) |
| YouTube AdSense (channels) | Monthly payout | AdSense dashboard |

### Revenue Tracking — Funnel (This Is Where the ROI Actually Lives)

| Funnel Event | Value | How to Track |
|-------------|-------|-------------|
| BenchGoblins Pro signup | $6-9/mo or $39/season | UTM links in post CTAs → RevenueCat conversion tracking |
| BenchGoblins League signup | $59-99/season | UTM links → RevenueCat |
| YouTube channel subscriber | ~$0.50-2.00 LTV (estimated) | Track subscriber source in YouTube Studio |
| GitHub star on a repo | Portfolio credibility (unquantifiable but real) | Track referral traffic in GitHub Insights |
| GitHub repo visitor | Portfolio signal | GitHub traffic analytics + UTM referrers |
| Job interview mention | Highest value, hardest to track | Manual log — "did the interviewer reference my projects?" |
| Email list signup | Future monetization potential | If/when you add a mailing list |

### Attribution — How to Know What's Working

Every link in every post gets a UTM tag:

```
https://benchgoblins.com/?utm_source=twitter&utm_medium=short&utm_campaign=nfl_draft_2026&utm_content=start_sit_demo

https://github.com/AreteDriver/Animus?utm_source=linkedin&utm_medium=post&utm_campaign=architecture_series&utm_content=swarm_explainer
```

**UTM structure:**
- `utm_source` — platform (twitter, linkedin, reddit, youtube, tiktok)
- `utm_medium` — content type (post, short, thread, comment)
- `utm_campaign` — campaign name (nfl_draft_2026, linux_tools_launch, etc.)
- `utm_content` — specific post identifier

The Analytics Agent (Phase 6) pulls UTM data weekly and correlates with:
- Which platform drove the most conversions (not just views)
- Which content stream has the highest conversion rate
- Which Shorts drove subscribers vs. one-time views
- Cost per acquisition by channel

### Weekly ROI Dashboard

The Analytics Agent generates a weekly summary:

```
WEEK OF 2026-03-02

SPEND
├── AI Video (Kling):     $12.40  (28 clips)
├── LLM (Claude API):     $8.20   (drafts + formatting)
├── TTS:                   $3.10   (narration)
└── Total:                 $23.70

DIRECT REVENUE
├── YouTube Shorts:        $2.40   (48K views)
├── YouTube Long-form:     $18.60  (12K views, higher RPM)
├── TikTok:                $0.80   (22K views)
└── Total:                 $21.80

FUNNEL CONVERSIONS
├── BenchGoblins Pro:      3 signups ($18-27/mo recurring)
├── YouTube subs gained:   142 (across 3 channels)
├── GitHub stars gained:   8 (Animus: 3, LinuxTools: 5)
├── GitHub profile views:  340
└── Estimated funnel value: $45-80/mo (recurring + LTV)

NET ROI: -$1.90 direct, +$21-56/mo including funnel value
TOP PERFORMER: "LikX OCR demo" Short → 12K views, 5 GitHub stars, 2 BG signups
WORST PERFORMER: "Holmes deduction tip #4" → 800 views, 0 conversions → KILL or revise format
```

### Decision Rules

Build these into the Research Agent's weekly optimization:

**Scale up** if:
- A content stream's funnel conversion rate exceeds 0.1% (views → signup/star/subscribe)
- A specific Short format consistently drives >10K views
- Cost per BenchGoblins acquisition is under $5

**Scale down** if:
- A content stream's cost per conversion exceeds $20
- A platform consistently delivers views but zero funnel conversions (vanity metrics)
- AI-generated video Shorts underperform archive-sourced Shorts at higher cost

**Kill** if:
- A content stream has zero funnel conversions after 4 weeks
- A platform's API costs increase without proportional conversion increase
- Audience engagement signals (comments, shares, saves) are near zero despite views

### Monthly Budget Caps

Hard limits to prevent runaway spend while testing:

| Month | AI Video Budget | Total Marketing Budget | Review Gate |
|-------|----------------|----------------------|-------------|
| Month 1 | $20 | $50 | Can I see any funnel signal at all? |
| Month 2 | $30 | $75 | Which streams convert? Double down. |
| Month 3 | $50 | $120 | Cut non-performers. Scale winners. |
| Month 4+ | Based on ROI | Based on ROI | Spend should be self-funding from BG revenue |

**The goal:** By Month 4, BenchGoblins subscription revenue from marketing-driven signups should exceed total marketing spend. If it doesn't, the content strategy needs revision, not more budget.

---

## What This Is NOT

- **Not a social media management SaaS.** This is a personal Forge workflow, not a product to sell.
- **Not spam.** Every post must provide genuine value. Reddit especially will destroy you if posts are hollow promotion.
- **Not fully autonomous.** You approve every batch. The AI drafts and publishes — you maintain editorial control.
- **Not replacing engagement.** Automated posting without replying to comments is obvious and off-putting. You still engage manually with replies and conversations.
