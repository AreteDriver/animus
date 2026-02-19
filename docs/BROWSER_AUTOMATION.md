# Animus Browser Automation Layer

> Giving Animus eyes, hands, and feet on the web.

---

## Why Animus Needs a Browser

Three use cases require browser automation today:

1. **Marketing Engine fallback** — If TikTok's Content Posting API approval is denied or a platform removes API access, Forge needs to post via browser as a fallback.
2. **Form filling and account management** — Signing up for services, filling applications, managing platform settings that have no API.
3. **Web scraping for Research Agent** — Some data sources (forums, paywalled previews, dynamic JS-rendered pages) can't be fetched with simple HTTP requests.

Future use cases:
4. **3D printer web interfaces** — Some printers (Bambu Lab) use web dashboards instead of REST APIs.
5. **Voice-driven web interaction** — "Animus, order more PLA filament from Amazon" (far future, high complexity).

---

## Architecture

The browser is an **Animus Core interface adapter** — same level as CLI, voice, and desktop interfaces. Forge workflows call it through Core when an agent needs browser interaction.

```
Forge Agent: "Post this video to TikTok"
    ↓
Core: Checks if TikTok API is available
    ↓ (API unavailable or action requires browser)
Core → Browser Adapter: "Navigate to tiktok.com/upload, upload video, set caption"
    ↓
Browser Adapter: Executes via Playwright
    ↓ (hits CAPTCHA)
Browser Adapter → CAPTCHA Solver: Solve and continue
    ↓
Browser Adapter → Core: "Posted successfully, URL: ..."
    ↓
Core → Forge Agent: Result
```

### Component Stack

```
┌─────────────────────────────────────────┐
│           ANIMUS CORE                    │
│  (routes requests to appropriate adapter)│
├─────────────────────────────────────────┤
│        BROWSER ADAPTER                   │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │  Playwright  │  │  Action Scripts  │  │
│  │  (headless   │  │  (per-site       │  │
│  │   browser)   │  │   recipes)       │  │
│  └──────┬──────┘  └────────┬─────────┘  │
│         │                  │             │
│  ┌──────┴──────────────────┴─────────┐  │
│  │        Session Manager            │  │
│  │  (cookies, auth state, profiles)  │  │
│  └──────────────┬────────────────────┘  │
│                 │                        │
│  ┌──────────────┴────────────────────┐  │
│  │        CAPTCHA Handler            │  │
│  │  (2Captcha / CapSolver API)       │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

---

## Core Components

### 1. Playwright Engine

**Why Playwright over Selenium:**
- Async-native (fits Animus's async architecture)
- Built-in auto-wait (no manual sleep/wait hacks)
- Multi-browser support (Chromium, Firefox, WebKit)
- Headless and headed modes
- Built-in screenshot and video recording (useful for debugging and audit trails)
- Network interception (can mock/modify requests)
- Python-first API

```bash
pip install playwright
playwright install chromium
```

```python
# core/browser/engine.py
from playwright.async_api import async_playwright

class BrowserEngine:
    def __init__(self, headless: bool = True, profile_dir: str = None):
        self.headless = headless
        self.profile_dir = profile_dir  # Persistent cookies/auth

    async def start(self):
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch_persistent_context(
            user_data_dir=self.profile_dir or "~/.animus/browser_profile",
            headless=self.headless,
            viewport={"width": 1920, "height": 1080},
        )
        self.page = self.browser.pages[0] if self.browser.pages else await self.browser.new_page()

    async def navigate(self, url: str):
        await self.page.goto(url, wait_until="networkidle")

    async def screenshot(self, path: str = None) -> bytes:
        return await self.page.screenshot(path=path, full_page=True)

    async def close(self):
        await self.browser.close()
        await self.pw.stop()
```

### 2. Action Scripts (Per-Site Recipes)

Each site/platform gets a dedicated action script. These are **brittle by nature** (sites change their HTML), so they're isolated and versioned independently.

```
core/browser/actions/
├── tiktok.py          ← Upload video, set caption, post
├── instagram.py       ← Upload photo/reel (future)
├── generic_form.py    ← Fill arbitrary forms from structured data
├── scraper.py         ← Extract content from JS-rendered pages
└── printer_ui.py      ← Interact with 3D printer web dashboards
```

```python
# core/browser/actions/tiktok.py
from core.browser.engine import BrowserEngine
from core.browser.captcha import CaptchaSolver

class TikTokAction:
    def __init__(self, engine: BrowserEngine, captcha: CaptchaSolver):
        self.engine = engine
        self.captcha = captcha

    async def upload_video(self, video_path: str, caption: str) -> str:
        page = self.engine.page

        # Navigate to upload page
        await page.goto("https://www.tiktok.com/upload")

        # Wait for upload button
        upload_input = await page.wait_for_selector('input[type="file"]')
        await upload_input.set_input_files(video_path)

        # Wait for upload to complete
        await page.wait_for_selector('[class*="upload-success"]', timeout=60000)

        # Set caption
        caption_box = await page.wait_for_selector('[class*="caption-editor"]')
        await caption_box.fill(caption)

        # Check for CAPTCHA
        if await self._detect_captcha(page):
            await self.captcha.solve(page)

        # Post
        post_button = await page.wait_for_selector('button:has-text("Post")')
        await post_button.click()

        # Wait for confirmation and extract URL
        await page.wait_for_url("**/video/**", timeout=30000)
        return page.url

    async def _detect_captcha(self, page) -> bool:
        try:
            await page.wait_for_selector('[class*="captcha"]', timeout=3000)
            return True
        except:
            return False
```

### 3. Session Manager

Persistent browser profiles so you don't re-authenticate every time.

```python
# core/browser/session.py
from pathlib import Path

class SessionManager:
    BASE_DIR = Path.home() / ".animus" / "browser_sessions"

    def __init__(self):
        self.BASE_DIR.mkdir(parents=True, exist_ok=True)

    def get_profile_dir(self, platform: str) -> str:
        """Each platform gets its own persistent browser profile."""
        profile_dir = self.BASE_DIR / platform
        profile_dir.mkdir(exist_ok=True)
        return str(profile_dir)

    def list_sessions(self) -> list[str]:
        return [d.name for d in self.BASE_DIR.iterdir() if d.is_dir()]

    def clear_session(self, platform: str):
        import shutil
        profile_dir = self.BASE_DIR / platform
        if profile_dir.exists():
            shutil.rmtree(profile_dir)
```

### 4. CAPTCHA Handler

CAPTCHAs are the main friction point. Options:

| Service | Cost | Speed | Types Supported |
|---------|------|-------|----------------|
| **2Captcha** | $1-3 per 1000 solves | 10-30s | reCAPTCHA v2/v3, hCaptcha, image, text, FunCaptcha |
| **CapSolver** | $0.80-2.50 per 1000 | 5-20s | Same + Cloudflare Turnstile |
| **Anti-Captcha** | $1-3 per 1000 | 10-30s | Same as 2Captcha |

**Recommendation: CapSolver** — cheapest, fastest, supports Cloudflare Turnstile (increasingly common).

At your expected volume (<50 CAPTCHA solves/week), cost is negligible: ~$0.10-0.25/week.

```python
# core/browser/captcha.py
import httpx

class CaptchaSolver:
    def __init__(self, provider: str = "capsolver", api_key: str = None):
        self.provider = provider
        self.api_key = api_key
        self.base_url = {
            "capsolver": "https://api.capsolver.com",
            "2captcha": "https://api.2captcha.com",
        }[provider]

    async def solve_recaptcha(self, site_key: str, page_url: str) -> str:
        """Solve reCAPTCHA v2 and return the token."""
        async with httpx.AsyncClient() as client:
            # Create task
            resp = await client.post(f"{self.base_url}/createTask", json={
                "clientKey": self.api_key,
                "task": {
                    "type": "ReCaptchaV2TaskProxyLess",
                    "websiteURL": page_url,
                    "websiteKey": site_key,
                }
            })
            task_id = resp.json()["taskId"]

            # Poll for result
            for _ in range(60):
                await asyncio.sleep(2)
                result = await client.post(f"{self.base_url}/getTaskResult", json={
                    "clientKey": self.api_key,
                    "taskId": task_id,
                })
                data = result.json()
                if data["status"] == "ready":
                    return data["solution"]["gRecaptchaResponse"]

            raise TimeoutError("CAPTCHA solve timed out")

    async def solve(self, page):
        """Auto-detect CAPTCHA type on page and solve it."""
        # Detect reCAPTCHA
        recaptcha = await page.query_selector('[data-sitekey]')
        if recaptcha:
            site_key = await recaptcha.get_attribute('data-sitekey')
            token = await self.solve_recaptcha(site_key, page.url)
            await page.evaluate(f'document.getElementById("g-recaptcha-response").innerHTML="{token}"')
            return True

        # Detect hCaptcha
        hcaptcha = await page.query_selector('[data-hcaptcha-widget-id]')
        if hcaptcha:
            # Similar flow for hCaptcha
            pass

        return False
```

---

## Integration with Forge

Forge workflows request browser actions through a standardized interface:

```yaml
# Example: Marketing Engine TikTok fallback
agents:
  - name: tiktok_publisher
    archetype: publisher
    tool: browser  # Signals that this agent needs browser access
    action: tiktok.upload_video
    inputs:
      video_path: forge.outputs.assembled_video
      caption: forge.outputs.formatted_caption
    fallback:
      on_captcha_fail: retry_once_then_flag
      on_login_expired: notify_user  # Core alerts you to re-authenticate
```

```python
# forge/tools/browser_tool.py
from core.browser.engine import BrowserEngine
from core.browser.session import SessionManager
from core.browser.captcha import CaptchaSolver
from core.browser.actions import tiktok, generic_form, scraper

class BrowserTool:
    """Forge-compatible tool that gives agents browser access."""

    ACTIONS = {
        "tiktok.upload_video": tiktok.TikTokAction.upload_video,
        "tiktok.post_text": tiktok.TikTokAction.post_text,
        "form.fill": generic_form.FormAction.fill,
        "scrape.extract": scraper.ScraperAction.extract,
    }

    async def execute(self, action: str, **kwargs):
        session = SessionManager()
        platform = action.split(".")[0]
        engine = BrowserEngine(
            headless=True,
            profile_dir=session.get_profile_dir(platform)
        )
        captcha = CaptchaSolver(
            provider="capsolver",
            api_key=os.environ["CAPSOLVER_API_KEY"]
        )

        try:
            await engine.start()
            action_fn = self.ACTIONS[action]
            result = await action_fn(engine=engine, captcha=captcha, **kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            screenshot = await engine.screenshot()  # Capture failure state
            return {"success": False, "error": str(e), "screenshot": screenshot}
        finally:
            await engine.close()
```

---

## Authentication Flow

Browser automation requires active sessions. The first time you use a platform:

1. **Manual login (once):** Animus launches a headed (visible) browser. You log in manually. Cookies are saved to the persistent profile.
2. **Subsequent runs:** Headless browser loads the saved profile. If the session is still valid, no login needed.
3. **Session expired:** Animus detects login page redirect → pauses workflow → notifies you via Core ("TikTok session expired, please re-authenticate") → launches headed browser for manual login → saves new session → resumes workflow.

**Never store plaintext passwords.** Authentication is always via browser cookie persistence or OAuth where available.

---

## Anti-Detection Measures

Platforms actively detect and block automation. Mitigations:

| Technique | Implementation |
|-----------|---------------|
| **Real browser fingerprint** | Playwright with persistent profile (not fresh profile each time) |
| **Human-like timing** | Random delays between actions (200-800ms). No instant fills. |
| **Realistic mouse movement** | `page.mouse.move()` with intermediate points |
| **Viewport variation** | Slight randomization of window size |
| **Stealth plugin** | `playwright-stealth` package patches common detection vectors |
| **Rate limiting** | Never exceed human-plausible posting frequency |
| **IP rotation** | Not needed at your volume. Only if you get blocked. |

```bash
pip install playwright-stealth
```

```python
from playwright_stealth import stealth_async

async def start(self):
    # ... launch browser ...
    await stealth_async(self.page)  # Apply anti-detection patches
```

---

## Ethical Boundaries

- **NEVER automate in-game actions for EVE Online** — violates CCP EULA and your own ESI compliance rules
- **NEVER automate interactions that impersonate a human in deceptive ways** — posting content is fine, pretending to be a human in conversations is not
- **NEVER store or transmit credentials** — cookies only, never passwords
- **NEVER use browser automation to bypass paywalls or access restrictions** — scraping is for public content only
- **Respect robots.txt** for scraping targets
- **Rate limit everything** — you're automating convenience, not scale abuse

---

## Dependencies

```
playwright>=1.40.0
playwright-stealth>=1.0.0
httpx>=0.25.0
```

Total additional dependency footprint is small. Playwright downloads Chromium (~150MB) on first install.

---

## Implementation Priority

| Phase | What | When |
|-------|------|------|
| Phase 1 | BrowserEngine + SessionManager + basic navigation | When Marketing Engine needs browser fallback |
| Phase 2 | CAPTCHA integration (CapSolver) | When first CAPTCHA is encountered |
| Phase 3 | TikTok action script | If TikTok API approval is denied |
| Phase 4 | Generic form filler + scraper | When Forge workflows need it |
| Phase 5 | 3D printer web UI integration | When you buy a printer |

**Don't build this until you need it.** The Marketing Engine's primary path is all API-based. Browser automation is the fallback layer.

---

## 3D Printer Web UI Integration (Future)

When you get a 3D printer, the browser adapter can interact with its web dashboard:

```python
# core/browser/actions/printer_ui.py
class PrinterUIAction:
    async def upload_and_print(self, engine, gcode_path: str, printer_url: str):
        page = engine.page
        await page.goto(printer_url)

        # Upload G-code file
        upload = await page.wait_for_selector('input[type="file"]')
        await upload.set_input_files(gcode_path)

        # Start print
        print_btn = await page.wait_for_selector('button:has-text("Print")')
        await print_btn.click()

        # Monitor progress
        while True:
            progress = await page.text_content('[class*="progress"]')
            if "complete" in progress.lower():
                return {"status": "complete"}
            await asyncio.sleep(30)
```

**Note:** Most serious printers (Prusa, Creality with Klipper) have proper REST APIs via OctoPrint or Moonraker. Browser automation for printers is only needed for closed-ecosystem printers like Bambu Lab. Consider this when choosing which printer to buy.
