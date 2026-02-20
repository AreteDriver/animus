---
name: web_scrape
version: 1.0.0
agent: browser
risk_level: low
description: "Fetch and extract content from web pages. Parse HTML, extract text, tables, and structured data. Handle JavaScript-rendered content with headless browser."
---

# Web Scrape Skill

## Purpose

Fetch web pages and extract structured content. This skill enables Gorgon agents to retrieve full page content, extract specific elements, parse tables, and handle dynamic JavaScript-rendered content.

## Safety Rules

### ETHICAL SCRAPING
1. **Respect robots.txt** - check before scraping
2. **Honor rate limits** - don't overwhelm servers
3. **Identify as bot** - use honest User-Agent when appropriate
4. **Cache aggressively** - don't re-fetch unchanged content
5. **Respect noindex/nofollow** - honor meta directives

### CONTENT RESTRICTIONS
1. **Never scrape login-protected content without credentials**
2. **Never bypass paywalls**
3. **Never scrape for personal data harvesting**
4. **Respect copyright** - don't bulk-download copyrighted content

### CONSENSUS REQUIREMENTS
| Operation | Risk Level | Consensus Required |
|-----------|------------|-------------------|
| fetch_page | low | any |
| extract_text | low | any |
| extract_tables | low | any |
| extract_links | low | any |
| screenshot | low | any |
| fill_form | medium | majority |

## Capabilities

### fetch_page
Retrieve a web page's HTML content.

**Using curl:**
```bash
# Simple fetch
curl -s "https://example.com/page" -o page.html

# With headers
curl -s "https://example.com/page" \
  -H "User-Agent: GorgonBot/1.0 (+https://github.com/yourrepo)" \
  -H "Accept: text/html" \
  -o page.html

# Follow redirects
curl -sL "https://example.com/page" -o page.html

# With timeout
curl -s --max-time 30 "https://example.com/page" -o page.html
```

**Using Python requests:**
```python
import requests
from urllib.parse import urlparse

def fetch_page(url: str, timeout: int = 30) -> dict:
    """Fetch a web page with proper error handling."""
    
    headers = {
        "User-Agent": "GorgonBot/1.0 (+https://github.com/yourrepo)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        return {
            "success": True,
            "url": response.url,  # Final URL after redirects
            "status_code": response.status_code,
            "content_type": response.headers.get('content-type', ''),
            "html": response.text,
            "encoding": response.encoding,
        }
    
    except requests.RequestException as e:
        return {
            "success": False,
            "url": url,
            "error": str(e)
        }
```

**Using Playwright for JavaScript-rendered content:**
```python
from playwright.sync_api import sync_playwright

def fetch_js_page(url: str, wait_for: str = None) -> dict:
    """Fetch page with full JavaScript rendering."""
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="GorgonBot/1.0 (+https://github.com/yourrepo)"
        )
        page = context.new_page()
        
        try:
            page.goto(url, wait_until="networkidle")
            
            if wait_for:
                page.wait_for_selector(wait_for, timeout=10000)
            
            return {
                "success": True,
                "url": page.url,
                "title": page.title(),
                "html": page.content(),
            }
        
        except Exception as e:
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
        
        finally:
            browser.close()
```

---

### extract_text
Extract readable text content from HTML.

**Using BeautifulSoup:**
```python
from bs4 import BeautifulSoup
import re

def extract_text(html: str, selector: str = None) -> dict:
    """Extract clean text from HTML."""
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    
    if selector:
        # Extract from specific element
        target = soup.select_one(selector)
        if target:
            text = target.get_text(separator='\n', strip=True)
        else:
            text = ""
    else:
        # Extract all text
        text = soup.get_text(separator='\n', strip=True)
    
    # Clean up whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return {
        "success": True,
        "text": text,
        "char_count": len(text),
        "word_count": len(text.split())
    }
```

**Extract main article content:**
```python
def extract_article(html: str) -> dict:
    """Extract main article content, removing boilerplate."""
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Try common article selectors
    article_selectors = [
        'article',
        'main',
        '[role="main"]',
        '.post-content',
        '.article-content',
        '.entry-content',
        '#content',
    ]
    
    for selector in article_selectors:
        article = soup.select_one(selector)
        if article and len(article.get_text(strip=True)) > 200:
            # Clean it up
            for element in article(['script', 'style', 'nav', 'aside']):
                element.decompose()
            
            return {
                "success": True,
                "title": soup.title.string if soup.title else "",
                "content": article.get_text(separator='\n', strip=True),
                "selector_used": selector
            }
    
    # Fallback to body
    body = soup.body
    if body:
        for element in body(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        return {
            "success": True,
            "title": soup.title.string if soup.title else "",
            "content": body.get_text(separator='\n', strip=True),
            "selector_used": "body (fallback)"
        }
    
    return {"success": False, "error": "Could not extract content"}
```

---

### extract_tables
Extract tables from HTML into structured data.

```python
import pandas as pd
from bs4 import BeautifulSoup

def extract_tables(html: str) -> dict:
    """Extract all tables from HTML."""
    
    soup = BeautifulSoup(html, 'html.parser')
    tables = []
    
    for i, table in enumerate(soup.find_all('table')):
        try:
            # Try pandas for clean extraction
            df = pd.read_html(str(table))[0]
            tables.append({
                "index": i,
                "rows": len(df),
                "columns": list(df.columns),
                "data": df.to_dict('records')
            })
        except Exception:
            # Fallback to manual extraction
            rows = []
            for tr in table.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if cells:
                    rows.append(cells)
            
            if rows:
                tables.append({
                    "index": i,
                    "rows": len(rows),
                    "columns": rows[0] if rows else [],
                    "data": rows[1:] if len(rows) > 1 else rows
                })
    
    return {
        "success": True,
        "table_count": len(tables),
        "tables": tables
    }
```

---

### extract_links
Extract all links from a page.

```python
from urllib.parse import urljoin, urlparse

def extract_links(html: str, base_url: str, filter_domain: bool = False) -> dict:
    """Extract all links from HTML."""
    
    soup = BeautifulSoup(html, 'html.parser')
    base_domain = urlparse(base_url).netloc
    
    links = []
    seen = set()
    
    for a in soup.find_all('a', href=True):
        href = a['href']
        
        # Make absolute
        absolute_url = urljoin(base_url, href)
        
        # Skip if already seen
        if absolute_url in seen:
            continue
        seen.add(absolute_url)
        
        # Parse URL
        parsed = urlparse(absolute_url)
        
        # Filter by domain if requested
        if filter_domain and parsed.netloc != base_domain:
            continue
        
        # Skip non-http(s)
        if parsed.scheme not in ('http', 'https'):
            continue
        
        links.append({
            "url": absolute_url,
            "text": a.get_text(strip=True),
            "title": a.get('title', ''),
            "external": parsed.netloc != base_domain
        })
    
    return {
        "success": True,
        "link_count": len(links),
        "links": links
    }
```

---

### extract_metadata
Extract page metadata (title, description, OpenGraph, etc.)

```python
def extract_metadata(html: str, url: str) -> dict:
    """Extract page metadata."""
    
    soup = BeautifulSoup(html, 'html.parser')
    
    metadata = {
        "url": url,
        "title": None,
        "description": None,
        "keywords": None,
        "author": None,
        "published_date": None,
        "og": {},
        "twitter": {}
    }
    
    # Title
    if soup.title:
        metadata["title"] = soup.title.string
    
    # Meta tags
    for meta in soup.find_all('meta'):
        name = meta.get('name', meta.get('property', '')).lower()
        content = meta.get('content', '')
        
        if name == 'description':
            metadata["description"] = content
        elif name == 'keywords':
            metadata["keywords"] = content
        elif name == 'author':
            metadata["author"] = content
        elif name in ('article:published_time', 'date'):
            metadata["published_date"] = content
        elif name.startswith('og:'):
            metadata["og"][name[3:]] = content
        elif name.startswith('twitter:'):
            metadata["twitter"][name[8:]] = content
    
    return {"success": True, "metadata": metadata}
```

---

### screenshot
Take a screenshot of a web page.

```python
from playwright.sync_api import sync_playwright
from pathlib import Path

def screenshot(url: str, output_path: str, full_page: bool = False) -> dict:
    """Take a screenshot of a web page."""
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        
        try:
            page.goto(url, wait_until="networkidle")
            
            page.screenshot(
                path=output_path,
                full_page=full_page
            )
            
            return {
                "success": True,
                "url": url,
                "output_path": output_path,
                "full_page": full_page
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
        
        finally:
            browser.close()
```

---

### check_robots_txt
Check if URL is allowed by robots.txt.

```python
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

def check_robots(url: str, user_agent: str = "GorgonBot") -> dict:
    """Check if URL is allowed by robots.txt."""
    
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    
    rp = RobotFileParser()
    rp.set_url(robots_url)
    
    try:
        rp.read()
        allowed = rp.can_fetch(user_agent, url)
        crawl_delay = rp.crawl_delay(user_agent)
        
        return {
            "success": True,
            "url": url,
            "robots_url": robots_url,
            "allowed": allowed,
            "crawl_delay": crawl_delay
        }
    
    except Exception as e:
        # If robots.txt doesn't exist or errors, assume allowed
        return {
            "success": True,
            "url": url,
            "allowed": True,
            "note": f"Could not fetch robots.txt: {e}"
        }
```

---

## Examples

### Example 1: Scrape documentation page
**Intent:** "Get the content of the Python asyncio documentation"

**Execution:**
```python
# Check robots.txt first
robots = check_robots("https://docs.python.org/3/library/asyncio.html")
print(f"Allowed: {robots['allowed']}")

if robots['allowed']:
    # Fetch page
    result = fetch_page("https://docs.python.org/3/library/asyncio.html")
    
    if result['success']:
        # Extract article content
        article = extract_article(result['html'])
        
        print(f"Title: {article['title']}")
        print(f"Content length: {len(article['content'])} chars")
        print(f"\n{article['content'][:500]}...")
```

---

### Example 2: Extract data table from Wikipedia
**Intent:** "Get the table of programming languages from Wikipedia"

**Execution:**
```python
# Fetch page
result = fetch_page("https://en.wikipedia.org/wiki/List_of_programming_languages")

if result['success']:
    # Extract all tables
    tables = extract_tables(result['html'])
    
    print(f"Found {tables['table_count']} tables")
    
    # Get first table with data
    for table in tables['tables']:
        if table['rows'] > 5:
            print(f"\nTable with {table['rows']} rows:")
            print(f"Columns: {table['columns']}")
            for row in table['data'][:5]:
                print(f"  {row}")
            break
```

---

### Example 3: Crawl links from a page
**Intent:** "Find all documentation links on a page"

**Execution:**
```python
# Fetch page
result = fetch_page("https://docs.python.org/3/")

if result['success']:
    # Extract links
    links = extract_links(
        result['html'], 
        "https://docs.python.org/3/",
        filter_domain=True  # Only same domain
    )
    
    print(f"Found {links['link_count']} internal links:\n")
    
    # Filter to documentation links
    doc_links = [l for l in links['links'] if '/library/' in l['url']]
    
    for link in doc_links[:10]:
        print(f"- {link['text']}: {link['url']}")
```

---

## Rate Limiting and Caching

```python
import time
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta

class ScraperWithCache:
    CACHE_DIR = Path.home() / ".gorgon" / "cache" / "scraper"
    CACHE_TTL = timedelta(hours=24)
    MIN_REQUEST_INTERVAL = 1.0  # seconds
    
    def __init__(self):
        self.last_request_time = {}  # Per-domain
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def _get_domain(self, url: str) -> str:
        return urlparse(url).netloc
    
    def _rate_limit(self, url: str):
        domain = self._get_domain(url)
        last_time = self.last_request_time.get(domain, 0)
        elapsed = time.time() - last_time
        
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        
        self.last_request_time[domain] = time.time()
    
    def _cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cached(self, url: str) -> dict | None:
        cache_file = self.CACHE_DIR / f"{self._cache_key(url)}.json"
        
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            cached_time = datetime.fromisoformat(data['timestamp'])
            
            if datetime.now() - cached_time < self.CACHE_TTL:
                data['cached'] = True
                return data
        
        return None
    
    def _save_cache(self, url: str, data: dict):
        cache_file = self.CACHE_DIR / f"{self._cache_key(url)}.json"
        data['timestamp'] = datetime.now().isoformat()
        cache_file.write_text(json.dumps(data))
    
    def fetch(self, url: str) -> dict:
        # Check cache first
        cached = self._get_cached(url)
        if cached:
            return cached
        
        # Check robots.txt
        robots = check_robots(url)
        if not robots['allowed']:
            return {"success": False, "error": "Blocked by robots.txt"}
        
        # Rate limit
        self._rate_limit(url)
        
        # Fetch
        result = fetch_page(url)
        
        if result['success']:
            self._save_cache(url, result)
        
        return result
```

## Error Handling

| Error | Response |
|-------|----------|
| Connection timeout | Retry up to 3 times with backoff |
| 403 Forbidden | Check robots.txt, try with different User-Agent |
| 404 Not Found | Report, do not retry |
| 429 Too Many Requests | Wait and retry with longer delay |
| SSL Error | Report, suggest http fallback if appropriate |
| Encoding Error | Try common encodings (utf-8, latin-1) |

## Output Format

```json
{
  "success": true,
  "operation": "fetch_page",
  "url": "https://example.com/page",
  "status_code": 200,
  "content_type": "text/html",
  "cached": false,
  "timestamp": "2026-01-27T10:30:00Z"
}
```
