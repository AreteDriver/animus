---
name: web_search
version: 1.0.0
agent: browser
risk_level: low
description: "Search the web for information using search engines. Retrieve, parse, and synthesize search results. Supports multiple search engines and result filtering."
---

# Web Search Skill

## Purpose

Search the web for information to support Gorgon tasks. This skill enables agents to gather current information, research topics, verify facts, and find resources not available in local knowledge.

## Safety Rules

### CONTENT RESTRICTIONS
1. **Never search for illegal content** - CSAM, weapons manufacturing, etc.
2. **Never search for personal information for stalking/harassment**
3. **Avoid excessive automated queries** - respect rate limits
4. **Never bypass CAPTCHAs or anti-bot measures**
5. **Cache results when appropriate** - reduce redundant queries

### PRIVACY PRACTICES
1. **Use privacy-respecting search engines when possible** (DuckDuckGo)
2. **Don't include personal identifiers in searches without explicit need**
3. **Clear search history after task completion**

### CONSENSUS REQUIREMENTS
| Operation | Risk Level | Consensus Required |
|-----------|------------|-------------------|
| web_search | low | any |
| image_search | low | any |
| news_search | low | any |
| site_search | low | any |

## Capabilities

### web_search
Perform a general web search.

**Using curl + DuckDuckGo HTML:**
```bash
# Simple search (DuckDuckGo HTML version - no JS required)
query="python asyncio tutorial"
encoded_query=$(echo "$query" | sed 's/ /+/g')
curl -s "https://html.duckduckgo.com/html/?q=${encoded_query}" | \
  grep -oP '(?<=<a rel="nofollow" class="result__a" href=")[^"]*' | \
  head -10
```

**Using Python + requests:**
```python
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

def web_search(query: str, num_results: int = 10) -> list[dict]:
    """Search DuckDuckGo and return results."""
    
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
    }
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    results = []
    for result in soup.select('.result')[:num_results]:
        title_elem = result.select_one('.result__a')
        snippet_elem = result.select_one('.result__snippet')
        
        if title_elem:
            results.append({
                "title": title_elem.get_text(strip=True),
                "url": title_elem.get('href', ''),
                "snippet": snippet_elem.get_text(strip=True) if snippet_elem else ""
            })
    
    return results

# Usage
results = web_search("Gorgon multi-agent orchestration")
for r in results:
    print(f"- {r['title']}\n  {r['url']}\n  {r['snippet']}\n")
```

**Using Playwright for JS-heavy sites:**
```python
from playwright.sync_api import sync_playwright

def search_with_playwright(query: str) -> list[dict]:
    """Search using full browser for JS-rendered results."""
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # DuckDuckGo
        page.goto(f"https://duckduckgo.com/?q={query}")
        page.wait_for_selector('.result', timeout=10000)
        
        results = []
        for result in page.query_selector_all('.result')[:10]:
            title = result.query_selector('.result__a')
            snippet = result.query_selector('.result__snippet')
            
            if title:
                results.append({
                    "title": title.inner_text(),
                    "url": title.get_attribute('href'),
                    "snippet": snippet.inner_text() if snippet else ""
                })
        
        browser.close()
        return results
```

---

### news_search
Search for recent news articles.

**Using DuckDuckGo News:**
```python
def news_search(query: str, num_results: int = 10) -> list[dict]:
    """Search DuckDuckGo News."""
    
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}&iar=news"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
    }
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    results = []
    for result in soup.select('.result')[:num_results]:
        title_elem = result.select_one('.result__a')
        snippet_elem = result.select_one('.result__snippet')
        
        if title_elem:
            results.append({
                "title": title_elem.get_text(strip=True),
                "url": title_elem.get('href', ''),
                "snippet": snippet_elem.get_text(strip=True) if snippet_elem else "",
                "type": "news"
            })
    
    return results
```

**Alternative: Using RSS feeds for specific sources:**
```python
import feedparser

def search_rss_news(feed_url: str, query: str) -> list[dict]:
    """Search RSS feed for matching entries."""
    
    feed = feedparser.parse(feed_url)
    query_lower = query.lower()
    
    results = []
    for entry in feed.entries:
        if query_lower in entry.title.lower() or query_lower in entry.get('summary', '').lower():
            results.append({
                "title": entry.title,
                "url": entry.link,
                "published": entry.get('published', ''),
                "snippet": entry.get('summary', '')[:200]
            })
    
    return results

# Common news RSS feeds
RSS_FEEDS = {
    "tech": "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "bbc": "http://feeds.bbci.co.uk/news/rss.xml",
    "reuters": "https://www.reutersagency.com/feed/",
}
```

---

### image_search
Search for images.

**Using DuckDuckGo Images:**
```python
def image_search(query: str, num_results: int = 10) -> list[dict]:
    """Search for images via DuckDuckGo."""
    
    # DuckDuckGo Images requires JavaScript, use the API endpoint
    url = "https://duckduckgo.com/i.js"
    params = {
        "q": query,
        "o": "json",
        "p": 1,
        "s": 0
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
    }
    
    # First get vqd token
    token_response = requests.get(f"https://duckduckgo.com/?q={quote_plus(query)}", headers=headers)
    vqd_match = re.search(r'vqd="([^"]+)"', token_response.text)
    
    if vqd_match:
        params['vqd'] = vqd_match.group(1)
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        results = []
        for img in data.get('results', [])[:num_results]:
            results.append({
                "title": img.get('title', ''),
                "url": img.get('image', ''),
                "thumbnail": img.get('thumbnail', ''),
                "source": img.get('source', ''),
                "width": img.get('width'),
                "height": img.get('height')
            })
        
        return results
    
    return []
```

---

### site_search
Search within a specific website.

**Usage:**
```python
def site_search(query: str, site: str, num_results: int = 10) -> list[dict]:
    """Search within a specific domain."""
    
    full_query = f"site:{site} {query}"
    return web_search(full_query, num_results)

# Examples
results = site_search("asyncio", "docs.python.org")
results = site_search("multi-agent", "github.com")
```

---

### search_and_summarize
Search and provide a synthesized summary.

**Usage:**
```python
def search_and_summarize(query: str) -> dict:
    """Search and return structured summary."""
    
    results = web_search(query, num_results=5)
    
    return {
        "query": query,
        "result_count": len(results),
        "top_results": results,
        "sources": [r['url'] for r in results],
        "key_snippets": [r['snippet'] for r in results if r['snippet']]
    }
```

---

## Examples

### Example 1: Research a technical topic
**Intent:** "Find information about Python asyncio best practices"

**Execution:**
```python
# Search for the topic
results = web_search("Python asyncio best practices 2024")

# Output structured results
print(f"Found {len(results)} results:\n")
for i, r in enumerate(results, 1):
    print(f"{i}. {r['title']}")
    print(f"   URL: {r['url']}")
    print(f"   {r['snippet'][:150]}...")
    print()
```

**Output:**
```
Found 10 results:

1. Python Asyncio Best Practices - Real Python
   URL: https://realpython.com/async-io-python/
   Learn how to use asyncio for concurrent programming in Python...

2. Asyncio Patterns and Best Practices - Stack Overflow
   URL: https://stackoverflow.com/questions/...
   Best practices for structuring asyncio applications...
```

---

### Example 2: Find recent news about a topic
**Intent:** "Get latest news about AI agent frameworks"

**Execution:**
```python
# Search news
results = news_search("AI agent framework LLM", num_results=5)

# Format for reporting
print("Recent AI Agent Framework News:\n")
for r in results:
    print(f"📰 {r['title']}")
    print(f"   {r['url']}")
    print(f"   {r['snippet'][:100]}...")
    print()
```

---

### Example 3: Search specific documentation site
**Intent:** "Find Playwright documentation about browser contexts"

**Execution:**
```python
# Search within Playwright docs
results = site_search("browser context isolation", "playwright.dev")

print("Playwright Documentation Results:\n")
for r in results:
    print(f"📄 {r['title']}")
    print(f"   {r['url']}")
    print()
```

---

## Rate Limiting

To be a good web citizen:

```python
import time
from functools import wraps

# Simple rate limiter
last_request_time = 0
MIN_REQUEST_INTERVAL = 2.0  # seconds

def rate_limited(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global last_request_time
        
        elapsed = time.time() - last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        
        result = func(*args, **kwargs)
        last_request_time = time.time()
        
        return result
    return wrapper

@rate_limited
def web_search(query: str, num_results: int = 10) -> list[dict]:
    # ... search implementation
    pass
```

## Caching

Cache results to reduce redundant queries:

```python
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta

CACHE_DIR = Path.home() / ".gorgon" / "cache" / "web_search"
CACHE_TTL = timedelta(hours=1)

def get_cache_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()

def get_cached_results(query: str) -> list[dict] | None:
    cache_file = CACHE_DIR / f"{get_cache_key(query)}.json"
    
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        cached_time = datetime.fromisoformat(data['timestamp'])
        
        if datetime.now() - cached_time < CACHE_TTL:
            return data['results']
    
    return None

def cache_results(query: str, results: list[dict]):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{get_cache_key(query)}.json"
    
    cache_file.write_text(json.dumps({
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'results': results
    }))
```

## Error Handling

| Error | Response |
|-------|----------|
| Rate limited (429) | Wait and retry with exponential backoff |
| Connection timeout | Retry up to 3 times |
| No results found | Try alternative query formulations |
| CAPTCHA encountered | Report, do not bypass |
| Invalid URL in results | Skip and continue |

## Output Format

```json
{
  "success": true,
  "operation": "web_search",
  "query": "Python asyncio tutorial",
  "results": [
    {
      "title": "Async IO in Python: A Complete Walkthrough",
      "url": "https://realpython.com/async-io-python/",
      "snippet": "Learn how to use Python's asyncio module..."
    }
  ],
  "result_count": 10,
  "cached": false,
  "timestamp": "2026-01-27T10:30:00Z"
}
```
