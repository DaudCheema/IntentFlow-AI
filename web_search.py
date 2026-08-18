import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

# Updated import for the new ddgs package
from ddgs import DDGS
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("AgentLogger")

BLOCKED_DOMAINS = [
    "youtube.com", "youtu.be", "facebook.com", "twitter.com", "tiktok.com",
    "pinterest.com", "instagram.com"
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def _scrape_single_url(url: str) -> Tuple[str, str]:
    """Custom scraper: fetches page body and cleans HTML text with a 3.5s cutoff."""
    try:
        with httpx.Client(headers=HEADERS, verify=False, timeout=3.5, follow_redirects=True) as client:
            response = client.get(url)
            if response.status_code != 200:
                return url, ""

            soup = BeautifulSoup(response.text, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "svg"]):
                element.decompose()

            text = soup.get_text(separator=" ", strip=True)
            cleaned_text = re.sub(r"\s+", " ", text).strip()
            return url, cleaned_text[:1800]
    except Exception:
        return url, ""


def perform_web_search(query: str, max_results: int = 4) -> Tuple[str, List[str]]:
    """Performs search and fetches pages in parallel using standard ThreadPoolExecutor."""
    logger.info(f"[WEB SEARCH] Initializing search for query: '{query}'")
    candidate_links = []

    try:
        with DDGS(timeout=7) as ddgs:
            # 1. First attempt standard web search
            raw_results = list(ddgs.text(query, max_results=max_results + 2))
            
            # 2. If standard search yields no results (e.g. for general news terms), try news search
            if not raw_results:
                raw_results = list(ddgs.news(query, max_results=max_results + 2))

            for res in raw_results:
                href = res.get("href") or res.get("url") or ""
                body = res.get("body") or res.get("snippet") or ""
                if href and not any(blocked in href.lower() for blocked in BLOCKED_DOMAINS):
                    candidate_links.append({
                        "url": href,
                        "snippet": body
                    })
                if len(candidate_links) >= max_results:
                    break
    except Exception as e:
        logger.error(f"[WEB SEARCH ERROR] DDGS Search failed: {e}")
        return "", []

    if not candidate_links:
        logger.warning("[WEB SEARCH] No valid search results found.")
        return "", []

    urls_to_scrape = [item["url"] for item in candidate_links]

    # Parallel scraping via multi-threading
    scraped_map = {}
    with ThreadPoolExecutor(max_workers=len(urls_to_scrape)) as executor:
        futures = {executor.submit(_scrape_single_url, url): url for url in urls_to_scrape}
        for future in as_completed(futures):
            url, text = future.result()
            scraped_map[url] = text

    compiled_snippets = []
    final_sources = []

    logger.info(f"[SCRAPER] Processing candidate URLs ({len(candidate_links)} found)...")

    for item in candidate_links:
        url = item["url"]
        body_text = scraped_map.get(url, "")
        snippet = item["snippet"]

        if body_text and len(body_text) > 100:
            # Logs successful full-page scrapes with character length
            logger.info(f"   ├─ [PARSED BODY] Extracted {len(body_text)} chars from: {url}")
            compiled_snippets.append(f"[SOURCE: {url}]\n{body_text}")
            final_sources.append(url)
        elif snippet:
            # Logs fallback to search snippet when scraping fails or gets blocked
            logger.info(f"   ├─ [SNIPPET FALLBACK] Used search snippet ({len(snippet)} chars) from: {url}")
            compiled_snippets.append(f"[SOURCE (Snippet): {url}]\n{snippet}")
            final_sources.append(url)
        else:
            # Logs when both scraper and snippet returned nothing
            logger.warning(f"   └─ [SKIPPED] No readable text found for: {url}")

    combined_context = "\n\n---\n\n".join(compiled_snippets)
    logger.info(f"[WEB SEARCH COMPLETE] Total combined context compiled: {len(combined_context)} chars from {len(final_sources)} source(s).")
    return combined_context, final_sources