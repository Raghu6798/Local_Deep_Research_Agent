import asyncio
import concurrent.futures
from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CacheMode,
    DefaultMarkdownGenerator,
)

# --- Configuration (can be defined globally as it's the same for all tasks) ---

# Define the HTML tags to be removed before markdown generation.
TAGS_TO_EXCLUDE = ["header", "footer", "nav", "aside", "script", "style", "form"]

# Configure the crawler run once.
RUN_CONFIG = CrawlerRunConfig(
    excluded_tags=TAGS_TO_EXCLUDE,
    cache_mode=CacheMode.BYPASS,
    markdown_generator=DefaultMarkdownGenerator()
)

# --- The Core Asynchronous Crawling Logic ---

async def crawl_one_url(url: str) -> tuple[str, str | None]:
    """
    An async function that performs a single crawl.
    This is what will be run inside the asyncio.run() bridge.
    Returns the URL and the cleaned markdown, or None on failure.
    """
    print(f"[THREAD] Starting crawl for: {url}")
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url, config=RUN_CONFIG)
            
            markdown_result = result.markdown
            if result.success and markdown_result and markdown_result.raw_markdown:
                print(f"[THREAD] SUCCESS for: {url}")
                return (url, markdown_result.raw_markdown.strip())
            else:
                error = result.error_message or "No markdown content"
                print(f"[THREAD] FAILED for: {url} - {error}")
                return (url, None)
    except Exception as e:
        print(f"[THREAD] CRITICAL FAILURE for: {url} - {e}")
        return (url, None)

# --- The Synchronous Worker (The Bridge) ---

def worker_task(url: str) -> tuple[str, str | None]:
    """
    A synchronous function that the ThreadPoolExecutor will call.
    It creates a new asyncio event loop to run our async crawl function.
    """
    return asyncio.run(crawl_one_url(url))


# --- The Main Orchestrator ---

def main():
    """
    Orchestrates the crawling using a ThreadPoolExecutor.
    """
    urls_to_crawl = [
        "https://www.w3.org/WAI/fundamentals/accessibility-intro/",
        "https://developer.mozilla.org/en-US/docs/Web/Accessibility",
        "https://www.a11yproject.com/",
        "https://web.dev/learn/accessibility",
    ]

    # This now controls your concurrency, just like max_session_permit did.
    # It defines the number of threads (and thus browsers) to run at once.
    max_workers = 2 

    aggregated_markdown = []
    print(f"Starting crawl with a ThreadPoolExecutor (max_workers={max_workers})...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map applies the worker_task to each URL and returns results
        # as they are completed, maintaining the original order.
        results = executor.map(worker_task, urls_to_crawl)

        for url, markdown_content in results:
            if markdown_content:
                aggregated_markdown.append(f"## Content Source: {url}\n\n")
                aggregated_markdown.append(markdown_content)
                aggregated_markdown.append("\n\n---\n\n")
            else:
                print(f"[MAIN] Skipping failed URL in final aggregation: {url}")


    # Combine all the cleaned markdown content into a single string
    final_markdown = "".join(aggregated_markdown)

    # Write the aggregated content to a .md file
    output_filename = "aggregated_threaded_results.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_markdown)

    print(f"\nCrawling complete. Aggregated results saved to '{output_filename}'")


if __name__ == "__main__":
    main()