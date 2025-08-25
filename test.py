import os
import json
import requests
import asyncio
from playwright.async_api import async_playwright, Playwright
import concurrent.futures
from dotenv import load_dotenv
from exa_py import Exa
from typing import Dict, List, Optional, Sequence, Type, Callable, Union,Any

# --- LangChain Core Imports ---
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.output_parsers.openai_tools import parse_tool_calls

from langgraph.prebuilt import create_react_agent
# --- Crawling Library Import ---
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, DefaultMarkdownGenerator,CacheMode

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================

load_dotenv()

try:
    exa = Exa(api_key=os.getenv("EXA_SEARCH_API"))
except Exception as e:
    print(f"Error initializing Exa client: {e}")
    print("Please ensure your EXA_SEARCH_API key is correctly set in your .env file.")
    exit()


TAGS_TO_EXCLUDE = ["header", "footer", "nav", "aside", "script", "style", "form"]

# Configure the crawler run once to be used by all threads.
CRAWLER_RUN_CONFIG = CrawlerRunConfig(
    excluded_tags=TAGS_TO_EXCLUDE,
    cache_mode=CacheMode.BYPASS,
    markdown_generator=DefaultMarkdownGenerator()
)
CRAWLER_BROWSER_CONFIG = BrowserConfig(headless=True, verbose=False)

# ==============================================================================
# 2. CUSTOM CHAT MODEL (ChatQwen - Unchanged)
# ==============================================================================

class ChatQwen(BaseChatModel):
    """LangChain wrapper for a local OpenAI-compatible server with full tool support."""
    base_url: str = "http://127.0.0.1:8082"
    model_name: str = "Qwen2.5-7B-Instruct"
    temperature: float = 0.1

    @property
    def _llm_type(self) -> str: return "Qwen2.5-7B-Instruct-with-Tools"
    def _send_request(self, messages: List[BaseMessage], **kwargs: Any) -> Dict[str, Any]:
        processed_messages = []
        for m in messages:
            if isinstance(m, SystemMessage): processed_messages.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage): processed_messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                msg_dict = {"role": "assistant", "content": m.content or ""}
                if m.tool_calls:
                    msg_dict["tool_calls"] = [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}} for tc in m.tool_calls]
                processed_messages.append(msg_dict)
            elif isinstance(m, ToolMessage): processed_messages.append({"role": "tool", "content": m.content, "tool_call_id": m.tool_call_id})
            else: raise TypeError(f"Unsupported message type: {type(m)}")
        payload = {"model": self.model_name, "messages": processed_messages, "temperature": self.temperature, **kwargs}
        if "tools" in payload and payload["tools"] is None: del payload["tools"]
        response = requests.post(f"{self.base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        resp_json = self._send_request(messages, **kwargs)
        message_data = resp_json["choices"][0]["message"]
        content = message_data.get("content", "") or ""
        tool_calls = parse_tool_calls(message_data["tool_calls"]) if message_data.get("tool_calls") else []
        ai_message = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])
    def bind_tools(self, tools: Sequence[Union[Dict, Type[BaseModel], Callable, BaseTool]], **kwargs: Any) -> Runnable:
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted_tools, **kwargs)
    def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any):
        raise NotImplementedError("Async generation not implemented for local model.")

# ==============================================================================
# 3. TOOL DEFINITIONS & CONCURRENT CRAWLING LOGIC
# ==============================================================================

# --- Tool 1: Search for URLs (Unchanged) ---
class ExaUrlSearchRequest(BaseModel):
    query: str = Field(..., description="A clear, descriptive search query.")
    numResults: Optional[int] = Field(5, description="The number of URLs to retrieve.")

@tool("exa_search_for_urls", args_schema=ExaUrlSearchRequest)
def exa_search_for_urls(**kwargs) -> str:
    """Searches the web for a given query and returns a list of relevant URLs."""
    try:
        request = ExaUrlSearchRequest(**kwargs)
        response = exa.search(request.query, num_results=request.numResults, use_autoprompt=True)
        url_list = [result.url for result in response.results]
        return json.dumps({"urls": url_list})
    except Exception as e:
        return json.dumps({"error": f"Search failed: {e}"})

# --- Tool 2: Crawl a Website (Placeholder Definition) ---
class CrawlWebsiteRequest(BaseModel):
    url: str = Field(..., description="The single URL of the website to crawl.")

@tool("crawl_website_for_content", args_schema=CrawlWebsiteRequest)
def crawl_website_for_content(**kwargs) -> str:
    """
    Crawls a single website URL to get its content. 
    NOTE: The agent orchestrator handles the actual execution of this tool.
    This definition is for the LLM to know the tool exists and how to call it.
    """
    # This function is never actually called directly. The agent loop intercepts
    # calls to this tool and uses our multithreaded handler instead.
    return "Crawl job submitted."

# --- Core Asynchronous Crawling Logic for a Single URL ---
async def crawl_one_url_async(url: str) -> tuple[str, str | None]:
    """An async function that performs a single, cleaned crawl."""
    print(f"  [CRAWL-THREAD] Starting crawl for: {url}")
    try:
        async with AsyncWebCrawler(config=CRAWLER_BROWSER_CONFIG) as crawler:
            result = await crawler.arun(url, config=CRAWLER_RUN_CONFIG)
            markdown_result = result.markdown
            if result.success and markdown_result and markdown_result.raw_markdown:
                print(f"  [CRAWL-THREAD] SUCCESS for: {url}")
                return (url, markdown_result.raw_markdown.strip())
            else:
                error = result.error_message or "No markdown content"
                print(f"  [CRAWL-THREAD] FAILED for: {url} - {error}")
                return (url, f"Failed to crawl {url}: {error}")
    except Exception as e:
        print(f"  [CRAWL-THREAD] CRITICAL FAILURE for: {url} - {e}")
        return (url, f"A critical error occurred while crawling {url}: {e}")

# --- Synchronous Bridge for Threading ---
def worker_task_bridge(url: str) -> tuple[str, str | None]:
    """A synchronous function that the ThreadPoolExecutor calls to run the async crawl."""
    return asyncio.run(crawl_one_url_async(url))

# --- Main Concurrent Crawl Orchestrator ---
def run_crawls_with_threadpool(urls_to_crawl: List[str], max_workers: int = 2) -> Dict[str, str]:
    """
    Orchestrates multiple crawls using a ThreadPoolExecutor for memory safety.
    Returns a dictionary mapping each URL to its crawled content or an error message.
    """
    results_dict = {}
    print(f"  > Starting concurrent crawl for {len(urls_to_crawl)} URLs with {max_workers} threads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # map() efficiently applies the worker function to each URL
        future_results = executor.map(worker_task_bridge, urls_to_crawl)
        for url, content in future_results:
            results_dict[url] = content
    return results_dict


@tool 
async def playwright_scraper():
    """

    """


# ==============================================================================
# 4. MAIN AGENT EXECUTION
# ==============================================================================

if __name__ == '__main__':
    # --- Agent Initialization ---
    local_model = ChatQwen()
    tools = [exa_search_for_urls,crawl_website_for_content]
    system_prompt = """You are a powerful research assistant with two tools:
1. `exa_search_for_urls`: To find relevant web pages for a query.
2. `crawl_website_for_content`: To read the content of a specific URL.

Your workflow is as follows:
1. First, use `exa_search_for_urls` to get a list of URLs based on the user's query.
2. Second, after you receive the list of URLs, you MUST call `crawl_website_for_content` for the top 3 most promising URLs from the list to read their content. You should make multiple, parallel tool calls in a single turn for this.
3. Finally, once you have gathered the content from the websites, synthesize ALL the information into a single, comprehensive answer to the user's original query. Do not list the URLs again, just provide the answer.
"""
    model_with_tools =create_react_agent(model=local_model,tools = tools,prompt = system_prompt)
    available_tools = { "exa_search_for_urls": exa_search_for_urls }
    
    user_query = "What's the trending news in LLMs?"
    final_answer = None
    for chunk in model_with_tools.stream({"messages": [{"role": "user", "content": user_query}]}):

        # The 'agent' key holds the output from the core LLM.
        if "agent" in chunk:
            # Get the list of messages from the agent's output
            agent_messages = chunk["agent"]["messages"]
            if agent_messages:
                # The most recent message is the last one in the list.
                last_message = agent_messages[-1]

                # A final answer is an AIMessage that has content
                # but does NOT have any tool_calls.
                if isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls:
                    final_answer = last_message.content

    # After the stream is complete, print the captured final answer.
    if final_answer:
        print("==================================================")
        print("Final Answer:")
        print("==================================================")
        print(final_answer)
    else:
        print("Could not retrieve a final answer from the agent.")


   