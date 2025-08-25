import os
import json
from dotenv import load_dotenv
from exa_py import Exa
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================

def initialize_exa_client() -> Optional[Exa]:
    """Loads API key and safely initializes the Exa client."""
    load_dotenv()
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        print("FATAL ERROR: EXA_SEARCH_API key not found.")
        print("Please ensure your API key is correctly set in your .env file.")
        return None
    try:
        print("Initializing Exa client...")
        return Exa(api_key=api_key)
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize Exa client: {e}")
        return None

# ==============================================================================
# 2. PYDANTIC MODELS FOR DATA VALIDATION
# ==============================================================================

class SummaryRequest(BaseModel):
    """Configuration for LLM-generated summaries."""
    query: Optional[str] = Field(None)

class ContentsRequest(BaseModel):
    """Defines the format for the contents of the search results."""
    summary: Optional[SummaryRequest] = Field(None)

class ExaSearchRequest(BaseModel):
    """Pydantic model for making a request to the Exa Search API."""
    query: str
    numResults: Optional[int] = 10
    contents: Optional[ContentsRequest] = Field(default_factory=ContentsRequest)

class ResultWithContent(BaseModel):
    """A single search result with its content."""
    title: str
    url: str
    id: str
    publishedDate: Optional[str] = None
    author: Optional[str] = None
    summary: Optional[str] = None

class ExaSearchResponse(BaseModel):
    """Pydantic model for the response from the Exa Search API."""
    results: List[ResultWithContent]

# ==============================================================================
# 3. ROBUST TOOL DEFINITION
# ==============================================================================

def exa_search_api(exa_client: Exa, **kwargs) -> str:
    """
    Performs a web search using the Exa API, handling errors and filtering results.

    (The detailed, LLM-facing docstring is omitted here for script clarity but
    would be included in the actual agent tool definition.)
    """
    # 1. Validate input arguments against the Pydantic schema
    try:
        request = ExaSearchRequest(**kwargs)
    except ValidationError as e:
        return json.dumps({"error": "Invalid arguments provided", "details": str(e)})

    # 2. Prepare arguments for the Exa SDK, being careful with nested objects
    sdk_args = {"num_results": request.numResults}
    if request.contents and request.contents.summary and request.contents.summary.query:
        sdk_args["summary"] = {"query": request.contents.summary.query}
    else:
        # If no summary is requested, the tool can't get the info the LLM needs.
        return json.dumps({"error": "A summary query is required inside the 'contents' object to get useful results."})

    print(f"-> Performing Exa search for query: '{request.query}'")
    print(f"-> With SDK arguments: {sdk_args}")
    
    # 3. Make the API call with robust error handling
    try:
        response = exa_client.search_and_contents(request.query, **sdk_args)
    except Exception as e:
        return json.dumps({"error": "An error occurred during the Exa API call", "details": str(e)})

    # 4. Filter for successful results and handle empty summaries
    successful_results = []
    for result in response.results:
        # Only include the result if the API actually returned a summary
        if result.summary:
            successful_results.append({
                "title": result.title, "url": result.url, "id": result.id,
                "publishedDate": result.published_date, "author": result.author,
                "summary": result.summary,
            })

    if not successful_results:
        return json.dumps({"message": "Search was successful, but no web pages could be summarized due to paywalls or website restrictions."})

    # 5. Validate the final output and return as a JSON string
    return ExaSearchResponse(results=successful_results).model_dump_json(indent=2)

# ==============================================================================
# 4. MAIN EXECUTION LOGIC
# ==============================================================================

def main():
    """Main function to simulate the agent and run the tool."""
    exa_client = initialize_exa_client()
    if not exa_client:
        return # Exit if client initialization failed

    user_query = "What's the trending news about LLMs?"
    print(f"\n--- Simulating Agent for User Query: '{user_query}' ---\n")

    # This dictionary simulates the `args` an LLM would generate
    llm_generated_args = {
        "query": "trending news in large language models",
        "numResults": 5,
        "contents": {
            "summary": {
                "query": "Summarize the key points of the article regarding recent news or developments in LLMs."
            }
        }
    }
    
    print("--- Step 1: LLM decided to call the search tool with these arguments: ---")
    print(json.dumps(llm_generated_args, indent=2))
    print("\n--- Step 2: Invoking the exa_search_api function... ---\n")

    results_json = exa_search_api(exa_client, **llm_generated_args)

    print("\n--- Step 3: Received results from the tool: ---")
    
    # The final try/except block now cleanly handles both success and error JSON
    try:
        results_data = json.loads(results_json)
        print(json.dumps(results_data, indent=2))
        
        if "error" in results_data or "message" in results_data:
            print("\n--- Step 4: A real agent would use this message to respond to the user. ---")
        else:
            print("\n--- Step 4: A real agent would now summarize these results for the user. ---")
            
    except json.JSONDecodeError:
        # This case is less likely now since we always return JSON, but it's good practice
        print("FATAL ERROR: The tool returned a non-JSON response.")
        print(results_json)

if __name__ == '__main__':
    main()
