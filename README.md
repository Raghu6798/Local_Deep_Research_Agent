# Autonomous Research Agent with Local LLM and Web Crawling

This project showcases a powerful autonomous research agent built with LangChain, leveraging a locally hosted Large Language Model (LLM) for reasoning and two powerful tools for information retrieval: Exa for targeted URL searches and an asynchronous web crawler for in-depth content extraction.

## 🌟 Features

- **Local LLM Integration**: Utilizes a local, OpenAI-compatible server (like `llama.cpp`) to run powerful language models such as Qwen2.5-7B-Instruct, ensuring privacy and control over the AI's reasoning capabilities.
- **Advanced Web Search**: Integrates with the Exa Search API to find relevant URLs based on a user's query.
- **Concurrent Web Crawling**: Employs a sophisticated, multi-threaded web crawler (`crawl4ai`) to efficiently extract clean, markdown-formatted content from multiple websites simultaneously.
- **Intelligent Agent Workflow**: Implements a ReAct (Reasoning and Acting) agent using LangGraph that follows a structured workflow:
    1.  **Search**: Uses Exa to find relevant web pages.
    2.  **Crawl**: Concurrently crawls the top search results to gather information.
    3.  **Synthesize**: Analyzes the crawled content to provide a comprehensive and consolidated answer to the user's initial query.
- **Asynchronous & Thread-Safe**: The crawling logic is built with `asyncio` and `concurrent.futures` to handle multiple network requests efficiently and safely.
- **Customizable**: Easily configurable to use different local models, adjust crawling parameters, and expand with more tools.

## 🚀 Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

-   Python 3.8+
-   A C++ compiler for building `llama.cpp` (e.g., GCC, Clang, or MSVC).
-   `git` for cloning the repositories.

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a Virtual Environment**:
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python Dependencies**:
    Create a `requirements.txt` file with the following content:

    ```
    langchain
    langgraph
    pydantic
    python-dotenv
    requests
    exa_py
    crawl4ai
    playwright
    ```

    Then, install the packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Playwright Browsers and crawl4AI setup**:
    The `crawl4ai` library uses Playwright to render web pages. You need to install the necessary browser binaries.
    ```bash
    playwright install
    crawl4ai-setup
    ```

5.  **Set Up Environment Variables**:
    Create a file named `.env` in the root of the project directory and add your Exa API key:
    ```
    EXA_SEARCH_API="your_exa_api_key_here"
    ```

## 🛠️ Tool Configuration

### 1. Local LLM Server (`llama.cpp`)

This project is designed to connect to a local LLM server that is compatible with the OpenAI API format. We recommend using `llama.cpp`.
Build llama.cpp using CMake:

```bash
cmake -B build
cmake --build build --config Release
```
1.  **Clone and Build `llama.cpp`**:
    ```bash
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    # Build the project (on Linux/macOS)
    make
    # For Windows or other platforms, follow the build instructions in the llama.cpp repository.
    ```

2.  **Download a GGUF Model**:
    You need a model in the GGUF format. You can download one from Hugging Face. For this project, `Qwen2.5-7B-Instruct-q8_0.gguf` is a good choice.

3.  **Start the Server**:
    From within the `llama.cpp` directory, run the following command to start the server. This command downloads the model directly from Hugging Face if you don't have it locally.

    ```bash
    ./llama-server --jinja -fa -hf "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M" --chat-template-file "\llama.cpp\models\templates\Qwen-Qwen2.5-7B-Instruct.jinja" --port 8082
    ```
    *   The script is configured to connect to `http://127.0.0.1:8082`. If you use a different port, make sure to update the `base_url` in the `ChatQwen` class in the Python script.

### 2. Exa Search API

1.  **Get an API Key**:
    -   Sign up for a free account at [exa.ai](https://exa.ai).
    -   Navigate to your dashboard to find your API key.

2.  **Configure the API Key**:
    -   As mentioned in the installation steps, paste your API key into the `.env` file in the root of your project directory.

## Usage

Once you have completed the setup and configuration, you can run the main agent script.

1.  **Ensure your local LLM server is running.**
2.  **Run the Python script from your terminal**:
    ```bash
    python test.py
    ```

The script will then execute the agent with the predefined query: `"What's the trending news in LLMs?"`.

### Expected Output

You will see a series of log messages in your console as the agent works through its process:

1.  The agent will first call the `exa_search_for_urls` tool to get a list of relevant URLs.
2.  Next, it will make parallel calls to the `crawl_website_for_content` tool for the top URLs. You will see "[CRAWL-THREAD]" messages indicating the progress of each crawl.
3.  Finally, after the stream of agent actions is complete, the script will print the synthesized final answer to the console, which is the agent's comprehensive response to the initial query.

```
 > Starting concurrent crawl for 3 URLs with 2 threads...
   [CRAWL-THREAD] Starting crawl for: <URL_1>
   [CRAWL-THREAD] Starting crawl for: <URL_2>
   [CRAWL-THREAD] SUCCESS for: <URL_1>
   [CRAWL-THREAD] Starting crawl for: <URL_3>
   [CRAWL-THREAD] SUCCESS for: <URL_2>
   [CRAWL-THREAD] FAILED for: <URL_3> - Some error
==================================================
Final Answer:
==================================================
Based on the latest information, the trending news in Large Language Models (LLMs) includes... [synthesized answer from crawled content]

```

