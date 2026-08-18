# IntentFlow-AI ⚡

An intelligent, multi-turn conversational agent built with **FastAPI** and **Groq Cloud**. The agent dynamically evaluates user intent, resolves conversational ambiguity across session history, orchestrates concurrent web scraping when real-time information is needed, and synthesizes structured JSON responses.

---

## 🚀 Key Features

* **Multi-Turn Intent & Ambiguity Resolution:** Distinguishes between vague entities (triggering targeted clarifying questions) and concrete queries without making blind assumptions.
* **Conditional Real-Time Web Retrieval:** Automatically detects time-sensitive or factual queries, searches candidate links, and concurrently scrapes clean web content using multi-threaded `httpx` + `BeautifulSoup`.
* **Bounded Latency & Fail-Safe Fallbacks:** Strict per-site timeouts and automatic search snippet fallbacks ensure queries never hang on unresponsive domains.
* **Strict Structured JSON Outputs:** Leverages Groq's low-latency inference with enforced JSON schemas for intent classification and response payloads.
* **Session-Aware Context Management:** Maintains conversation history per session to resolve pronouns and follow-up constraints.

---

## 🛠️ Architecture & Workflow

1. **Step 1 — Intent Evaluation:** Evaluates user query and session history against JSON schemas to check for ambiguity and real-time retrieval requirements.
2. **Step 2 — Concurrent Scraper:** Dispatches parallel HTTP requests via `ThreadPoolExecutor`, parses DOM trees, strips non-content tags, and caps context length.
3. **Step 3 — Multi-Source Synthesis:** Synthesizes scraped evidence into a coherent response alongside proactive clarifying and suggested questions.

---

## 📦 Tech Stack

* **Framework:** FastAPI, Uvicorn, Pydantic
* **LLM Engine:** Groq Cloud API
* **Web Discovery & Scraping:** `ddgs`, `httpx`, `BeautifulSoup4`
* **Concurrency:** `concurrent.futures.ThreadPoolExecutor`

---

## ⚙️ Quickstart

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/intentflow-ai.git](https://github.com/your-username/intentflow-ai.git)
   cd intentflow-ai
