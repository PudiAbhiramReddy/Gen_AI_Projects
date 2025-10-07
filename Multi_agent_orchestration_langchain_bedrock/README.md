# Multi-Agent City Intelligence (LangGraph + Mistral on Amazon Bedrock)

A production-style personal project where I designed and implemented a multi-agent system that plans, routes, and synthesizes information about a city: events, weather, activities, dining, and outfit suggestions. The system uses LangGraph for orchestration, Mistral models via Amazon Bedrock for reasoning and tool-use, FAISS for vector search, and external tools/APIs (SQLite, Tavily, OpenWeatherMap) for retrieval.

## Highlights
- End-to-end multi-agent workflow: Events → Web Search (fallback) → Weather → Restaurant RAG → Final Analysis
- Tool-use capable LLM: Mistral (via Bedrock Converse) invokes tools and consumes results iteratively
- RAG for restaurants: FAISS vector store built from a curated/synthetic dataset
- Deterministic routing: Conditional edges choose when to search web vs. use local data
- Reproducible notebook: Single `ipynb` with clear sections and visualized graph

## Architecture
- LangGraph Orchestration
  - Nodes: `Events Database Agent`, `Online Search Agent`, `Weather Agent`, `Restaurants Recommendation Agent`, `Analysis Agent`
  - Conditional routing from Events → Search or Weather
  - State persisted with `MemorySaver`
- LLM: Mistral via Amazon Bedrock Converse API
- Tools
  - `events_database_tool`: SQLite local events
  - `search_tool`: Tavily web search
  - `weather_tool`: OpenWeatherMap wrapper
  - `query_restaurants_RAG`: FAISS similarity search over restaurant corpus
- Data
  - Local events in SQLite (seeded from JSON)
  - Restaurant dataset (pre-built JSON; optional synthetic generation)

## Tech Stack
- Python, Jupyter Notebook
- Amazon Bedrock (Mistral), LangGraph, LangChain
- FAISS, SQLite, Pandas
- Tavily API, OpenWeatherMap API

## What this project demonstrates
- Agentic design: breaking down a user goal into specialized agents
- LLM tool-use: invoking code paths and grounding outputs with retrieved data
- Retrieval Augmented Generation: domain-specific RAG for restaurant insights
- Robustness: fallback to web search when local data is missing
- Visualization: Mermaid graph rendering of the agent workflow

## Getting Started
1. Clone the repo
```bash
git clone https://github.com/<your-username>/Multi_agent_orchestration_langcjain_bedrock.git
cd Multi_agent_orchestration_langcjain_bedrock
```

2. Open the notebook
- File: `Multi_Agent_LangGraph_Mistral.ipynb`
- Kernel: Python 3.x with internet access and AWS credentials configured

3. Configure environment variables
- `AWS_REGION` (e.g., `us-east-1`)
- `TAVILY_API_KEY`
- `OPENWEATHERMAP_API_KEY`

4. Install dependencies
- The first notebook cell installs required packages (`boto3`, `langgraph`, `langchain`, `faiss-cpu`, etc.)

## Usage
- Run the notebook top-to-bottom.
- The graph compiles and renders.
- Execute the main section for sample cities (e.g., Tampa, Philadelphia, New York).
- The system prints intermediate agent outputs and a final consolidated analysis.

## Project Structure
- `Multi_Agent_LangGraph_Mistral.ipynb` — full, runnable implementation
- `data/` — JSON datasets (events, restaurants)
- `local_info.db` — generated SQLite DB for events (runtime)

## Results (Examples)
- Handles both scenarios: cities with local events and cities requiring web search
- Produces weather-aware activity and outfit suggestions
- Returns top restaurant insights via RAG when data is available

## Notes & Trade-offs
- Restaurant dataset can be generated synthetically (time-consuming) or loaded from `data/restaurant_data.json`.
- For production, consider Bedrock Knowledge Bases instead of ad-hoc FAISS for larger corpora.
- API quotas/rate limits apply for Tavily and OpenWeatherMap.

## Roadmap
- Add caching layer for API/tool calls
- Expand dataset coverage and add evaluation harness
- Containerize (Docker) and add CI for automated checks
- Optional: Streamed UI (Gradio/Streamlit) on top of the graph

## License
This project is provided under the MIT License. See `LICENSE` if present, or add one for public release.


