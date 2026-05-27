# Multi-Agent Pipeline Demo

A minimal multi-agent system built with LangGraph demonstrating agent orchestration, shared state, and tool calling.

## Architecture

Three specialized agents communicate through shared state:
```
Planner → Executor → Reporter → END
```

- **Planner** — decomposes a task into concrete steps
- **Executor** — executes each step using web search (DuckDuckGo) via tool calls
- **Reporter** — synthesizes results into a final report

## Key Concepts

- **Shared state** — agents communicate through a common state object, not directly
- **Tool calling** — Executor uses real web search instead of hallucinating answers
- **Agentic loop** — LLM decides when to call a tool, receives results, formulates response based on real data

## Stack

- LangGraph
- LangChain
- OpenAI GPT-4o-mini
- DuckDuckGo Search

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install langgraph langchain-openai langchain-community duckduckgo-search python-dotenv
```

Create `.env`:
```
OPENAI_API_KEY=sk-...
```

Run:
```bash
python main.py
```

## Example

Input task: `"Automate employee onboarding in a company"`

The pipeline will:
1. Break it into 3 actionable steps
2. Search the web for real information on each step
3. Generate a structured report based on actual findings
