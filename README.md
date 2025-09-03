## Workforce Planning Modeling

Agentic, OpenAI-powered tools for workforce planning, use-case evaluation, and knowledge-base grounded Q&A. This repo includes a Streamlit chatbot UI, modular agents, and domain tools to evaluate Generative AI impact on G&A functions and to answer questions grounded in a curated knowledge base.

### Key Features
- **Streamlit Chatbot UI**: Simple chat interface to interact with agents
  - Select between `Basic_QA_Agent` and `GenAI_Use_Case_Agent`
  - Persistent chat history per session
- **Agents**:
  - **Basic_QA_Agent**: Answers only from a vector store with citations
  - **GenAI_Use_Case_Agent**: Evaluates GenAI appropriateness and cites knowledge base
  - (Preview) **Advanced_Q_A_Agent**: Scenario modeling and research workflow (requires extra tools; WIP)
- **Tools**:
  - `Agentic_Calculator_Tool`: Structured 5-dimension scoring rubric with typed output
  - `GenAI_Process_Knowledge_Base_Tool`: File search over a vector store of curated PDFs
- **Data & Notebooks**:
  - Curated PDFs under `Data/GenAI_Process_Knowledge_Base/`
  - Notebooks to create assistants and build knowledge bases

---

## Quickstart

### 1) Environment
Create a `.env` file in the project root with at least:

```dotenv
OPENAI_API_KEY=your_openai_api_key

# Model used by agents (examples: gpt-4o, gpt-4o-mini, o4-mini)
LLM_MODEL=gpt-4o-mini

# Vector store IDs (from OpenAI Assistants/Vector Store)
ASSISTANT_VECTOR_KEY=vs_XXXXXXXXXXXXXXXXXXXXXXXX
GENAI_PROCESS_KNOWLEDGE_BASE_ASSISTANT_KEY=vs_YYYYYYYYYYYYYYYYYYYYYYYY
```

Optional keys if you extend tools (not required for current UI):
- `TAVILY_API_KEY`, `EXA_API_KEY`, etc.

### 2) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Run the Streamlit app

```bash
streamlit run Basic_QA_Streamlit_Chatbot.py
```

Use the sidebar to select an agent and clear the conversation if needed.

---

## How it Works

### Streamlit UI
- Entrypoint: `Basic_QA_Streamlit_Chatbot.py`
- Uses `openai-agents` abstractions (`Runner`, `SQLiteSession`) to execute selected agent chains.
- Persists in-memory chat messages during the session; click "Clear conversation" to reset.

### Agents
- `src/Agents/Basic_QA_Agent.py`
  - Answers from a vector store only and provides citations
  - Requires `ASSISTANT_VECTOR_KEY` (OpenAI vector store id)
- `src/Agents/GenAI_Use_Case_Agent.py`
  - Systematically evaluates GenAI use-case appropriateness
  - Uses `Agentic_Calculator_Tool` and `GenAI_Process_Knowledge_Base_Tool`
  - Requires `GENAI_PROCESS_KNOWLEDGE_BASE_ASSISTANT_KEY`
- `src/Agents/Advanced_Q_A_Agent.py` (preview/WIP)
  - Rich scenario-planning prompt and multi-tool workflow
  - References additional tools not included in this repo (e.g., SEC, web search, critic)
  - Treat as a template; complete the missing tools before using

### Tools
- `src/Tools/Agentic_Calculator_Tool.py`
  - Rubric-driven evaluator with 5 independent dimensions (1–5)
  - Returns typed output (score, reasoning, hallucination_score)
  - Uses `LLM_MODEL`
- `src/Tools/GenAI_Process_Knowledge_Base_Tool.py`
  - File search over an OpenAI vector store
  - Uses `GENAI_PROCESS_KNOWLEDGE_BASE_ASSISTANT_KEY`

---

## Project Structure

```text
Workforce_Model/
  Basic_QA_Streamlit_Chatbot.py      # Streamlit UI
  requirements.txt                   # Python dependencies
  Notebooks/
    01. Create Generic Assistant.ipynb
    02. Create Process Knowledge Base.ipynb
  src/
    Agents/
      Basic_QA_Agent.py
      GenAI_Use_Case_Agent.py
      Advanced_Q_A_Agent.py          # Preview (requires extra tools)
    Tools/
      Agentic_Calculator_Tool.py
      GenAI_Process_Knowledge_Base_Tool.py
  Data/
    GenAI_Process_Knowledge_Base/    # Curated PDFs
    Knowledge_Base/                  # Intermediate/Raw artifacts
    Intermediate/, Raw/, Results/
```

### Data
- `Data/GenAI_Process_Knowledge_Base/`: PDFs used to build the knowledge base.
- `Data/Knowledge_Base/Intermediate/`: JSON artifacts from processing pipelines.
- Other folders (`Intermediate/`, `Raw/`, `Results/`) are reserved for your ETL/experiments.

---

## Creating/Updating Vector Stores

The agents search over vector stores via `openai-agents` `FileSearchTool`. You need two vector store ids:
- `ASSISTANT_VECTOR_KEY` for `Basic_QA_Agent`
- `GENAI_PROCESS_KNOWLEDGE_BASE_ASSISTANT_KEY` for `GenAI_Process_Knowledge_Base_Tool`

You can create vector stores and upload documents using the OpenAI Assistants API or UI. The provided notebooks can be adapted to automate ingest from `Data/GenAI_Process_Knowledge_Base/`.

High-level steps:
1. Create a vector store in OpenAI (Assistants UI or API)
2. Upload PDFs from `Data/GenAI_Process_Knowledge_Base/`
3. Note the vector store id and place it in `.env`

---

## Notebooks
- `01. Create Generic Assistant.ipynb`: Template to create/generalize an assistant and test toolchains.
- `02. Create Process Knowledge Base.ipynb`: Outline for building a process knowledge base and populating a vector store.

---

## Requirements
See `requirements.txt`. Key libraries:
- `openai`, `openai-agents` for agent execution and file search
- `streamlit` for the chat UI
- `pandas`, `numpy`, `plotly`, `matplotlib`, `seaborn` for analysis/visualization (optional for notebooks)

---

## Troubleshooting
- **Module `agents` or `openai-agents` not found**: Ensure `pip install -r requirements.txt` ran in your active virtualenv.
- **Invalid vector store id**: Verify `ASSISTANT_VECTOR_KEY` and `GENAI_PROCESS_KNOWLEDGE_BASE_ASSISTANT_KEY` are correct and that your OpenAI API key has access.
- **No citations returned**: Confirm your vector store contains documents and the agent is configured to include search results.
- **Streamlit doesn’t start**: Check Python version (3.9+ recommended), virtualenv activation, and that port 8501 is free.
- **macOS gatekeeper/SSL issues**: Ensure certificates are installed (e.g., `Install Certificates.command` for Apple Python) or use a Homebrew Python.

---

## Roadmap
- Finish integrating the advanced Q&A/research agent and missing tools
- Add automated pipeline to build/update vector stores from `Data/`
- Provide Dockerfile and devcontainer for consistent setup
- Expand evaluation metrics and add unit tests for tools

---

## Contributing
Issues and PRs are welcome. For larger changes, please open an issue first to discuss the approach.

## License
No license file is included. If you plan to open-source, add an appropriate `LICENSE` file.
