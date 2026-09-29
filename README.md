# OBRIO Support Automation (AI Ops)

This repository contains a technical MVP designed to optimize and automate the customer support workflow for the Nebula product. 

The project consists of two main components:
1. **AI Ticket Classifier:** An automated routing system that analyzes incoming support requests, categorizes them, sets priority levels, and suggests the next recommended action.
2. **Internal AI Assistant:** A tool for support agents that generates ticket summaries, provides response options in various tones (formal, empathetic, short), and retrieves relevant information from the knowledge base.

**Key Features Focus:** LLM integration, prompt engineering, edge-case mitigation, and system architecture design.


---


## Installation & Setup

This project requires Python 3.10 or higher.

1. **Clone the repository and enter the directory:**
   ```bash
   git clone <repository_url>
   cd obrio-aiops-assignment
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Copy the `.env.example` file to `.env` and insert your OpenAI API key:
   ```bash
   cp .env.example .env
   # Open .env and add: OPENAI_API_KEY="your-key-here"
   ```


---


## Usage (Task 2 - CLI Assistant)

The internal support assistant is accessible via a command-line interface. 

You can pass a customer ticket directly as an argument:
```bash
python -m src.assistant "I was charged twice this month for my subscription. I want a refund."
```

Or read from `stdin` if no argument is provided. To get a raw, parseable JSON output instead of the human-readable format, append the `--json` flag:
```bash
python -m src.assistant "How do I reset my password?" --json
```

## Running Tests & Evaluation

The project includes unit tests covering retrieval accuracy and resilience mechanisms (retries/fallbacks).

**To run the test suite:**
```bash
python -m pytest
```

**To run the LLM evaluation benchmark (Latency vs Cost):**
```bash
python eval/run_eval.py
```