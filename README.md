# LangGraph Interview Agent

A multi-agent system for conducting technical interviews with automatic review and feedback loops.

## Overview

This project demonstrates a multi-agent interview system where two specialized agents work together:
- **Interview Agent**: Conducts the interview and asks questions
- **Review Agent**: Evaluates the candidate's responses and can request additional information

The Review Agent can send feedback back to the Interview Agent when it needs more information, creating a dynamic and adaptive interview process.

## Architecture

```
User → Interview Agent ↔ Review Agent → End
                    ↑         ↓
                    └─────────┘
                  (feedback loop)
```

The system also includes:
- Memory storage for tracking candidate information
- A job requirements retrieval system
- Loop guards to prevent infinite feedback cycles

## Installation

```bash
git clone https://github.com/your-username/langgraph-interview-agent.git
cd langgraph-interview-agent
pip install -r requirements.txt
```

## Usage

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

Run the demo:

```bash
python run_demo.py
```

## Project Structure

```
.
├── agent_core.py      # Core agent logic and tools
├── run_demo.py        # CLI demo interface
├── requirements.txt   # Dependencies
└── README.md
```

## How It Works

1. The Interview Agent asks questions and gathers information from the user
2. When ready, it transfers control to the Review Agent
3. The Review Agent evaluates the collected information
4. If more information is needed, control returns to the Interview Agent with specific feedback
5. This continues until the Review Agent has enough information to make a decision

The system uses LangGraph's `Command` pattern for agent coordination and includes a loop counter to prevent infinite cycles.

## License

MIT
