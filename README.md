# AgentDiff

**Visual Diff & Replay Debugger for AI Agents**

AgentDiff is an open-source tool for capturing, comparing, and replaying AI agent executions. Think of it as "git diff" for AI agents - track how your agent behaves across different versions, prompts, and models.

## Features

- **🔍 Trace Capture**: Automatically capture agent executions including LLM calls, tool usage, and decision points
- **📊 Visual Diff**: Compare agent behavior between versions with structural and semantic diffs
- **⏪ Time-Travel Replay**: Replay executions from any checkpoint with the ability to modify inputs
- **💰 Cost Tracking**: Track and compare costs per decision node with detailed breakdowns
- **🗄️ Multiple Backends**: Support for SQLite (default), Oracle Autonomous Database, and PostgreSQL
- **🔌 Framework Integrations**: Works with OpenAI, Anthropic, LangChain, CrewAI, and more

## Installation

```bash
# Basic installation
pip install agentdiff

# With Oracle Autonomous Database support
pip install agentdiff[oracle]

# With semantic diff support
pip install agentdiff[semantic]

# With web UI
pip install agentdiff[api]

# All features
pip install agentdiff[all]
```

## Quick Start

### 1. Trace an Agent

```python
import agentdiff
from agentdiff.storage import SQLiteStore

# Configure storage
storage = SQLiteStore("my_traces.db")
tracer = agentdiff.configure(storage)

# Trace your agent
@tracer.trace(name="my-agent", version="1.0.0")
async def my_agent(query: str):
    with tracer.span("process_query"):
        # Your agent logic here
        result = await call_llm(query)
        tracer.log_llm_call(
            model="gpt-4",
            provider="openai",
            input_messages=[{"role": "user", "content": query}],
            output_message=result,
            token_usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
    return result
```

### 2. Auto-Capture with Hooks

```python
from agentdiff.capture import patch_openai, patch_anthropic

# Automatically capture all OpenAI/Anthropic calls
patch_openai()
patch_anthropic()

# Now all LLM calls are automatically traced
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(...)  # Automatically traced!
```

### 3. Compare Traces

```bash
# CLI
agentdiff diff <trace_id_a> <trace_id_b>

# Or in Python
from agentdiff.diff import StructuralDiffEngine

engine = StructuralDiffEngine()
diff = engine.diff(trace_a, trace_b)

print(f"Structural similarity: {diff.structural_similarity:.1%}")
print(f"Cost delta: ${diff.cost_delta:+.4f}")
```

### 4. Replay from Checkpoint

```python
from agentdiff.replay import ReplayEngine, ReplayMode

engine = ReplayEngine(storage, mode=ReplayMode.HYBRID)

# Replay with modified input
result = await engine.replay_from_span(
    trace_id="...",
    span_id="...",
    modified_input={"query": "new question"}
)

print(f"Diverged: {result.diverged}")
print(f"Divergence point: {result.divergence_point}")
```

## CLI Usage

```bash
# List traces
agentdiff list

# Show trace details
agentdiff show <trace_id>

# Compare traces
agentdiff diff <trace_id_a> <trace_id_b> --format html --output diff.html

# Analyze costs
agentdiff cost <trace_id>

# Replay a trace
agentdiff replay <trace_id> --from-span <span_id>

# Start web UI
agentdiff serve --port 8000
```

## Oracle Autonomous Database Setup

AgentDiff supports Oracle Autonomous Database for enterprise deployments:

```python
from agentdiff.storage import create_oracle_store

# Using wallet authentication
store = create_oracle_store(wallet_path="/path/to/wallet")

# Or using environment variables
# export ORACLE_USER=admin
# export ORACLE_PASSWORD=your_password
# export ORACLE_DSN=your_adb_tns_name
store = create_oracle_store()
```

See [docs/oracle-setup.md](docs/oracle-setup.md) for detailed setup instructions.

## Framework Integrations

### LangChain

```python
from agentdiff.capture import LangChainCallback
from langchain.chat_models import ChatOpenAI

callback = LangChainCallback()
llm = ChatOpenAI(callbacks=[callback])

# All LangChain operations are now traced
chain.invoke(input, config={"callbacks": [callback]})
```

### CrewAI

```python
from agentdiff.capture import CrewAICallback
from crewai import Crew

callback = CrewAICallback()
crew = Crew(agents=[...], tasks=[...], callbacks=[callback])
```

## API Reference

See [docs/api-reference.md](docs/api-reference.md) for the complete API documentation.

## Development

```bash
# Clone the repository
git clone https://github.com/agentdiff/agentdiff.git
cd agentdiff

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests
ruff check src tests
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.
