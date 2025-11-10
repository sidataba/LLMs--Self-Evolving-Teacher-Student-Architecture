# Using Real LLM APIs

This guide shows how to use the system with real LLM APIs (OpenAI, Anthropic) instead of mock models.

## 🔑 Setup API Keys

### Option 1: Environment Variables

```bash
export OPENAI_API_KEY=sk-proj-abc123...
export ANTHROPIC_API_KEY=sk-ant-abc123...
```

### Option 2: .env File (Recommended)

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=sk-proj-abc123...
   ANTHROPIC_API_KEY=sk-ant-abc123...
   ```

### Getting API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/settings/keys

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `openai>=1.0.0` - OpenAI API client
- `anthropic>=0.18.0` - Anthropic API client
- `tiktoken>=0.5.0` - Accurate token counting for OpenAI

## 🚀 Quick Start

### Simple Usage

```python
from src.models.openai_model import create_gpt4_model

# Create model
model = create_gpt4_model(model_id="my-gpt4", domain="general")

# Generate response
response = model.generate_response("What is machine learning?")

print(response.response_text)
print(f"Cost: ${response.metadata['cost_usd']:.6f}")
```

### Using Anthropic Claude

```python
from src.models.anthropic_model import create_claude_sonnet_model

# Create model
model = create_claude_sonnet_model(model_id="my-claude", domain="science")

# Generate response
response = model.generate_response("Explain photosynthesis")

print(response.response_text)
```

## 🏗️ Architecture Setup

### Cost-Optimized Configuration

Use different model tiers for different roles:

```python
from src.models.openai_model import create_gpt4_model, create_gpt35_turbo_model
from src.models.anthropic_model import create_claude_haiku_model
from src.core.orchestrator import Orchestrator

orchestrator = Orchestrator()

# Supervisor: Highest quality (GPT-4)
supervisor = create_gpt4_model("supervisor", domain=None)
supervisor.role = supervisor.role.SUPERVISOR

# Teachers: Mid-tier (GPT-3.5-turbo)
teacher = create_gpt35_turbo_model("teacher-math", domain="mathematics")
teacher.role = teacher.role.TEACHER

# Students: Cheapest (Claude Haiku)
student = create_claude_haiku_model("student-math", domain="mathematics")
student.role = student.role.STUDENT
student.teacher_id = "teacher-math"

# Add to orchestrator
orchestrator.models = {
    "supervisor": supervisor,
    "teacher-math": teacher,
    "student-math": student,
}
orchestrator.supervisor = supervisor

# Process query
result = orchestrator.process_query("What is calculus?", domain="mathematics")
```

## 💰 Cost Tracking

All real LLM models automatically track costs:

```python
# Get cost statistics
stats = model.get_cost_statistics()

print(f"Total queries: {stats['query_count']}")
print(f"Total tokens: {stats['total_tokens']:,}")
print(f"Total cost: ${stats['total_cost_usd']:.6f}")
print(f"Avg cost/query: ${stats['avg_cost_per_query']:.6f}")
```

### Current Pricing (as of 2024)

| Model | Input ($/1K tokens) | Output ($/1K tokens) |
|-------|--------------------:|---------------------:|
| GPT-4 | $0.03 | $0.06 |
| GPT-4 Turbo | $0.01 | $0.03 |
| GPT-3.5-turbo | $0.0005 | $0.0015 |
| Claude 3 Opus | $0.015 | $0.075 |
| Claude 3 Sonnet | $0.003 | $0.015 |
| Claude 3 Haiku | $0.00025 | $0.00125 |

## 📊 Run Demo

```bash
# Make sure API keys are set
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Run demo
python examples/real_llm_demo.py
```

**Warning**: This makes real API calls and will incur costs (~$0.50-$1.00).

## ⚙️ Configuration

Use the provided configuration for real LLMs:

```bash
# Copy real LLM config
cp config/real_llm_config.yaml config/my_config.yaml

# Edit as needed
vim config/my_config.yaml

# Run with config
python examples/enterprise_demo.py --config=config/my_config.yaml
```

## 🎯 Model Selection Guide

### When to use GPT-4
- Highest quality requirements
- Complex reasoning tasks
- Code generation
- Critical decisions

### When to use GPT-3.5-turbo
- Balanced quality/cost
- General Q&A
- Content generation
- Most production workloads

### When to use Claude Opus
- Maximum quality needed
- Long context (200K tokens)
- Complex analysis
- Alternative to GPT-4

### When to use Claude Sonnet
- Balanced quality/cost
- Fast responses needed
- Good alternative to GPT-3.5-turbo
- Production workloads

### When to use Claude Haiku
- Cost optimization
- Simple queries
- High throughput needed
- Student models in architecture

## 🔒 Best Practices

### 1. API Key Security

```python
# ✅ Good - Use environment variables
import os
api_key = os.getenv("OPENAI_API_KEY")

# ❌ Bad - Hardcode keys
api_key = "sk-abc123..."  # Never do this!
```

### 2. Error Handling

```python
try:
    response = model.generate_response(query)
except Exception as e:
    logger.error(f"API call failed: {e}")
    # Fallback to cheaper model or cached response
```

### 3. Rate Limiting

The system automatically handles rate limiting with:
- Minimum 0.1s between requests
- Exponential backoff on failures
- Automatic retries (up to 3 attempts)

### 4. Cost Control

```python
# Set budget limits
orchestrator.config['cost_tracking'] = {
    'enabled': True,
    'alert_threshold_usd': 100.0,  # Alert at $100/day
    'budget_limit_usd': 1000.0,    # Stop at $1000/month
}
```

## 🧪 Testing Without Costs

Use mock models for testing:

```python
from src.models.student import StudentModel

# Mock model - no API calls, no costs
student = StudentModel("test-student", domain="math")

# Still works like real models
response = student.generate_response("What is 2+2?")
```

## 📈 Production Deployment

For production with real LLMs:

1. **Set up monitoring**:
   ```python
   orchestrator.enable_monitoring()
   ```

2. **Configure cost alerts**:
   ```python
   orchestrator.set_cost_alert(threshold=100.0)
   ```

3. **Use caching**:
   ```python
   orchestrator.enable_response_cache(ttl=3600)
   ```

4. **Implement fallbacks**:
   ```python
   # If GPT-4 fails, fallback to GPT-3.5
   orchestrator.set_fallback_chain([
       "gpt-4",
       "gpt-3.5-turbo",
       "mock-model"
   ])
   ```

## 🔍 Debugging

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Now see all API calls
response = model.generate_response("test")
```

Output:
```
DEBUG:src.models.openai_model:Calling OpenAI API...
DEBUG:src.models.openai_model:Request cost: $0.000450 (15 in, 23 out)
INFO:src.models.openai_model:Response generated successfully
```

## 📚 Examples

See `examples/real_llm_demo.py` for:
- Simple queries
- Model comparison
- Teacher-student architecture
- Cost optimization

## ❓ Troubleshooting

### Error: "No API key found"
- Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variable
- Or add to `.env` file

### Error: "Insufficient credits"
- Add credits to your OpenAI/Anthropic account
- Check billing settings

### Error: "Rate limit exceeded"
- Wait a few seconds and retry
- System automatically retries with backoff
- Consider upgrading API tier

### Error: "Model not found"
- Check model name (e.g., `gpt-4`, not `gpt4`)
- Ensure you have access to the model
- Some models require special access

## 🎓 Next Steps

1. Try the demo: `python examples/real_llm_demo.py`
2. Read the paper: `paper/research_paper.md`
3. Run experiments: `python paper/experiments/validate_theorems.py`
4. Deploy to production: See `ENTERPRISE_GUIDE.md`

## 💡 Tips

- Start with GPT-3.5-turbo for development
- Use Claude Haiku for students (cheapest)
- Monitor costs closely in production
- Cache responses when possible
- Use mock models for testing

---

**Cost Savings**: The self-evolving architecture achieves **67% cost reduction** by intelligently routing queries to cheaper models while maintaining quality through continuous learning and promotion.
