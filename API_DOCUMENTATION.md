# API Documentation

Complete API reference for the Self-Evolving Teacher-Student Architecture.

## Table of Contents

1. [Core Components](#core-components)
2. [Model APIs](#model-apis)
3. [Orchestrator API](#orchestrator-api)
4. [Benchmarking APIs](#benchmarking-apis)
5. [Human Evaluation APIs](#human-evaluation-apis)
6. [Distributed System APIs](#distributed-system-apis)
7. [Configuration](#configuration)
8. [Examples](#examples)

---

## Core Components

### Orchestrator

Main coordinator for the entire system.

```python
from src.core.orchestrator import Orchestrator

# Initialize
orchestrator = Orchestrator(config_path="./config/default_config.yaml")

# Process query
result = orchestrator.process_query(
    query_text="What is machine learning?",
    domain="computer_science"  # Optional
)
```

#### `Orchestrator.process_query()`

**Parameters:**
- `query_text` (str): The user's query
- `domain` (str, optional): Domain hint for routing

**Returns** (dict):
```python
{
    'final_response': str,  # Best response
    'winner_model': str,  # Model ID that won
    'winner_score': float,  # Confidence score
    'routing_strategy': str,  # 'targeted', 'parallel', or 'hybrid'
    'candidate_responses': List[dict],  # All responses
    'domain': str,  # Detected domain
}
```

---

## Model APIs

### Base Model

All models inherit from `BaseModel`.

```python
from src.models.base import BaseModel, ModelRole

class MyModel(BaseModel):
    def generate_response(self, query_text: str, context: dict = None):
        # Implementation
        pass
```

### Real LLM Models

#### OpenAI Models

```python
from src.models.openai_model import (
    create_gpt4_model,
    create_gpt35_turbo_model,
    OpenAIModel
)

# GPT-4
model = create_gpt4_model(
    model_id="my-gpt4",
    domain="mathematics",
    api_key="sk-..."  # Or set OPENAI_API_KEY env var
)

# Custom OpenAI model
model = OpenAIModel(
    model_id="custom",
    model_name="gpt-4-turbo-preview",
    domain="science",
    base_confidence=0.92,
    max_tokens=2048,
    temperature=0.7
)

# Generate response
response = model.generate_response("What is calculus?")
```

**OpenAIModel Parameters:**
- `model_id` (str): Unique identifier
- `model_name` (str): OpenAI model name (gpt-4, gpt-3.5-turbo, etc.)
- `domain` (str, optional): Domain specialization
- `base_confidence` (float): Base confidence score (0-1)
- `api_key` (str, optional): API key
- `max_tokens` (int): Maximum response tokens
- `temperature` (float): Sampling temperature (0-2)

**Response Object:**
```python
@dataclass
class ModelResponse:
    model_id: str
    response_text: str
    confidence: float
    reasoning: str
    metadata: dict  # Contains: model, input_tokens, output_tokens, cost_usd
```

#### Anthropic Models

```python
from src.models.anthropic_model import (
    create_claude_opus_model,
    create_claude_sonnet_model,
    create_claude_haiku_model,
    AnthropicModel
)

# Claude Sonnet
model = create_claude_sonnet_model(
    model_id="my-claude",
    domain="science",
    api_key="sk-ant-..."  # Or set ANTHROPIC_API_KEY env var
)

# Custom Anthropic model
model = AnthropicModel(
    model_id="custom",
    model_name="claude-3-opus-20240229",
    domain="philosophy",
    base_confidence=0.94,
    max_tokens=1024,
    temperature=0.7
)
```

**AnthropicModel Parameters:** (same as OpenAI but temperature is 0-1)

### Cost Tracking

All real LLM models track costs automatically:

```python
# Get cost statistics
stats = model.get_cost_statistics()
# Returns:
# {
#     'total_input_tokens': int,
#     'total_output_tokens': int,
#     'total_tokens': int,
#     'total_cost_usd': float,
#     'avg_cost_per_query': float,
#     'cost_per_1k_input': float,
#     'cost_per_1k_output': float,
# }
```

---

## Orchestrator API

### Adding Models

```python
from src.models.openai_model import create_gpt4_model

orchestrator = Orchestrator()

# Add supervisor
supervisor = create_gpt4_model("supervisor")
supervisor.role = supervisor.role.SUPERVISOR
orchestrator.models["supervisor"] = supervisor
orchestrator.supervisor = supervisor

# Add teacher
teacher = create_gpt35_turbo_model("teacher-math", domain="mathematics")
teacher.role = teacher.role.TEACHER
orchestrator.models["teacher-math"] = teacher

# Add student
student = StudentModel("student-math", domain="mathematics")
student.teacher_id = "teacher-math"
orchestrator.models["student-math"] = student
```

### Query Processing

```python
# Simple query
result = orchestrator.process_query("What is 2+2?")

# Query with domain hint
result = orchestrator.process_query(
    "Explain derivatives",
    domain="mathematics"
)

# Access results
print(result['final_response'])
print(f"Confidence: {result['winner_score']:.3f}")
print(f"Routed to: {result['winner_model']}")
```

### Statistics

```python
# Get system statistics
stats = orchestrator.get_statistics()
# Returns:
# {
#     'total_queries': int,
#     'model_count': int,
#     'promotions': int,
#     'avg_response_time': float,
#     'routing_distribution': dict,
# }
```

---

## Benchmarking APIs

### Running Benchmarks

```python
from experiments.benchmarks.mmlu_benchmark import run_mmlu_benchmark
from experiments.benchmarks.truthfulqa_benchmark import run_truthfulqa_benchmark
from experiments.benchmarks.gsm8k_benchmark import run_gsm8k_benchmark

# Run MMLU
result = run_mmlu_benchmark(
    orchestrator,
    dataset_path=Path("./data/mmlu"),  # Optional
    max_samples=100,  # Optional limit
)

# Run TruthfulQA
result = run_truthfulqa_benchmark(
    orchestrator,
    max_samples=50
)

# Run GSM8K
result = run_gsm8k_benchmark(
    orchestrator,
    max_samples=100
)
```

### Unified Benchmark Runner

```python
from experiments.run_all_benchmarks import BenchmarkSuite

suite = BenchmarkSuite(orchestrator)

# Run all benchmarks
summary = suite.run_all(
    max_samples_per_benchmark=100,
    benchmarks=['mmlu', 'truthfulqa', 'gsm8k']  # Optional subset
)

# Get results
print(summary['overall']['overall_accuracy'])
print(f"Total cost: ${summary['overall']['total_cost_usd']:.2f}")
```

---

## Human Evaluation APIs

```python
from experiments.human_eval.evaluation_framework import HumanEvaluationFramework

framework = HumanEvaluationFramework()

# Create evaluation batch
queries = ["What is AI?", "Explain quantum computing"]
samples = framework.create_evaluation_batch(
    orchestrator,
    queries,
    batch_name="pilot_study",
    num_samples=10
)

# Export for MTurk
framework.export_for_mturk(samples)

# Create local evaluation interface
framework.create_evaluation_interface_html(samples)

# Load ratings
ratings = framework.load_ratings(Path("./ratings.json"))

# Calculate statistics
stats = framework.generate_statistics()
framework.print_statistics(stats)
```

---

## Distributed System APIs

```python
from src.distributed.distributed_orchestrator import DistributedOrchestrator

# Initialize distributed system
dist_orch = DistributedOrchestrator()

# Add nodes
nodes = ["node1.example.com:8000", "node2.example.com:8000"]
dist_orch.initialize_distributed_mode(nodes)

# Scale up
dist_orch.scale_up(num_nodes=2)

# Scale down
dist_orch.scale_down(num_nodes=1)

# Process query (automatically distributed)
result = dist_orch.process_query("What is ML?")
```

---

## Configuration

### YAML Configuration

```yaml
# config/my_config.yaml

system:
  name: "My LLM System"
  version: "1.0.0"
  mode: "production"

models:
  supervisor:
    model_id: "supervisor-gpt4"
    model_type: "openai"  # openai, anthropic, mock
    model_name: "gpt-4"
    max_tokens: 2048
    temperature: 0.3
    base_confidence: 0.92

  teachers:
    - model_id: "teacher-math"
      model_type: "openai"
      model_name: "gpt-3.5-turbo"
      domain: "mathematics"
      max_tokens: 1024
      temperature: 0.5

routing:
  similarity_threshold: 0.80
  novel_query_threshold: 0.50

promotion:
  student_to_ta:
    min_queries: 50
    min_confidence: 0.75
    min_win_rate: 0.65
```

### Environment Variables

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database
VECTOR_DB_PATH=./data/vector_db
METRICS_DB_PATH=./data/metrics.db

# System
LOG_LEVEL=INFO
CONFIDENCE_THRESHOLD=0.7
```

---

## Examples

### Complete Example: Production System

```python
from src.core.orchestrator import Orchestrator
from src.models.openai_model import create_gpt4_model, create_gpt35_turbo_model
from src.models.anthropic_model import create_claude_haiku_model

# Initialize orchestrator
orch = Orchestrator()

# Add supervisor (GPT-4)
supervisor = create_gpt4_model("supervisor")
supervisor.role = supervisor.role.SUPERVISOR
orch.supervisor = supervisor
orch.models["supervisor"] = supervisor

# Add teachers
teacher_math = create_gpt35_turbo_model("teacher-math", domain="mathematics")
teacher_math.role = teacher_math.role.TEACHER
orch.models["teacher-math"] = teacher_math

# Add students (Claude Haiku - cheapest)
student_math = create_claude_haiku_model("student-math", domain="mathematics")
student_math.role = student_math.role.STUDENT
student_math.teacher_id = "teacher-math"
orch.models["student-math"] = student_math

# Process queries
queries = [
    ("What is calculus?", "mathematics"),
    ("Solve: 2x + 5 = 13", "mathematics"),
]

for query, domain in queries:
    result = orch.process_query(query, domain)
    print(f"Q: {query}")
    print(f"A: {result['final_response']}")
    print(f"Model: {result['winner_model']}")
    print()
```

### Example: Run Full Evaluation Pipeline

```python
# 1. Initialize system
orchestrator = Orchestrator(config_path="./config/real_llm_config.yaml")

# 2. Run benchmarks
from experiments.run_all_benchmarks import BenchmarkSuite

suite = BenchmarkSuite(orchestrator)
results = suite.run_all(max_samples_per_benchmark=500)

# 3. Generate figures
from paper.generate_figures import generate_all_figures
generate_all_figures()

# 4. Create human evaluation batch
from experiments.human_eval.evaluation_framework import HumanEvaluationFramework

framework = HumanEvaluationFramework()
queries = ["Query 1", "Query 2", ...]  # Your queries
samples = framework.create_evaluation_batch(orchestrator, queries)
framework.create_evaluation_interface_html(samples)

# 5. Print final statistics
print(f"Benchmark accuracy: {results['overall']['overall_accuracy']:.3f}")
print(f"Total cost: ${results['overall']['total_cost_usd']:.2f}")
```

---

## Error Handling

### API Errors

```python
from openai import OpenAIError
from anthropic import AnthropicError

try:
    response = model.generate_response("Test query")
except OpenAIError as e:
    print(f"OpenAI API error: {e}")
    # Fallback logic
except AnthropicError as e:
    print(f"Anthropic API error: {e}")
    # Fallback logic
```

### Rate Limiting

The system automatically handles rate limiting with exponential backoff (up to 3 retries).

---

## Best Practices

### 1. Cost Management

```python
# Set budget limits
orchestrator.config['cost_tracking'] = {
    'enabled': True,
    'alert_threshold_usd': 100.0,
    'budget_limit_usd': 1000.0,
}

# Monitor costs
for model_id, model in orchestrator.models.items():
    if hasattr(model, 'total_cost'):
        print(f"{model_id}: ${model.total_cost:.4f}")
```

### 2. Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Now see detailed logs
result = orchestrator.process_query("Test")
```

### 3. Caching

```python
# Enable response caching
orchestrator.enable_response_cache(ttl=3600)
```

### 4. Error Recovery

```python
# Set fallback chain
orchestrator.set_fallback_chain([
    "gpt-4",
    "gpt-3.5-turbo",
    "mock-model"
])
```

---

## CLI Commands

```bash
# Run benchmarks
python experiments/run_all_benchmarks.py --config=config/real_llm_config.yaml --max-samples=100

# Generate figures
python paper/generate_figures.py

# Run demo
python examples/real_llm_demo.py

# Run enterprise demo
python examples/enterprise_demo.py

# Validate theorems
python paper/experiments/validate_theorems.py --queries=1000
```

---

## Docker Deployment

```bash
# Build image
docker build -t llm-teacher-student .

# Run container
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v $(pwd)/data:/app/data \
  llm-teacher-student

# Or use docker-compose
docker-compose up -d
```

---

## Kubernetes Deployment

```bash
# Create secrets
kubectl create secret generic llm-secrets \
  --from-literal=OPENAI_API_KEY=$OPENAI_API_KEY \
  --from-literal=ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -n llm-system

# Deploy
kubectl apply -f k8s/

# Check status
kubectl get pods -n llm-system

# View logs
kubectl logs -f deployment/llm-teacher-student -n llm-system
```

---

## Troubleshooting

### Common Issues

**Error: "No API key found"**
- Set environment variable: `export OPENAI_API_KEY=sk-...`
- Or add to `.env` file

**Error: "Rate limit exceeded"**
- System automatically retries with backoff
- Consider upgrading API tier

**Error: "Model not found"**
- Check model name (e.g., `gpt-4`, not `gpt4`)
- Ensure you have access to the model

**High costs:**
- Use mock models for testing
- Enable caching
- Use cheaper models for students

---

## Support

- GitHub Issues: https://github.com/sidataba/LLMs--Self-Evolving-Teacher-Student-Architecture/issues
- Email: hieuhip4444@gmail.com
- Documentation: See README.md and paper/research_paper.md

---

**Version:** 0.1.0
**Last Updated:** 2025-01-10
