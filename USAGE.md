# Usage Guide

This guide provides detailed instructions on how to use the Self-Evolving Teacher-Student Architecture.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Core Concepts](#core-concepts)
5. [API Reference](#api-reference)
6. [Advanced Usage](#advanced-usage)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/sidataba/LLMs--Self-Evolving-Teacher-Student-Architecture.git
cd LLMs--Self-Evolving-Teacher-Student-Architecture

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Quick Start

### Running the Basic Demo

```python
from src.core.orchestrator import Orchestrator
from src.monitoring.dashboard import Dashboard

# Initialize the system
orchestrator = Orchestrator(config_path="./config/default_config.yaml")

# Create dashboard for monitoring
dashboard = Dashboard(orchestrator)

# Process a query
result = orchestrator.process_query("What is machine learning?")

# View the result
print(f"Winner: {result['winner_model']}")
print(f"Response: {result['final_response']}")

# Check system status
dashboard.print_system_status()
```

### Running Example Scripts

```bash
# Basic demo with automated queries
python examples/basic_demo.py

# Interactive CLI demo
python examples/interactive_demo.py
```

## Configuration

### Configuration File

The system is configured via YAML files. The default configuration is in `config/default_config.yaml`.

#### Key Configuration Sections

**1. Models Configuration**

```yaml
models:
  supervisor:
    model_id: "supervisor-gpt4"
    model_type: "openai"  # openai, anthropic, mock
    model_name: "gpt-4"

  teachers:
    - model_id: "teacher-math"
      domain: "mathematics"
      specialization: ["algebra", "calculus"]

  students:
    - model_id: "student-math-1"
      domain: "mathematics"
      teacher_id: "teacher-math"
```

**2. Routing Configuration**

```yaml
routing:
  similarity_threshold: 0.80  # Route to best performer if similarity > this
  novel_query_threshold: 0.50  # Consider novel if similarity < this
```

**3. Promotion Criteria**

```yaml
promotion:
  student_to_ta:
    min_queries: 30
    min_confidence: 0.75
    min_win_rate: 0.60

  ta_to_teacher:
    min_queries: 50
    min_confidence: 0.85
    min_win_rate: 0.70
```

**4. Evaluation Weights**

```yaml
evaluation:
  weights:
    relevance: 0.3
    correctness: 0.4
    completeness: 0.2
    clarity: 0.1
```

## Core Concepts

### 1. Model Roles

- **Supervisor**: Generalist model that manages routing and evaluation
- **Teacher**: Specialized expert in a specific domain
- **TA (Teaching Assistant)**: Promoted student with higher privileges
- **Student**: Learning model that improves over time

### 2. Query Flow

1. **Query Reception**: User submits a query
2. **Routing Decision**: System determines if query is novel or similar to past queries
3. **Response Generation**:
   - Novel queries → All models respond in parallel
   - Similar queries → Route to best historical performer
4. **Evaluation**: Supervisor evaluates all responses
5. **Feedback Distribution**: All models receive learning signals
6. **Promotion Check**: System checks if any students qualify for promotion

### 3. Promotion System

Students can be promoted based on performance:

```
Student (avg_conf < 0.75)
  ↓ [30+ queries, 0.75+ confidence, 0.60+ win rate]
TA (0.75 ≤ avg_conf < 0.85)
  ↓ [50+ queries, 0.85+ confidence, 0.70+ win rate]
Teacher (avg_conf ≥ 0.85)
```

## API Reference

### Orchestrator

**Main class for system coordination**

```python
from src.core.orchestrator import Orchestrator

# Initialize
orchestrator = Orchestrator(config_path="./config/default_config.yaml")

# Process a query
result = orchestrator.process_query(
    query_text="What is AI?",
    domain="general"  # optional
)

# Get system status
status = orchestrator.get_system_status()

# Add a new student
student = orchestrator.add_student_model(
    model_id="new-student",
    domain="science",
    teacher_id="teacher-science"
)

# Export metrics
files = orchestrator.export_metrics(output_path="./exports")
```

### Dashboard

**Monitoring and visualization**

```python
from src.monitoring.dashboard import Dashboard

dashboard = Dashboard(orchestrator)

# Print system status
dashboard.print_system_status()

# Print model details
dashboard.print_model_details(model_id="student-math-1")

# Print query result
dashboard.print_query_summary(result)

# Export report
dashboard.export_dashboard_report("./report.json")
```

### Models

**Working with individual models**

```python
from src.models.base import ModelConfig, ModelRole
from src.models.student import StudentModel

# Create a student model
config = ModelConfig(
    model_id="custom-student",
    model_type="mock",
    role=ModelRole.STUDENT,
    domain="programming",
)

student = StudentModel(config)

# Generate response
response = student.generate_response("How do I write a function?")

# Check promotion eligibility
eligibility = student.is_ready_for_promotion(
    min_queries=30,
    min_confidence=0.75,
    min_win_rate=0.60
)
```

## Advanced Usage

### Custom Model Implementation

You can implement custom LLM models by extending the `BaseModel` class:

```python
from src.models.base import BaseModel, ModelResponse

class CustomLLM(BaseModel):
    def generate_response(self, query, context=None):
        # Your custom implementation
        response_text = self.call_your_llm_api(query)

        return ModelResponse(
            model_id=self.model_id,
            response_text=response_text,
            confidence=0.85,
            reasoning="Custom reasoning"
        )

    def evaluate_response(self, query, response, context=None):
        # Your custom evaluation logic
        return {
            "relevance": 0.9,
            "correctness": 0.85,
            "completeness": 0.8,
            "clarity": 0.9,
        }
```

### Real LLM Integration

To use real LLM APIs (OpenAI, Anthropic, etc.):

1. Set your API keys in `.env`:
```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

2. Update `config/default_config.yaml`:
```yaml
models:
  supervisor:
    model_type: "openai"
    model_name: "gpt-4"
```

3. Implement the API integration in your model class

### Metrics and Analytics

Export and analyze metrics:

```python
# Export to CSV
files = orchestrator.export_metrics("./analysis")

# Load with pandas
import pandas as pd

queries_df = pd.read_csv(files["queries"])
model_stats_df = pd.read_csv(files["model_stats"])

# Analyze performance
top_models = model_stats_df.sort_values("win_rate", ascending=False)
print(top_models.head())
```

### Custom Routing Strategy

Implement custom routing logic:

```python
from src.routing.query_router import QueryRouter

class CustomRouter(QueryRouter):
    def _make_routing_decision(self, query, similarity_score, similar_queries, is_novel):
        # Your custom routing logic
        if domain_is_critical(query):
            return use_best_teacher()
        else:
            return use_ensemble()
```

## Best Practices

1. **Start with Mock Models**: Test your workflow with mock models before integrating real LLMs
2. **Monitor Promotions**: Regularly check promotion events to ensure students are learning
3. **Adjust Thresholds**: Fine-tune similarity and promotion thresholds based on your use case
4. **Export Metrics**: Regularly export and analyze metrics to understand system behavior
5. **Gradual Rollout**: Start with a small number of models and gradually increase

## Troubleshooting

### Common Issues

**Issue: Students never get promoted**
- Check if min_queries threshold is too high
- Verify that students are getting enough diverse queries
- Lower the min_win_rate threshold temporarily

**Issue: Too many parallel queries (high cost)**
- Increase similarity_threshold to route more queries directly
- Adjust novel_query_threshold to be more selective

**Issue: Low-quality responses**
- Review evaluation weights
- Check if domain matching is working correctly
- Consider adjusting confidence thresholds

## Next Steps

- Explore the [examples](./examples/) directory for more use cases
- Read the architecture details in the [README](./README.md)
- Check the [tests](./tests/) for implementation examples
- Contribute improvements via pull requests
