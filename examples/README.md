# Examples

This directory contains example scripts demonstrating the Self-Evolving Teacher-Student Architecture.

## Available Examples

### 1. Basic Demo (`basic_demo.py`)

A comprehensive demonstration that shows the complete workflow:

- System initialization with multiple models
- Processing a series of queries across different domains
- Automatic model evaluation and selection
- Student model promotions based on performance
- System monitoring and statistics
- Metrics export

**Run:**
```bash
python examples/basic_demo.py
```

### 2. Interactive Demo (`interactive_demo.py`)

An interactive CLI application that lets you:

- Process custom queries in real-time
- View system status and statistics
- Inspect individual model performance
- Add new student models dynamically
- Export metrics and reports

**Run:**
```bash
python examples/interactive_demo.py
```

## Expected Output

### Basic Demo

The basic demo will:
1. Initialize a system with 1 supervisor, 3 teachers, and 4 students
2. Process 25 sample queries across mathematics, science, and programming domains
3. Show automatic promotions as students improve
4. Display detailed statistics for all models
5. Export metrics to CSV files

### Interactive Demo

The interactive demo provides a menu-driven interface where you can:
- Enter custom queries
- Monitor the system in real-time
- Experiment with different scenarios

## Data Output

Both demos create data files in the `./data/` directory:

- `vector_db/` - Vector database with query embeddings
- `metrics/` - Performance metrics and logs
- `demo_export/` - Exported CSV files with detailed metrics
- `demo_dashboard.json` - Dashboard snapshot

## Next Steps

After running the examples, you can:

1. Examine the exported CSV files to analyze model performance
2. Modify `config/default_config.yaml` to adjust system parameters
3. Add your own queries to test domain-specific scenarios
4. Implement real LLM models instead of mock models
