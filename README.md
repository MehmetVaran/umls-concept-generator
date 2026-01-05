# UMLS Concept Generation and Evaluation Framework

A comprehensive Python framework for generating and evaluating UMLS (Unified Medical Language System) medical concepts for diseases using LangChain multi-agent orchestration and semantic similarity analysis.

## Overview

This project contains two main components:

1. **Concept Generator** (`concept_generator_main.py`) - Multi-agent system that discovers UMLS concepts for diseases
2. **Concept Evaluator** (`evaluate_umls_concepts.py`) - Framework for evaluating the quality and relevance of generated concepts

### Supported Diseases

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Effusion
- Emphysema
- Fibrosis
- Hernia
- Infiltration
- Mass
- No Finding
- Nodule
- Pleural Thickening
- Pneumonia
- Pneumothorax

## Installation

### Prerequisites

- Python 3.10+
- UMLS API Key (from [NLM](https://www.nlm.nih.gov/research/umls/))
- OpenAI API Key (for LangChain multi-agent system)

### Setup

1. **Clone and install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Configure API keys:**

Create two files in the project root directory:

**`.umls_api_key`** - Your UMLS API key
```
YOUR_UMLS_API_KEY_HERE
```

**`.openai_api_key`** - Your OpenAI API key
```
YOUR_OPENAI_API_KEY_HERE
```

Alternatively, set environment variables:
```bash
export UMLS_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
```

3. **Install sentence transformers** (required for semantic similarity):

```bash
pip install sentence-transformers numpy
```

## Project Structure

```
ceng568-termproject/
├── concept_generator_main.py    # Multi-agent concept discovery system
├── evaluate_umls_concepts.py    # Concept evaluation framework
├── requirements.txt             # Python dependencies
├── data/
│   ├── chestxray_classes.txt   # List of diseases to process
│   ├── umls_concepts/
│   │   ├── base_concepts/      # Manually curated base concepts
│   │   └── 20260104_183005/    # Generated concepts (timestamped)
│   └── reports/                # Evaluation reports
└── README.md
```

## Usage

### 1. Generating UMLS Concepts

The concept generator uses a LangChain multi-agent system to intelligently search UMLS and expand disease concepts.

#### Single Disease

```bash
python3 concept_generator_main.py --disease "Pneumonia" --max-concepts 10
```

#### Multiple Diseases from File

```bash
python3 concept_generator_main.py \
  --disease-file data/chestxray_classes.txt \
  --max-concepts 10
```

#### Generate Concept Sets from Existing JSON

```bash
python3 concept_generator_main.py \
  --from-json data/umls_concepts/20260104_183005/ \
  --output-dir data/
```

#### Command Options

| Option | Description |
|--------|-------------|
| `--disease DISEASE` | Single disease name to process |
| `--disease-file PATH` | Path to newline-delimited disease list |
| `--max-concepts N` | Maximum concepts per disease (default: 10) |
| `--relation-types TYPE [TYPE ...]` | Specific UMLS relations (e.g., RO, RB) |
| `--verbose` | Enable debug logging |
| `--from-json PATH` | Generate concept sets from existing JSON |

### 2. Evaluating Concepts

The evaluator analyzes concept sets across multiple dimensions: coverage, quality, diversity, and relevance.

#### Evaluate Single Disease

```bash
python3 evaluate_umls_concepts.py data/umls_concepts/base_concepts/Pneumonia.txt
```

#### Evaluate Directory of Diseases

```bash
python3 evaluate_umls_concepts.py data/umls_concepts/base_concepts/
```

#### With Output Report

```bash
python3 evaluate_umls_concepts.py \
  --data-path data/umls_concepts/base_concepts/ \
  --output data/reports/evaluation.txt
```

#### Evaluate Specific Disease Only

```bash
python3 evaluate_umls_concepts.py \
  --data-path data/umls_concepts/base_concepts/ \
  --disease Pneumonia
```

#### Command Options

| Option | Description |
|--------|-------------|
| `data_path` or `--data-path PATH` | .txt file or directory with .txt files |
| `--disease NAME` | Evaluate specific disease only |
| `--output PATH` | Save report to file |
| `--no-embeddings` | Disable semantic similarity (faster) |
| `--verbose` | Enable debug logging |

## Data Formats

### Input: Text File Format (`.txt`)

Disease concepts are stored as **one concept per line**, with the filename serving as the disease name.

**Example: `data/umls_concepts/base_concepts/Pneumonia.txt`**
```
Lung infection
Bacterial pneumonia
Viral pneumonia
Alveolitis
Bronchopneumonia
Aspiration pneumonia
Community-acquired pneumonia
Pneumococcal pneumonia
Lower respiratory tract infection
Pulmonary infiltrate
```

### Output: JSON Format

Generated concepts include detailed metadata from UMLS:

```json
{
  "disease": "Pneumonia",
  "candidate_concepts": [
    {
      "name": "Pneumonia",
      "ui": "C0032285",
      "semantic_types": ["T047"],
      "definition": "..."
    }
  ],
  "related_concepts": {
    "C0032285": [
      {
        "name": "Bacterial Infection",
        "relatedId": "C0004623",
        "relationLabel": "RO",
        "semantic_types": ["T047"]
      }
    ]
  }
}
```

## Evaluation Metrics

The evaluator provides comprehensive scoring across multiple dimensions:

### 1. **Coverage Score** (0-1)
- Measures the number of concepts generated
- Normalized to 50 concepts as baseline
- Indicates comprehensiveness of concept set

### 2. **Quality Score** (0-1)
- Filters out noise patterns (all caps, all lowercase, numbers, special characters)
- Detects duplicates (case-insensitive)
- Penalizes concepts that are too short (<3 chars) or too long (>100 chars)
- Formula: `1 - (noise_ratio × 0.5 + duplicate_ratio × 0.5)`

### 3. **Diversity Score** (0-1)
- Measures uniqueness of concepts
- Calculates word overlap between concepts
- Higher score means more distinct concepts
- Formula: `unique_concepts / total_concepts`

### 4. **Relevance Score** (0-1)
- Uses sentence transformers (`all-mpnet-base-v2`) for semantic similarity
- Compares each concept embedding to disease name embedding
- Normalized to 0-1 range
- Requires `sentence-transformers` package

### 5. **Semantic Diversity Score** (0-1)
- Analyzes distribution of UMLS semantic types
- Normalized to 10 unique types as baseline
- Indicates breadth of concept categorization

### Overall Score
Weighted combination of component scores:
- Coverage: 20%
- Quality: 30%
- Diversity: 20%
- Relevance: 20%
- Semantic Diversity: 10%

*Note: When evaluating `.txt` files (which lack semantic type data), weights are redistributed proportionally.*

## Evaluation Report Example

```
================================================================================
UMLS Concept Set Evaluation Report: Pneumonia
================================================================================

SUMMARY METRICS
--------------------------------------------------------------------------------
Total Concepts: 45
Unique Concept Names: 42
Candidate Concepts: 5
Related Concepts: 40

OVERALL SCORE
--------------------------------------------------------------------------------
Overall Score: 0.752 / 1.0

Component Scores:
  - Coverage: 0.900
  - Quality: 0.800
  - Diversity: 0.933
  - Relevance: 0.712
  - Semantic_diversity: 0.000

QUALITY METRICS
--------------------------------------------------------------------------------
Noise Count: 3
Quality Concepts: 42
Duplicates: 0
Noise Ratio: 6.67%
Duplicate Ratio: 0.00%

SEMANTIC TYPES
--------------------------------------------------------------------------------
Unique Semantic Types: 0

RELATION TYPES
--------------------------------------------------------------------------------
Unique Relation Types: 0

RELEVANCE SCORES (Semantic Similarity)
--------------------------------------------------------------------------------
Average Relevance: 0.425
Min Relevance: -0.123
Max Relevance: 0.892

DIVERSITY
--------------------------------------------------------------------------------
Diversity Score: 0.933
Unique Ratio: 0.933

================================================================================
```

## Workflow Example

### Complete Pipeline: From Concept Generation to Evaluation

```bash
# 1. Generate UMLS concepts for diseases
python3 concept_generator_main.py \
  --disease-file data/chestxray_classes.txt \
  --max-concepts 10 \
  --verbose

# 2. Evaluate the generated concepts
python3 evaluate_umls_concepts.py \
  --data-path data/umls_concepts/20260104_183005/ \
  --output data/reports/multiagents_concepts.txt \
  --verbose

# 3. Compare with base concepts
python3 evaluate_umls_concepts.py \
  --data-path data/umls_concepts/base_concepts/ \
  --output data/reports/base_concepts.txt \
  --verbose
```

## Dependencies

Core dependencies:
- `langchain` - AI agent orchestration
- `langchain-openai` - OpenAI integration
- `sentence-transformers` - Semantic similarity (optional but recommended)
- `numpy` - Numerical computations
- `pydantic` - Data validation
- `requests` - HTTP client for UMLS API

See `requirements.txt` for complete list.

## API Reference

### Concept Generator

```python
from concept_generator_main import LangChainConceptGenerator

# Initialize generator
generator = LangChainConceptGenerator(
    api_key="your_umls_key",
    llm_model="gpt-4o-mini",
    verbose=True
)

# Generate concepts for single disease
concepts = generator.generate(
    "Pneumonia",
    max_concepts=10,
    relation_types=["RO", "RB"]
)

# Generate for multiple diseases
batch_results = generator.generate_batch(
    ["Pneumonia", "Atelectasis", "Edema"],
    max_concepts=10
)
```

### Concept Evaluator

```python
from evaluate_umls_concepts import UMLSConceptEvaluator, load_disease_data

# Load concept data
data = load_disease_data("data/umls_concepts/base_concepts/")

# Initialize evaluator
evaluator = UMLSConceptEvaluator(use_embeddings=True)

# Evaluate single disease
metrics = evaluator.evaluate_concept_set("Pneumonia", data["Pneumonia"])

# Generate report
report = evaluator.generate_evaluation_report(
    data["Pneumonia"],
    output_path="report.txt"
)
```

## Troubleshooting

### "sentence-transformers not available" Warning

The sentence-transformers library is not installed. Install it with:
```bash
pip install sentence-transformers
```

### UMLS API Key Error

Verify your UMLS API key:
1. Check `.umls_api_key` file exists in project root
2. Ensure the key is valid at [NLM](https://www.nlm.nih.gov/research/umls/)
3. Try setting as environment variable: `export UMLS_API_KEY="your_key"`

### OpenAI API Error

Verify your OpenAI API key:
1. Check `.openai_api_key` file exists in project root
2. Ensure the key has sufficient credits
3. Try setting as environment variable: `export OPENAI_API_KEY="your_key"`

### Out of Memory Error

If processing many diseases, reduce `max_concepts` or evaluate diseases individually.

## Performance Notes

- **First run**: Sentence transformer model downloads (~400MB) on first use
- **Embedding calculation**: ~1-2 seconds per disease with embeddings enabled
- **Batch processing**: ~30 seconds per disease (including UMLS API calls)

## Output Locations

| Component | Output | Location |
|-----------|--------|----------|
| Generated Concepts (JSON) | All diseases | `data/umls_concepts/{TIMESTAMP}/*.json` |
| Generated Concepts (Text) | All diseases | `data/umls_concepts/{TIMESTAMP}/*.txt` |
| Evaluation Reports | Combined | `data/reports/*.txt` |

## Citation

If you use this framework, please cite the UMLS:

```
UMLS Knowledge Sources [dataset on the Internet]. Release 2024AA. Bethesda (MD): National Library of Medicine (US); 2024 May 6 [cited 2024 Jul 15]. Available from: http://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html
```

## License

This project is part of CENG 568 Term Project.

## Contact

For issues or questions, please refer to the project repository.
