# Night_watcher Framework

A modular system for analyzing news, identifying divisive content, and generating counter-narratives with memory capabilities.

## Overview

Night_watcher is a sophisticated tool designed to analyze news articles, identify divisive content and manipulation techniques, and generate strategic counter-narratives aimed at reducing polarization. The framework uses large language models (LLMs) to perform its analysis and generation tasks, with a modular architecture that allows for easy extension and customization.

Key features include:
- Automated collection of news articles from diverse sources
- Deep analysis of media framing, emotional triggers, and manipulation techniques
- Generation of targeted counter-narratives for different demographic groups
- Strategic messaging optimized for different audiences
- Memory system for maintaining context across analyses
- Pattern recognition to identify trends in media coverage

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/night_watcher.git
cd night_watcher

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- Python 3.8+
- requests
- feedparser
- newspaper3k
- numpy
- Optional: 
  - faiss-cpu (for enhanced vector search)
  - chromadb (for persistent vector store)
  - sentence-transformers (for better embeddings)

## Usage

Initialize the configuration:
```bash
python -m night_watcher init --output config.json
```

Run the analysis workflow:
```bash
python -m night_watcher run --config config.json
```

Analyze existing data for patterns:
```bash
python -m night_watcher analyze --memory-file data/memory/night_watcher_memory.pkl --output-dir analysis_results
```

Search the memory system:
```bash
python -m night_watcher search --memory-file data/memory/night_watcher_memory.pkl --query "climate change"
```

## Architecture

The framework is structured around modular components:

- **Agents**: Each specialized for a specific task (collection, analysis, generation)
- **Memory System**: Vector-based storage for maintaining context across analyses
- **Pattern Recognition**: Tools for identifying trends and narratives in data
- **Workflow Orchestration**: Coordinates execution of agents in a configurable pipeline

## Configuration

The framework is configured via a JSON file. Here's a sample configuration:

```json
{
  "llm_provider": {
    "type": "lm_studio",
    "host": "http://localhost:1234"
  },
  "content_collection": {
    "article_limit": 5,
    "sources": [
      {"url": "https://www.reuters.com/rss/topNews", "type": "rss", "bias": "center"},
      {"url": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml", "type": "rss", "bias": "center-left"},
      {"url": "https://feeds.foxnews.com/foxnews/politics", "type": "rss", "bias": "right"}
    ]
  },
  "content_analysis": {
    "manipulation_threshold": 6
  },
  "memory": {
    "store_type": "simple",
    "file_path": "data/memory/night_watcher_memory.pkl"
  }
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request