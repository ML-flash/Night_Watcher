# Night_watcher Framework

A covert system for analyzing news, identifying divisive content, and generating counter-narratives with memory capabilities.

## Overview

Night_watcher is designed as a standalone, low-profile tool that analyzes media content for manipulation techniques and generates strategic counter-narratives. It uses large language models (LLMs) through LM Studio to perform its analysis and generation tasks.

## Installation Options

### Option 1: Quick Setup (Recommended)

1. Download the `install_nightwatcher.py` script
2. Run the installer:
   ```bash
   python install_nightwatcher.py
   ```
3. The installer will set up everything in a hidden directory at `~/.documents/analysis_tool` by default

### Option 2: Manual Setup

1. Clone or download the source code
2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Initialize the configuration:
   ```bash
   python main.py init
   ```

## Dependencies

- Python 3.8+
- Required:
  - requests
  - feedparser
  - newspaper3k
  - numpy
- Optional (enhanced capabilities): 
  - faiss-cpu (for enhanced vector search)
  - chromadb (for persistent vector store)
  - sentence-transformers (for better embeddings)

## Using Night_watcher

### Prerequisites

1. Install and run [LM Studio](https://lmstudio.ai/) locally
2. Start a local server with a language model of your choice
   - The default configuration expects the server at http://localhost:1234

### Commands

Initialize the configuration:
```bash
python nightwatcher.py init --output config.json
```

Run the analysis workflow:
```bash
python nightwatcher.py run --config config.json
```

Analyze existing data for patterns:
```bash
python nightwatcher.py analyze --memory-file data/memory/night_watcher_memory.pkl --output-dir analysis_results
```

Search the memory system:
```bash
python nightwatcher.py search --memory-file data/memory/night_watcher_memory.pkl --query "climate change"
```

### Configuration Options

Edit `config.json` to customize:
- LLM provider settings
- News sources and bias labels
- Manipulation thresholds
- Demographic targets for counter-narratives

## Security Considerations

- The framework is designed to be covert and leave minimal traces
- Files are stored in non-obvious locations by default
- No external API calls except to specified news sources
- All LLM interactions happen locally through LM Studio

## Architecture

The framework uses a modular architecture:

- **Agents**: Specialized components for collection, analysis, and generation
- **Memory System**: Vector-based storage for maintaining context over time
- **Workflow Orchestration**: Coordinates the analysis pipeline
- **Pattern Recognition**: Identifies trends and narratives in collected data

## Creating a Distribution Package

To create a standalone installer for distribution:

```bash
python bundle_nightwatcher.py
```

This creates a self-contained script that can be sent to other systems and will set up the entire framework when run.

## Operational Security

- Run behind a VPN or Tor when collecting content
- Use a separate user account for running the framework
- Consider using a dedicated machine not connected to personal accounts
- Regularly clear logs and memory if not needed for analysis

## License

This project is released into the public domain - see the [LICENSE](LICENSE) file for details.
