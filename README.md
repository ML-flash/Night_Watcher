# Night_watcher Intelligence Gathering System

A streamlined system for analyzing news and identifying authoritarian patterns with enhanced memory capabilities.

## Overview

Night_watcher is an intelligence gathering and analysis tool designed to monitor political content for authoritarian patterns. It collects articles from news sources, analyzes them for divisive content and authoritarian indicators, and builds a knowledge base of patterns over time.

## Quick Start

1. Make sure you have Python 3.8+ installed
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the system:
   ```bash
   python night_watcher.py
   ```

That's it! The script will automatically:
- Create necessary directories
- Generate a default configuration file if needed
- Connect to LM Studio at localhost:1234 (or use Anthropic API)
- Collect and analyze articles
- Identify authoritarian patterns
- Save all results in the output directory

## Requirements

- Python 3.8+
- Local LLM server via [LM Studio](https://lmstudio.ai/) running on http://localhost:1234
  - Or an Anthropic API key if using Claude
- Required Python packages (see requirements.txt)

## Command Line Options

The `night_watcher.py` script accepts several optional parameters:

```bash
python night_watcher.py [OPTIONS]
```

Options:
- `--config PATH` - Path to configuration file (default: config.json)
- `--llm-host URL` - LLM provider URL (default: http://localhost:1234)
- `--article-limit N` - Maximum articles to collect per source (default: 5)
- `--output-dir PATH` - Output directory (default: ./data)
- `--reset-date` - Reset date tracking to start from inauguration day (Jan 20, 2025)
- `--use-anthropic` - Force using Anthropic API instead of LM Studio

Example:
```bash
python night_watcher.py --llm-host http://192.168.1.100:1234 --article-limit 20 --output-dir ./outputs
```

## Configuration

The default configuration will be created automatically on first run. 
To customize, edit the generated `config.json` file:

```json
{
  "llm_provider": {
    "type": "lm_studio",
    "host": "http://localhost:1234",
    "model": "default"
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
    "file_path": "data/memory/night_watcher_memory.pkl",
    "embedding_provider": "simple"
  },
  "output": {
    "base_dir": "data",
    "save_collected": true,
    "save_analyses": true
  },
  "logging": {
    "level": "INFO",
    "log_dir": "logs"
  }
}
```

## Output Structure

After running, you'll find the following in your output directory:

- `collected/` - Raw article data
- `analyzed/` - Analysis results with manipulation scoring
- `analysis/` - Pattern analysis and intelligence outputs
- `memory/` - System memory for tracking patterns over time
- `logs/` - Execution logs

## Core Components

The Night_watcher system consists of the following core components:

1. **Content Collector**: Gathers politically-focused content from various RSS feeds with filtering
2. **Content Analyzer**: Identifies manipulation techniques and authoritarian patterns in content
3. **Memory System**: Maintains a vector database for tracking historical patterns
4. **Knowledge Graph**: Maps relationships between entities to track patterns over time
5. **Workflow Orchestrator**: Manages the intelligence gathering process

## Security Considerations

- The framework runs locally with no external API calls except to specified news sources
- All LLM interactions happen locally through LM Studio (or optionally via Anthropic API)
- No data is sent to external servers unless using Anthropic API

## License

This project is released into the public domain.
