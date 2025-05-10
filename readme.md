# Night_watcher Framework



## Installation

1. Ensure Python 3.8+ is installed
2. Set up a virtual environment (recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # Linux/Mac
   env\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage

```bash
# Run with default settings
python night_watcher.py

# Run with custom config
python night_watcher.py --config my_config.json

# Run with Anthropic API
python night_watcher.py --use-anthropic --anthropic-key YOUR_API_KEY

# Reset date tracking to start fresh
python night_watcher.py --reset-date

# Run with debug logging
python night_watcher.py --verbose
```

## Configuration

The default configuration will be generated automatically. You can modify `config.json` to:
- Add news sources
- Adjust analysis parameters
- Configure output locations
- Set up LLM preferences

Example config:
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
      {"url": "https://www.reuters.com/rss/topNews", "type": "rss", "bias": "center"}
    ]
  },
  "output": {
    "base_dir": "data"
  }
}
```

## LLM Setup Options

1. **Local LLM via LM Studio** (recommended):
   - Download [LM Studio](https://lmstudio.ai/)
   - Load a suitable model (32B+ recommended)
   - Start local server on port 1234
   
2. **Cloud API (Claude)**:
   - Obtain Anthropic API key
   - Run with `--use-anthropic --anthropic-key YOUR_API_KEY`

## Output Structure

All outputs are saved in the configured data directory:
- `collected/`: Raw article data
- `analyzed/`: Analysis results
- `memory/`: System memory and vector database
- `documents/`: Source document repository with provenance tracking

## Command Line Options

```
--config PATH          Path to configuration file
--llm-host URL         LLM provider URL
--article-limit N      Maximum articles per source
--output-dir PATH      Output directory
--reset-date           Reset date tracking
--use-anthropic        Use Anthropic API
--anthropic-key KEY    Anthropic API key
--verbose              Enable verbose logging
```
