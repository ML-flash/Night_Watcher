# Night_watcher Intelligence Gathering System

A system for analyzing news and identifying authoritarian patterns with enhanced memory capabilities.

## Quick Start

1. Extract the Night_watcher files to any directory
2. Make sure you have Python 3.8+ installed
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the framework:
   ```bash
   python run.py
   ```

That's it! The script will automatically:
- Create necessary directories
- Generate a default configuration file if needed
- Connect to LM Studio at localhost:1234
- Collect and analyze articles
- Identify authoritarian patterns
- Save all results in the current directory

## Requirements

- Python 3.8+
- Local LLM server via [LM Studio](https://lmstudio.ai/) running on http://localhost:1234
- Required Python packages (see requirements.txt):
  - requests
  - feedparser
  - newspaper3k
  - numpy

## Command Line Options

The `run.py` script accepts several optional parameters:

```bash
python run.py [OPTIONS]
```

Options:
- `--config PATH` - Path to configuration file (default: config.json)
- `--llm-host URL` - LLM provider URL (default: http://localhost:1234)
- `--article-limit N` - Maximum articles to collect per source (default: 50)
- `--output-dir PATH` - Output directory (default: current directory)
- `--reset-date` - Reset date tracking to start from inauguration day (Jan 20, 2025)
- `--use-anthropic` - Force using Anthropic API instead of LM Studio
- `--use-repository` - Use repository-based architecture for data provenance

Example:
```bash
python run.py --llm-host http://192.168.1.100:1234 --article-limit 20 --output-dir ./outputs
```

## Configuration

The default configuration will be created automatically on first run. 
To customize, edit the generated `config.json` file:

- **LLM Provider**: Connection settings for your local LLM
- **Content Sources**: RSS feeds to monitor
- **Manipulation Threshold**: Sensitivity for detecting divisive content

## Output Structure

After running, you'll find the following in your output directory:

- `data/collected/
