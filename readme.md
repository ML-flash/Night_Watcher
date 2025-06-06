# Night_watcher Intelligence Gathering System

A streamlined system for analyzing news and identifying authoritarian patterns with enhanced memory capabilities.

## Overview

Night_watcher is an intelligence gathering and analysis tool designed to monitor political content for authoritarian patterns. It collects articles from news sources and official government feeds, analyzes them for manipulation techniques and authoritarian indicators, and builds a knowledge base of patterns over time.

## Quick Start

1. Make sure you have Python 3.8+ installed
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the content collection:
   ```bash
   python night_watcher.py
   ```
4. Run the knowledge graph component:
   ```bash
   python night_watcher_kg.py
   ```

That's it! The system will automatically:
- Create necessary directories
- Generate default configuration files if needed
- Connect to LM Studio at localhost:1234 (or use Anthropic API)
- Collect and analyze articles
- Identify authoritarian patterns
- Build a comprehensive knowledge graph
- Generate intelligence reports
- Save all results in the output directory

## Components

The Night_watcher system consists of the following core components:

1. **Content Collector**: Gathers politically-focused content from various RSS feeds with filtering
2. **Content Analyzer**: Identifies manipulation techniques and authoritarian patterns in content
3. **Knowledge Graph**: Maps relationships between entities and events to track patterns over time
4. **Document Repository**: Stores documents with cryptographic provenance

## Knowledge Graph

The Knowledge Graph component is designed to track patterns of authoritarian behavior over time:

```bash
python night_watcher_kg.py [OPTIONS]
```

Options:
- `--config PATH` - Path to configuration file (default: config.json)
- `--graph-file PATH` - Override path to knowledge graph file
- `--taxonomy-file PATH` - Override path to taxonomy file
- `--analyzed-dir PATH` - Override path to analyzed directory
- `--file-pattern PATTERN` - Override file pattern for analysis files
- `--reports-dir PATH` - Override path to reports directory
- `--trend-days DAYS` - Override number of days for trend analysis
- `--no-viz` - Disable visualization generation
- `--create-config` - Create default config file and exit
- `--verbose` - Enable verbose logging

The Knowledge Graph tracks:
- Actors (people, institutions, media outlets)
- Events and their causal relationships
- Authoritarian action patterns
- Narratives and their normalization role
- Changes to legal frameworks and procedural norms

It generates reports on:
- Authoritarian trends (0-10 score)
- Democratic erosion analysis
- Influential actors and their relationships
- Coordination patterns between actors
- Temporal patterns revealing larger strategies

## Command Line Options

The main `night_watcher.py` script accepts several optional parameters:

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
To customize, edit the generated `config.json` file.  Each RSS source may
optionally define a `site_domain` field.  When collecting articles the
collector derives the base domain for Wayback queries from this field.  If it
is not provided, the domain is taken from the first article link.

## Output Structure

After running, you'll find the following in your output directory:

- `collected/` - Raw article data
- `analyzed/` - Analysis results with manipulation scoring
- `analysis/` - Pattern analysis and intelligence outputs
- `knowledge_graph/` - Entity and relationship data
- `documents/` - Document repository with cryptographic provenance
- `logs/` - Execution logs

## Security Considerations

 - The framework runs locally with no external API calls except to specified news and government sources
- All LLM interactions happen locally through LM Studio (or optionally via Anthropic API)
- Documents are stored with cryptographic provenance to prevent tampering
- No data is sent to external servers unless using Anthropic API

## License

This project is released into the public domain.
