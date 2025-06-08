# Night_watcher Intelligence Gathering System

A framework for monitoring and analyzing political media to detect authoritarian patterns and threats to democracy. This streamlined system includes enhanced memory capabilities for long-running investigations.

## 🚀 Quick Start (5 minutes)

### 1. Prerequisites
- Python 3.8+ installed
- 8GB+ RAM recommended
- ~4GB free disk space

### 2. Setup
```bash
# Clone the repository (or download files)
git clone [your-repo-url]
cd night_watcher

# Install dependencies
pip install -r requirements.txt

# Run setup check
python setup_night_watcher.py
```

### 3. Configure LLM (Choose One)

**Option A: Local with LM Studio (Recommended)**
1. Download [LM Studio](https://lmstudio.ai/)
2. Download a model (e.g., Qwen2.5-32B-Instruct or similar)
3. Start the local server in LM Studio (default port 1234)
4. Edit `config.json` if you run LM Studio on a different port:
```json
"llm_provider": {
  "type": "lm_studio",
  "host": "http://localhost:1234"
}
```
The web dashboard lists available models from LM Studio so you can switch models at runtime.

**Option B: Anthropic API**
```bash
export ANTHROPIC_API_KEY='your-key-here'
```

### 4. First Run

**Test Mode (Recommended for first time):**
```bash
# Collect just 5 articles to test
python Night_Watcher.py --collect --max-articles 5

# Analyze 3 articles
python Night_Watcher.py --analyze --max-articles 3

# Check what we found
python Night_Watcher.py --status
```

**Full Pipeline:**
```bash
python Night_Watcher.py --full
```

**Web Dashboard (Easier):**
```bash
python Night_Watcher.py --web
# Open browser to http://localhost:5000
```

## 📊 What Night_watcher Does

1. **Collects** - Gathers political news from diverse sources
2. **Analyzes** - Identifies manipulation techniques and authoritarian indicators
3. **Extracts** - Builds a knowledge graph of actors, events, and relationships
4. **Tracks** - Monitors patterns over time to identify democratic erosion
5. **Preserves** - Maintains cryptographic provenance of all data

## 🎯 Key Features

 - **Multi-source collection** from verified news outlets and official government feeds
- **7-round analysis pipeline** for deep content understanding
- **Knowledge graph** tracking entities and relationships
- **Vector search** for finding similar patterns
- **Review queue** for human validation
- **Web dashboard** for monitoring and control
- **Cryptographic provenance** to prevent tampering
- **Automatic archival retrieval** when collecting older dates

## 🔧 Command Line Usage

### Basic Commands
```bash
# Check system status
python Night_Watcher.py --status

# Run collection only
python Night_Watcher.py --collect

# Run analysis only  
python Night_Watcher.py --analyze

# Build knowledge graph
python Night_Watcher.py --build-kg

# Run everything
python Night_Watcher.py --full
```

### Advanced Options
```bash
# Collect from specific date
python Night_Watcher.py --collect --mode first_run

# Analyze more articles
python Night_Watcher.py --analyze --max-articles 50

# Use different data directory
python Night_Watcher.py --base-dir /path/to/data
```

## 🌐 Web Dashboard

The easiest way to use Night_watcher:

```bash
python Night_Watcher.py --web
```

Features:
- Real-time monitoring of collection and analysis
- Review queue for validating results  
- Source management
- Knowledge graph statistics
- Vector search interface

## 📁 Output Structure

```
data/
├── collected/          # Raw articles
├── analyzed/           # Analysis results with manipulation scores
├── documents/          # Document repository with provenance
├── knowledge_graph/    # Entity and relationship data
├── vector_store/       # Embeddings for similarity search
└── logs/              # System logs and JSON failures
```

## 📦 Exporting & Applying Updates

Create a signed bundle of the current repository, graph and vector store:

```bash
python export_artifact.py --output my_bundle.tar.gz

# Use custom paths with --kg-dir, --vector-dir and --documents-dir if needed

```

To integrate an update from another bundle:

```bash
python update_artifact.py my_bundle.tar.gz

# Use the same optional directories as above

```

All documents and analyses are verified before import and the knowledge graph is
merged safely.

## 🔍 Understanding the Analysis

Each article goes through 7 rounds of analysis:

1. **Fact Extraction** - Objective facts and quotes
2. **Article Analysis** - Bias, framing, manipulation techniques
3. **Node Extraction** - Entities (people, institutions, events)
4. **Node Deduplication** - Merging similar entities
5. **Edge Extraction** - Relationships between entities
6. **Edge Enrichment** - Severity and impact assessment
7. **Package Ingestion** - Final knowledge graph format

## ⚠️ Troubleshooting

### LLM Not Responding
- Check LM Studio is running with a model loaded
- Verify Anthropic API key is set correctly
- Test with: `curl http://localhost:1234/v1/models`

### JSON Extraction Errors
- Check `data/logs/json_failures_*.txt` for patterns
- The system will retry once automatically
- Failed analyses go to review queue

### RSS Feed Failures
- Normal - some feeds may be temporarily down
- System continues with working feeds
- Add new sources via web dashboard
- For Wayback queries the collector uses the `site_domain` specified in a
  source entry. If omitted, the domain is derived from the first article link.
- If the date range goes back more than 5 days, archived feed snapshots are queried automatically.

### Out of Memory
- Reduce `article_limit` in config.json
- Process fewer articles with `--max-articles`
- Use smaller embedding model

## 🛡️ Security & Privacy

- All analysis runs locally (with LM Studio)
- Documents stored with cryptographic signatures
- No data sent to external servers (unless using Anthropic)
- Review queue ensures quality control

## 📈 Monitoring Authoritarian Patterns

The system tracks:
- **Power consolidation** (purges, co-opts, expands_power)
- **Institutional capture** (undermines, restricts, delegitimizes)
- **Opposition suppression** (criminalizes, intimidates, targets)
- **Information control** (censors, narrative manipulation)
- **Democratic erosion** (procedural norm violations)


## 📜 License

This project is released into the public domain. Use it to defend democracy.

---

**Remember**: *"The price of freedom is eternal vigilance"* - Thomas Jefferson

For detailed documentation, see the docs/ directory.
For quick help: `python Night_Watcher.py --help`
