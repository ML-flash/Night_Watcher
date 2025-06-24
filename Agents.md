# Night_watcher Framework Agent Guidelines

## Purpose
This document provides guidelines for AI agents working on the Night_watcher framework to prevent breaking existing functionality while developing new features.

## Development Philosophy
- **NO OVER-ENGINEERING** - Take the direct, effective path
- Pragmatic solutions over complex abstractions
- Maintain existing patterns and conventions
- Focus on functionality, not perfection

## Core Components - Understand Before Modifying

### 1. Main Controller (`Night_Watcher.py`)
- Entry point orchestrating all operations
- Initialization sequence is order-dependent
- Component dependencies must be respected
- Command routing logic is central to CLI operation

### 2. Document Repository (`document_repository.py`)
- SHA-256 document ID generation is foundational
- Provenance tracking enables audit trails
- Storage/retrieval methods are used throughout system

### 3. Content Collector (`collector.py`)
- RSS feed parsing with fallback mechanisms
- Article extraction with multiple strategies
- Deduplication prevents redundant processing
- Source management for bias tracking

### 4. Content Analyzer (`analyzer.py`)
- Template-based multi-round analysis system
- JSON template loading and validation
- Token optimization for LLM context windows
- Response parsing with retry logic

### 5. Knowledge Graph (`knowledge_graph.py`)
- NetworkX-based graph implementation
- Dynamic node/edge type discovery
- Relationship extraction and storage
- Graph merging for distributed updates

### 6. Vector Store (`vector_store.py`)
- FAISS index for similarity search
- Embedding generation and storage
- Sync mechanism with knowledge graph
- Metadata tracking for search results

### 7. Web Interface (`night_watcher_web.py`)
- Flask API endpoints map to operations
- Background task management
- Real-time status updates
- File-based review queue system

## Critical Interfaces - Don't Break These

### LLM Provider Interface
```python
# Any provider MUST implement:
def complete(prompt: str, max_tokens: int) -> str
def count_tokens(text: str) -> int
```

### Analysis Template Structure
```json
{
  "name": "template_name",
  "version": "1.0",
  "status": "testing|production",
  "rounds": [
    {
      "name": "round_name",
      "prompt": "template with {variables}",
      "max_tokens": 2000
    }
  ]
}
```

### Document ID Generation
```python
# This logic is used everywhere - don't change it
doc_id = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
```

## Dependencies and Flow

### Processing Pipeline
```
Collect → Store Document → Analyze → Extract KG → Build Graph → Sync Vectors
```
Each step depends on the previous - breaking one breaks the chain.

### Component Dependencies
- Analyzer needs: LLM Provider, Templates
- KG Builder needs: Completed Analyses
- Vector Store needs: Knowledge Graph
- Web UI needs: All components

## Common Breaking Points

### 1. Template Variables
Analysis templates expect specific variables:
- `{article_content}`
- `{article_title}`
- `{article_url}`
- `{previous_round_results}`

### 2. File Paths
Components expect specific directory structure:
- Templates: `*.json` in root directory
- Data: `data/` subdirectories
- Configs: `config.json` in root

### 3. JSON Parsing
Many components parse LLM JSON responses:
- Extraction logic includes retry mechanisms
- Don't assume clean JSON from LLMs
- Recovery parsing is intentional

### 4. Event System
Web dashboard expects specific event names:
- `status_update`
- `collection_progress`
- `analysis_progress`

## Safe Development Practices

### When Adding Features
1. Follow existing patterns in similar code
2. Use established helper functions
3. Maintain backward compatibility
4. Add to existing structures rather than replacing

### When Modifying Core Logic
1. Understand the full call chain
2. Check what depends on the output format
3. Test the complete pipeline
4. Preserve existing method signatures

### When Adding New Components
1. Follow the existing component pattern
2. Integrate with the web interface
3. Add to the main controller
4. Update configuration handling

## Testing Checklist

Before committing changes, verify:
- [ ] Collection still fetches articles
- [ ] Analysis produces expected JSON structure
- [ ] KG builds without errors
- [ ] Vector search returns results
- [ ] Web UI displays data correctly
- [ ] CLI commands work as expected

## Code Patterns to Follow

### Error Handling
```python
try:
    # operation
except SpecificException as e:
    self.logger.error(f"Context: {e}")
    # graceful fallback
```

### Logging
```python
self.logger.info("Starting operation")
self.logger.debug(f"Details: {details}")
self.logger.error(f"Failed: {error}")
```

### File Operations
```python
os.makedirs(directory, exist_ok=True)
with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)
```

## Anti-Patterns to Avoid

1. **Complex Abstractions**: Don't add unnecessary layers
2. **Breaking Changes**: Modify behavior, don't replace
3. **Assumption Making**: Check if paths/files exist
4. **Silent Failures**: Always log errors
5. **Tight Coupling**: Keep components independent

## Quick Reference

### Key Files and Their Roles
- `Night_Watcher.py` - Main orchestrator
- `config.json` - System configuration
- `*.json` - Analysis templates
- `KG_Taxonomy.csv` - Graph type definitions
- `requirements.txt` - Python dependencies

### Important Methods
- `ContentCollector.collect_content()` - Fetches articles
- `ContentAnalyzer.analyze_article()` - Runs analysis
- `KnowledgeGraph.add_node/edge()` - Builds graph
- `VectorStore.add_vector()` - Indexes content

### Configuration Keys
- `llm_provider` - LLM backend settings
- `feeds` - RSS source configuration
- `analysis` - Analysis parameters

Remember: The goal is extending functionality without breaking what already works. When in doubt, add new code rather than modifying existing code.
