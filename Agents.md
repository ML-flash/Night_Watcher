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
- Event manager initialization depends on EventDatabase availability
- Export orchestrator loaded on-demand for package distribution

### 2. Document Repository (`document_repository.py`)
- SHA-256 document ID generation is foundational
- Provenance tracking enables audit trails with cryptographic signatures
- Storage/retrieval methods are used throughout system
- Integrated analysis provenance in `analysis_dir`
- HMAC-based signatures for document integrity
- Key derivation from environment or config

### 3. Content Collector (`collector.py`)
- RSS feed parsing with fallback mechanisms
- Article extraction with multiple strategies
- Deduplication prevents redundant processing
- Source management for bias tracking
- Integration with document repository for storage

### 4. Content Analyzer (`analyzer.py`)
- Template-based multi-round analysis system
- JSON template loading and validation
- Token optimization for LLM context windows
- Response parsing with retry logic
- Analysis provenance tracking with lineage
- Multi-template analysis support via document aggregation

### 5. Knowledge Graph (`knowledge_graph.py`)
- NetworkX-based graph implementation
- Dynamic node/edge type discovery from KG_Taxonomy.csv
- Relationship extraction and storage
- Graph merging for distributed updates
- Temporal relationship inference
- Export/import functionality for distribution

### 6. Vector Store (`vector_store.py`)
- FAISS index for similarity search
- Embedding generation and storage
- Sync mechanism with knowledge graph
- Metadata tracking for search results
- Export functionality for artifact creation

### 7. Web Interface (`night_watcher_web.py`)
- Flask API endpoints map to operations
- Background task management
- Real-time status updates via SSE
- File-based review queue system
- Event tracking integration at `/api/events`
- Distribution package creation endpoints

### 8. Event Tracking System
- **EventDatabase** (`event_database.py`): SQLite storage for events
- **EventManager** (`event_weight.py`): Weight calculation and observation tracking
- **EventSignature**: Normalized event matching with fuzzy logic
- **EventMatcher**: Cross-analysis event correlation
- **EventWeightCalculator**: Multi-factor confidence scoring

### 9. Event Aggregation (`event_aggregator.py`)
- Two-stage event-centric aggregation
- Cross-analysis event matching with similarity threshold
- Unified graph construction from event graphs
- Crypto lineage preservation for provenance
- Integration with main KG building process

### 10. Document Aggregation (`document_aggregator.py`)
- Multi-template analysis consolidation
- Event extraction from various analysis formats
- Event deduplication within documents
- Relationship inference between events
- Event-centric graph building

### 11. Utility Modules
- **file_utils.py**: Safe JSON operations with error handling
- **providers.py**: LLM provider management with context windows
- **export_signed_artifact.py**: Cryptographic artifact signing
- **event_analysis_inspector.py**: Analysis debugging tool

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

### Event Signature Format
```python
# Normalized event matching key
signature = f"{actor}|{action}|{date}|{location}"
```

## Dependencies and Flow

### Processing Pipeline
```
Collect → Store Document → Analyze → Extract Events → Build Graph → Sync Vectors
```
Each step depends on the previous - breaking one breaks the chain.

### Event Processing Flow
```
Analysis → Event Extraction → Event Matching → Weight Calculation → Graph Integration
```

### Component Dependencies
- Analyzer needs: LLM Provider, Templates, Document Repository
- KG Builder needs: Completed Analyses, Event Manager
- Vector Store needs: Knowledge Graph
- Web UI needs: All components
- Event Manager needs: EventDatabase, LLM Provider
- Document Aggregator needs: Multiple analyses per document

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
- Events DB: `data/events.db`
- Unified graphs: `data/unified_graph/`

### 3. JSON Parsing
Many components parse LLM JSON responses:
- Extraction logic includes retry mechanisms
- Don't assume clean JSON from LLMs
- Recovery parsing is intentional
- Event data often nested in `facts_data` or `kg_payload`

### 4. Event System
Web dashboard expects specific event names:
- `status_update`
- `collection_progress`
- `analysis_progress`
- Event API endpoints at `/api/events`

### 5. Database Schema
Event tables have specific structure:
- `events` table with weight and diversity metrics
- `event_observations` linking to source documents
- `match_decisions` for LLM-assisted matching

## Safe Development Practices

### When Adding Features
1. Follow existing patterns in similar code
2. Use established helper functions (e.g., `safe_json_load`)
3. Maintain backward compatibility
4. Add to existing structures rather than replacing

### When Modifying Core Logic
1. Understand the full call chain
2. Check what depends on the output format
3. Test the complete pipeline
4. Preserve existing method signatures
5. Check for crypto lineage dependencies

### When Adding New Components
1. Follow the existing component pattern
2. Integrate with the web interface
3. Add to the main controller
4. Update configuration handling
5. Consider event tracking integration

### When Working with Events
1. Use EventSignature for consistent matching
2. Store source references for provenance
3. Calculate weights using multiple factors
4. Preserve event observations across analyses

## Testing Checklist

Before committing changes, verify:
- [ ] Collection still fetches articles
- [ ] Analysis produces expected JSON structure
- [ ] Events are extracted and matched correctly
- [ ] KG builds without errors
- [ ] Event weights calculate properly
- [ ] Vector search returns results
- [ ] Web UI displays data correctly
- [ ] CLI commands work as expected
- [ ] Export functions preserve lineage

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
    data = safe_json_load(filepath, default={})
```

### Database Operations
```python
with self._get_connection() as conn:
    cur = conn.cursor()
    cur.execute(query, params)
    conn.commit()
```

## Anti-Patterns to Avoid

1. **Complex Abstractions**: Don't add unnecessary layers
2. **Breaking Changes**: Modify behavior, don't replace
3. **Assumption Making**: Check if paths/files exist
4. **Silent Failures**: Always log errors
5. **Tight Coupling**: Keep components independent
6. **Ignoring Lineage**: Preserve provenance chains

## Quick Reference

### Key Files and Their Roles
- `Night_Watcher.py` - Main orchestrator
- `config.json` - System configuration
- `*.json` - Analysis templates
- `KG_Taxonomy.csv` - Graph type definitions
- `requirements.txt` - Python dependencies
- `data/events.db` - Event tracking database

### Important Methods
- `ContentCollector.collect_content()` - Fetches articles
- `ContentAnalyzer.analyze_article()` - Runs analysis
- `DocumentRepository.store_document()` - Stores with provenance
- `EventManager.process_event()` - Tracks event observations
- `EventAggregator.aggregate_analyses()` - Consolidates events
- `KnowledgeGraph.add_node/edge()` - Builds graph
- `VectorStore.add_vector()` - Indexes content

### Configuration Keys
- `llm_provider` - LLM backend settings
- `feeds` - RSS source configuration
- `analysis` - Analysis parameters
- `repo_secret` - Repository crypto key

### Event-Related Tables
- `events` - Core event records with weights
- `event_observations` - Individual sightings
- `match_decisions` - LLM matching cache

Remember: The goal is extending functionality without breaking what already works. When in doubt, add new code rather than modifying existing code.
