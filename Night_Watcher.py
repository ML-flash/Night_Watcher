# Add these methods to the NightWatcher class:

def aggregate_events(self, analysis_window: int = 7) -> Dict[str, Any]:
    """
    Aggregate events from recent analyses.
    
    Args:
        analysis_window: Days of analyses to include
        
    Returns:
        Aggregation results
    """
    from event_aggregator import EventAggregator
    
    # Load recent analyses
    analyses = []
    analyzed_dir = f"{self.base_dir}/analyzed"
    cutoff_date = datetime.now() - timedelta(days=analysis_window)
    
    for filename in os.listdir(analyzed_dir):
        if not filename.startswith("analysis_"):
            continue
            
        filepath = os.path.join(analyzed_dir, filename)
        try:
            # Check file date
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            if mtime < cutoff_date:
                continue
                
            with open(filepath, 'r') as f:
                analysis = json.load(f)
                
            # Only include if it has KG payload
            if analysis.get("kg_payload"):
                analyses.append(analysis)
                
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {e}")
    
    if not analyses:
        return {"status": "no_analyses", "events": []}
    
    # Aggregate events
    aggregator = EventAggregator()
    results = aggregator.process_analysis_batch(analyses)
    
    # Save aggregation results
    output_file = f"{self.base_dir}/analyzed/event_aggregation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    self.logger.info(f"Aggregated {results['event_count']} unique events from {len(analyses)} analyses")
    
    # Create event-centric knowledge graph nodes
    event_nodes = aggregator.create_event_nodes()
    event_edges = aggregator.create_event_relationships()
    
    # Add to knowledge graph
    for node in event_nodes:
        self.knowledge_graph.graph.add_node(node["id"], **node)
    
    for edge in event_edges:
        self.knowledge_graph.graph.add_edge(
            edge["source_id"], 
            edge["target_id"],
            relation=edge["relation"],
            **edge.get("attributes", {})
        )
    
    self.knowledge_graph.save_graph()
    
    return {
        "status": "completed",
        "unique_events": results["event_count"],
        "cross_source_events": len(results["cross_source_events"]),
        "analyses_processed": len(analyses)
    }

def build_kg(self) -> Dict[str, Any]:
    """Enhanced KG build that includes event aggregation."""
    # First, do regular KG build from analyses
    regular_result = self._original_build_kg()  # Rename existing method
    
    # Then aggregate events
    event_result = self.aggregate_events()
    
    return {
        **regular_result,
        "event_aggregation": event_result
    }
