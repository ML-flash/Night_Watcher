"""
This file contains the key changes needed for the workflow.py file to integrate the knowledge graph.
These changes should be applied to the existing workflow.py file.
"""

# Add import for KnowledgeGraph
from knowledge_graph import KnowledgeGraph

# Update the NightWatcherWorkflow class initialization
class NightWatcherWorkflow:
    """Manages the Night_watcher workflow focused on intelligence gathering and analysis"""

    def __init__(self, llm_provider: LLMProvider, memory_system: Optional[MemorySystem] = None,
                 output_dir: str = "data"):
        """Initialize workflow with agents and output directory"""
        self.llm_provider = llm_provider
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize memory system if not provided
        self.memory = memory_system or MemorySystem()

        # Get knowledge graph from memory system
        self.knowledge_graph = self.memory.knowledge_graph

        # Initialize agents
        self.collector = ContentCollector()
        self.analyzer = ContentAnalyzer(llm_provider)

        # Set up logging
        self.logger = logging.getLogger("NightWatcherWorkflow")

        # Create output directories
        self._ensure_data_dirs()

# Update the run method to include pattern analysis
def run(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run the Night_watcher intelligence gathering workflow"""
    # ... existing code ...
    
    # 3. Run pattern analysis
    self.logger.info(f"Running pattern analysis over the last {pattern_analysis_days} days...")
    
    # Get recent analyses for pattern recognition
    recent_analyses = self.memory.get_recent_analyses(pattern_analysis_days)
    
    if recent_analyses:
        # Get authoritarian trends from knowledge graph
        auth_trends = self.knowledge_graph.get_authoritarian_trends(pattern_analysis_days)
        save_to_file(
            auth_trends,
            f"{self.analysis_dir}/authoritarian_trends_{self.timestamp}.json"
        )
        
        # Get influential actors
        influential_actors = self.knowledge_graph.get_influential_actors(10)
        save_to_file(
            influential_actors,
            f"{self.analysis_dir}/influential_actors_{self.timestamp}.json"
        )
        
        # Get comprehensive democratic erosion analysis
        democratic_erosion = self.knowledge_graph.analyze_democratic_erosion(pattern_analysis_days)
        save_to_file(
            democratic_erosion,
            f"{self.analysis_dir}/democratic_erosion_{self.timestamp}.json"
        )
        
        # Legacy analyses (keep for backward compatibility)
        topic_analysis = self._analyze_recurring_topics(recent_analyses)
        save_to_file(
            topic_analysis,
            f"{self.analysis_dir}/topic_analysis_{self.timestamp}.json"
        )
        
        actor_analysis = self._analyze_authoritarian_actors(recent_analyses)
        save_to_file(
            actor_analysis,
            f"{self.analysis_dir}/actor_analysis_{self.timestamp}.json"
        )
        
        patterns_generated = 5
    else:
        self.logger.warning("No recent analyses found for pattern recognition")
        patterns_generated = 0
    
    # ... rest of the existing code ...
