"""
Knowledge Graph Integration
Functions to integrate knowledge graph and entity extraction with the Night_watcher workflow.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from memory.knowledge_graph import KnowledgeGraph
from analysis.entity_extractor import EntityExtractor
from memory.system import MemorySystem
from agents.base import LLMProvider
from utils.io import save_to_file

logger = logging.getLogger(__name__)


class KnowledgeGraphManager:
    """Manages integration between knowledge graph and Night_watcher workflow"""
    
    def __init__(self, llm_provider: LLMProvider, memory_system: MemorySystem,
                 output_dir: str = "data"):
        """
        Initialize the knowledge graph manager
        
        Args:
            llm_provider: LLM provider for entity extraction
            memory_system: Memory system for storage
            output_dir: Output directory for saving results
        """
        self.llm_provider = llm_provider
        self.memory = memory_system
        self.output_dir = output_dir
        
        # Initialize knowledge graph
        self.knowledge_graph = self._initialize_knowledge_graph()
        
        # Initialize entity extractor
        self.entity_extractor = EntityExtractor(llm_provider, self.knowledge_graph)
        
        # Set up logging
        self.logger = logging.getLogger("KnowledgeGraphManager")
        
    def _initialize_knowledge_graph(self) -> KnowledgeGraph:
        """
        Initialize knowledge graph, loading from disk if it exists
        
        Returns:
            Initialized knowledge graph
        """
        # Create knowledge graph
        use_networkx = True
        knowledge_graph = KnowledgeGraph(use_networkx=use_networkx)
        
        # Check if knowledge graph file exists
        graph_path = os.path.join(self.output_dir, "memory", "knowledge_graph.pkl")
        if os.path.exists(graph_path):
            # Load existing graph
            self.logger.info(f"Loading knowledge graph from {graph_path}")
            knowledge_graph.load(graph_path)
        else:
            # Initialize new graph
            self.logger.info("Creating new knowledge graph")
            self._initialize_basic_entities(knowledge_graph)
        
        return knowledge_graph
        
    def _initialize_basic_entities(self, knowledge_graph: KnowledgeGraph):
        """
        Initialize basic entities in a new knowledge graph
        
        Args:
            knowledge_graph: Knowledge graph to initialize
        """
        # Add key institutions
        institutions = [
            {
                "name": "White House",
                "type": "institution",
                "attributes": {
                    "institution_type": "executive",
                    "description": "Office of the President of the United States"
                }
            },
            {
                "name": "Department of Justice",
                "type": "institution",
                "attributes": {
                    "institution_type": "executive",
                    "description": "Federal department responsible for law enforcement"
                }
            },
            {
                "name": "FBI",
                "type": "institution",
                "attributes": {
                    "institution_type": "executive",
                    "description": "Federal Bureau of Investigation"
                }
            },
            {
                "name": "Supreme Court",
                "type": "institution",
                "attributes": {
                    "institution_type": "judicial",
                    "description": "Highest court in the United States"
                }
            },
            {
                "name": "Congress",
                "type": "institution",
                "attributes": {
                    "institution_type": "legislative",
                    "description": "Legislative branch of the federal government"
                }
            },
            {
                "name": "House of Representatives",
                "type": "institution",
                "attributes": {
                    "institution_type": "legislative",
                    "description": "Lower house of Congress",
                    "parent": "Congress"
                }
            },
            {
                "name": "Senate",
                "type": "institution",
                "attributes": {
                    "institution_type": "legislative",
                    "description": "Upper house of Congress",
                    "parent": "Congress"
                }
            }
        ]
        
        # Add key actors
        actors = [
            {
                "name": "Donald Trump",
                "type": "actor",
                "attributes": {
                    "role": "President",
                    "party": "Republican",
                    "description": "45th and 47th President of the United States"
                }
            },
            {
                "name": "Kamala Harris",
                "type": "actor",
                "attributes": {
                    "role": "Former Vice President",
                    "party": "Democrat",
                    "description": "Former Vice President and presidential candidate"
                }
            },
            {
                "name": "Mike Johnson",
                "type": "actor",
                "attributes": {
                    "role": "Speaker of the House",
                    "party": "Republican",
                    "description": "Speaker of the House of Representatives"
                }
            }
        ]
        
        # Add institutional relationships
        inst_relationships = [
            {
                "source": "House of Representatives",
                "source_type": "institution",
                "target": "Congress",
                "target_type": "institution",
                "relation": "part_of",
                "attributes": {"relationship_type": "organizational"}
            },
            {
                "source": "Senate",
                "source_type": "institution",
                "target": "Congress",
                "target_type": "institution",
                "relation": "part_of",
                "attributes": {"relationship_type": "organizational"}
            }
        ]
        
        # Add to knowledge graph
        entity_map = {}
        
        # Add institutions
        for inst in institutions:
            entity_id = knowledge_graph.add_entity(
                inst["type"],
                inst["name"],
                inst["attributes"]
            )
            entity_map[inst["name"]] = entity_id
            
        # Add actors
        for actor in actors:
            entity_id = knowledge_graph.add_entity(
                actor["type"],
                actor["name"],
                actor["attributes"]
            )
            entity_map[actor["name"]] = entity_id
            
        # Add relationships
        for rel in inst_relationships:
            if rel["source"] in entity_map and rel["target"] in entity_map:
                knowledge_graph.add_relationship(
                    entity_map[rel["source"]],
                    entity_map[rel["target"]],
                    rel["relation"],
                    1.0,
                    rel["attributes"]
                )
                
        self.logger.info(f"Initialized knowledge graph with {len(institutions) + len(actors)} entities")
        
    def save_knowledge_graph(self):
        """Save knowledge graph to disk"""
        graph_path = os.path.join(self.output_dir, "memory", "knowledge_graph.pkl")
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        
        success = self.knowledge_graph.save(graph_path)
        if success:
            self.logger.info(f"Saved knowledge graph to {graph_path}")
        else:
            self.logger.error(f"Failed to save knowledge graph to {graph_path}")
            
    def process_analyses(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process analyses to extract entities and relationships
        
        Args:
            analyses: List of analyses to process
            
        Returns:
            Processing results
        """
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.output_dir, "knowledge_graph", timestamp)
        os.makedirs(results_dir, exist_ok=True)
        
        # Process analyses in batch
        self.logger.info(f"Processing {len(analyses)} analyses for entity extraction")
        results = self.entity_extractor.batch_process(analyses)
        
        # Save results
        save_to_file(results, os.path.join(results_dir, "extraction_results.json"))
        
        # Run pattern analysis
        self.logger.info("Analyzing knowledge graph for patterns")
        patterns = self.entity_extractor.extract_patterns()
        
        # Save patterns
        save_to_file(patterns, os.path.join(results_dir, "pattern_analysis.json"))
        
        # Save updated knowledge graph
        self.save_knowledge_graph()
        
        return {
            "extraction_results": results,
            "patterns": patterns,
            "timestamp": timestamp
        }
        
    def generate_intelligence_report(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Generate intelligence report based on knowledge graph
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            Intelligence report data
        """
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.output_dir, "reports", timestamp)
        os.makedirs(report_dir, exist_ok=True)
        
        # Get democratic erosion analysis
        erosion_analysis = self.knowledge_graph.analyze_democratic_erosion(lookback_days)
        
        # Get influential actors
        influential_actors = self.knowledge_graph.get_influential_actors(10)
        
        # Get authoritarian trends
        authoritarian_trends = self.knowledge_graph.get_authoritarian_trends(lookback_days)
        
        # Create intelligence report
        report = {
            "timestamp": timestamp,
            "lookback_days": lookback_days,
            "democratic_erosion": erosion_analysis,
            "influential_actors": influential_actors,
            "authoritarian_trends": authoritarian_trends,
            "risk_assessment": {
                "overall_risk": erosion_analysis.get("erosion_score", 0),
                "risk_level": erosion_analysis.get("risk_level", "Low"),
                "key_concerns": [],
                "assessment_date": datetime.now().isoformat()
            }
        }
        
        # Extract key concerns
        if erosion_analysis:
            # Add institution types at risk
            for inst_type, data in erosion_analysis.get("affected_institution_types", {}).items():
                if data.get("total_weight", 0) > 5:
                    report["risk_assessment"]["key_concerns"].append({
                        "concern_type": "institution_type",
                        "name": inst_type,
                        "severity": min(10, data.get("total_weight", 0) / 2),
                        "institutions": [inst.get("name", "") for inst in data.get("institutions", [])]
                    })
                    
            # Add coordination patterns
            for pattern in erosion_analysis.get("coordination_patterns", []):
                report["risk_assessment"]["key_concerns"].append({
                    "concern_type": "coordination",
                    "target": pattern.get("target_entity", {}).get("name", ""),
                    "actors": [a.get("name", "") for a in pattern.get("actors", [])],
                    "severity": 7.0  # Coordination is a high severity concern
                })
        
        # Save report
        save_to_file(report, os.path.join(report_dir, "intelligence_report.json"))
        
        # Generate text report using LLM
        text_report = self._generate_text_report(report)
        save_to_file(text_report, os.path.join(report_dir, "intelligence_report.txt"))
        
        return {
            "report": report,
            "text_report": text_report,
            "timestamp": timestamp
        }
        
    def _generate_text_report(self, report: Dict[str, Any]) -> str:
        """
        Generate text version of intelligence report using LLM
        
        Args:
            report: Intelligence report data
            
        Returns:
            Text version of report
        """
        # Create prompt for report generation
        risk_level = report.get("risk_assessment", {}).get("risk_level", "Low")
        risk_score = report.get("risk_assessment", {}).get("overall_risk", 0)
        timestamp = report.get("timestamp", "")
        lookback_days = report.get("lookback_days", 30)
        
        # Extract key concerns
        concerns = report.get("risk_assessment", {}).get("key_concerns", [])
        concerns_text = ""
        
        for i, concern in enumerate(concerns):
            if concern.get("concern_type") == "institution_type":
                concerns_text += f"{i+1}. Institutional undermining of {concern.get('name', '')} institutions"
                if concern.get("institutions"):
                    concerns_text += f": {', '.join(concern.get('institutions', []))}"
                concerns_text += f" (Severity: {concern.get('severity', 0):.1f}/10)\n"
            elif concern.get("concern_type") == "coordination":
                concerns_text += f"{i+1}. Coordination pattern targeting {concern.get('target', '')}"
                if concern.get("actors"):
                    concerns_text += f" by {', '.join(concern.get('actors', []))}"
                concerns_text += f" (Severity: {concern.get('severity', 0):.1f}/10)\n"
        
        # Extract influential actors
        actors = report.get("influential_actors", [])
        actors_text = ""
        
        for i, actor in enumerate(actors[:5]):  # Top 5
            actor_name = actor.get("name", "")
            influence = actor.get("influence_score", 0)
            actors_text += f"{i+1}. {actor_name} (Influence: {influence:.2f})\n"
        
        # Extract affected institutions
        institutions = report.get("authoritarian_trends", {}).get("affected_institutions", [])
        institutions_text = ""
        
        for i, inst in enumerate(institutions[:5]):  # Top 5
            inst_name = inst.get("institution", {}).get("name", "")
            impact = inst.get("total_weight", 0)
            institutions_text += f"{i+1}. {inst_name} (Impact: {impact:.1f})\n"
        
        # Create prompt
        prompt = f"""
        Generate a comprehensive intelligence report on authoritarian trends based on this data:
        
        RISK ASSESSMENT:
        - Overall Risk Level: {risk_level} ({risk_score:.1f}/10)
        - Analysis Period: Last {lookback_days} days
        - Report Date: {timestamp}
        
        KEY CONCERNS:
        {concerns_text}
        
        TOP INFLUENTIAL ACTORS:
        {actors_text}
        
        MOST AFFECTED INSTITUTIONS:
        {institutions_text}
        
        Create a professional intelligence report with these sections:
        1. EXECUTIVE SUMMARY: Brief overview of key findings and risk assessment
        2. KEY CONCERN ANALYSIS: Detailed analysis of each key concern
        3. ACTOR ASSESSMENT: Analysis of key actors and their patterns
        4. INSTITUTIONAL IMPACT: Analysis of institutional undermining
        5. RECOMMENDATIONS: Suggested actions to monitor and respond
        
        Focus on factual analysis with a clear, professional tone. Avoid political bias and speculative claims.
        """
        
        try:
            # Call LLM
            response = self.llm_provider.complete(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3
            )
            
            # Extract text
            text_report = response.get("choices", [{}])[0].get("text", "").strip()
            return text_report
            
        except Exception as e:
            self.logger.error(f"Error generating text report: {str(e)}")
            return f"Error generating report: {str(e)}"
