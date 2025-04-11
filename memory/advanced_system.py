"""
Night_watcher Advanced Memory System
Integrates enhanced memory store, knowledge graph, and pattern recognition.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

from memory.system import MemorySystem
from memory.advanced_store import create_memory_store, BaseAdvancedStore
from memory.knowledge_graph import KnowledgeGraph
from analysis.enhanced_patterns import EnhancedPatternRecognition

logger = logging.getLogger(__name__)


class AdvancedMemorySystem:
    """Enhanced memory system integrating vector store, knowledge graph, and pattern recognition"""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the advanced memory system"""
        self.config = config or {}
        self.logger = logging.getLogger("AdvancedMemorySystem")

        # Initialize core components
        self._initialize_components()

        # Flag for initialization status
        self.initialized = True

    def _initialize_components(self):
        """Initialize all memory and analysis components"""
        # Initialize legacy memory system for backward compatibility
        self.legacy_memory = MemorySystem(
            store_type=self.config.get("store_type", "simple"),
            config=self.config
        )

        # Initialize advanced vector store
        self.store = self._create_advanced_store()

        # Initialize knowledge graph
        use_networkx = self.config.get("use_networkx", True)
        self.knowledge_graph = KnowledgeGraph(use_networkx=use_networkx)

        # Initialize pattern recognition
        self.pattern_recognition = EnhancedPatternRecognition(
            memory_system=self.legacy_memory,
            knowledge_graph=self.knowledge_graph
        )

    def _create_advanced_store(self) -> BaseAdvancedStore:
        """Create the advanced memory store based on configuration"""
        store_type = self.config.get("store_type", "simple")
        host = self.config.get("vector_db_host", ":memory:")
        port = self.config.get("vector_db_port", 6333)

        # Use legacy embedding provider for compatibility
        embedding_provider = self.legacy_memory.embedding_provider

        # Create the store
        store = create_memory_store(
            embedding_provider=embedding_provider,
            store_type=store_type,
            host=host,
            port=port
        )

        # Initialize default collections
        self._initialize_collections(store)

        return store

    def _initialize_collections(self, store: BaseAdvancedStore):
        """Initialize default collections in the store"""
        try:
            # Create collections if they don't exist
            if "articles" not in store.get_collections():
                store.create_collection("articles", description="News articles and their content")

            if "analyses" not in store.get_collections():
                store.create_collection("analyses", description="Content analyses")

            if "counter_narratives" not in store.get_collections():
                store.create_collection("counter_narratives", description="Generated counter-narratives")

            if "reports" not in store.get_collections():
                store.create_collection("reports", description="Generated intelligence reports")

            if "entities" not in store.get_collections():
                store.create_collection("entities", description="Extracted named entities")

            if "patterns" not in store.get_collections():
                store.create_collection("patterns", description="Detected patterns")

            self.logger.info("Initialized all memory collections")
        except Exception as e:
            self.logger.error(f"Error initializing collections: {str(e)}")

    def store_article_analysis(self, analysis_result: Dict[str, Any]) -> str:
        """
        Store an article analysis in the memory system with enhanced organization.

        Args:
            analysis_result: Analysis result dict

        Returns:
            ID of the stored item
        """
        # Store in legacy system for backward compatibility
        legacy_id = self.legacy_memory.store_article_analysis(analysis_result)

        if not legacy_id:
            self.logger.warning("Failed to store in legacy memory system")
            return ""

        try:
            # Extract key information
            article = analysis_result.get("article", {})
            analysis = analysis_result.get("analysis", "")

            if not article or not analysis:
                self.logger.warning("Invalid analysis result format")
                return legacy_id  # Return legacy ID as fallback

            # Generate a unique ID based on legacy ID
            item_id = legacy_id

            # First store the article
            article_id = item_id + "_article"
            article_metadata = {
                "type": "article",
                "title": article.get("title", "Untitled"),
                "source": article.get("source", "Unknown"),
                "url": article.get("url", ""),
                "bias_label": article.get("bias_label", "unknown"),
                "published": article.get("published", ""),
                "analysis_id": item_id,
                "timestamp": analysis_result.get("timestamp", datetime.now().isoformat())
            }

            # Store article in advanced store
            self.store.add_item(
                collection="articles",
                item_id=article_id,
                text=article.get("content", ""),
                metadata=article_metadata
            )

            # Now store the analysis
            # Extract manipulation score and authoritarian elements
            manipulation_score = self._extract_manipulation_score(analysis)
            auth_elements = self.pattern_recognition.extract_authoritarian_elements(analysis)

            analysis_metadata = {
                "type": "analysis",
                "article_id": article_id,
                "title": article.get("title", "Untitled"),
                "source": article.get("source", "Unknown"),
                "url": article.get("url", ""),
                "bias_label": article.get("bias_label", "unknown"),
                "manipulation_score": manipulation_score,
                "authoritarian_score": auth_elements.get("authoritarian_score", 0),
                "indicators": {k: v.get("present", False) for k, v in auth_elements.items()
                              if k in self.pattern_recognition.authoritarian_indicators},
                "timestamp": analysis_result.get("timestamp", datetime.now().isoformat())
            }

            # Store analysis in advanced store
            self.store.add_item(
                collection="analyses",
                item_id=item_id,
                text=analysis,
                metadata=analysis_metadata
            )

            # Process entities and add to knowledge graph
            self._process_entities_for_knowledge_graph(auth_elements, article, analysis_metadata)

            return item_id
        except Exception as e:
            self.logger.error(f"Error storing analysis in advanced system: {str(e)}")
            return legacy_id  # Return legacy ID as fallback

    def _extract_manipulation_score(self, analysis: str) -> int:
        """Extract manipulation score from analysis text"""
        import re
        try:
            if "MANIPULATION SCORE" in analysis:
                score_text = analysis.split("MANIPULATION SCORE:")[1].split("\n")[0]
                # Extract numbers from text
                numbers = [int(s) for s in re.findall(r'\d+', score_text)]
                if numbers:
                    return numbers[0]
            return 0
        except Exception as e:
            self.logger.error(f"Error extracting manipulation score: {str(e)}")
            return 0

    def _process_entities_for_knowledge_graph(self, auth_elements: Dict[str, Any],
                                             article: Dict[str, Any],
                                             analysis_metadata: Dict[str, Any]):
        """
        Process extracted entities to add to the knowledge graph.

        Args:
            auth_elements: Extracted authoritarian elements
            article: Article data
            analysis_metadata: Analysis metadata
        """
        try:
            entities = auth_elements.get("entities", [])

            if not entities:
                return

            # Track which indicators are present
            present_indicators = [
                indicator for indicator, data in auth_elements.items()
                if indicator in self.pattern_recognition.authoritarian_indicators and data.get("present", False)
            ]

            # Add entities to knowledge graph
            for entity in entities:
                entity_name = entity["text"]
                entity_type = entity["type"]

                # Store the entity in the entities collection
                entity_id = f"entity_{entity_name.replace(' ', '_').lower()}"

                entity_metadata = {
                    "type": "entity",
                    "entity_type": entity_type,
                    "name": entity_name,
                    "source": article.get("source", "Unknown"),
                    "timestamp": datetime.now().isoformat()
                }

                # Add to entity collection
                self.store.add_item(
                    collection="entities",
                    item_id=entity_id,
                    text=entity_name,
                    metadata=entity_metadata
                )

                # Add to knowledge graph
                if entity_type == "PERSON":
                    # Add as actor
                    actor_id = self.knowledge_graph.find_or_create_actor(entity_name)

                    # Add relationships to indicators if present
                    for indicator in present_indicators:
                        indicator_id = self.knowledge_graph.find_or_create_entity(
                            indicator.replace("_", " ").title(),
                            self.knowledge_graph.INDICATOR
                        )

                        self.knowledge_graph.add_relationship(
                            actor_id, indicator_id, self.knowledge_graph.DEMONSTRATES,
                            weight=1.0,
                            attributes={
                                "source": article.get("source", "Unknown"),
                                "title": article.get("title", "Untitled"),
                                "date": analysis_metadata.get("timestamp", "")
                            }
                        )
                elif entity_type == "ORG":
                    # Add as organization
                    org_id = self.knowledge_graph.find_or_create_entity(
                        entity_name, self.knowledge_graph.ORGANIZATION
                    )

                    # Could add relationships here as well
        except Exception as e:
            self.logger.error(f"Error processing entities for knowledge graph: {str(e)}")

    def store_counter_narrative(self, narrative: Dict[str, Any], parent_id: str = "") -> str:
        """
        Store a counter-narrative in the advanced memory system.

        Args:
            narrative: Counter-narrative dict
            parent_id: ID of the parent article analysis

        Returns:
            ID of the stored item
        """
        # Store in legacy system for backward compatibility
        legacy_id = self.legacy_memory.store_counter_narrative(narrative, parent_id)

        if not legacy_id:
            self.logger.warning("Failed to store in legacy memory system")
            return ""

        try:
            demographic = narrative.get("demographic", "unknown")
            content = narrative.get("content", "")

            if not content:
                self.logger.warning("Empty counter-narrative content")
                return legacy_id  # Return legacy ID as fallback

            # Generate a unique ID
            item_id = legacy_id

            # Create metadata
            metadata = {
                "type": "counter_narrative",
                "demographic": demographic,
                "parent_id": parent_id,
                "timestamp": narrative.get("timestamp", datetime.now().isoformat())
            }

            # Store in advanced store
            self.store.add_item(
                collection="counter_narratives",
                item_id=item_id,
                text=content,
                metadata=metadata
            )

            return item_id
        except Exception as e:
            self.logger.error(f"Error storing counter-narrative in advanced system: {str(e)}")
            return legacy_id  # Return legacy ID as fallback

    def store_pattern_analysis(self, analysis_type: str, analysis_data: Dict[str, Any]) -> str:
        """
        Store pattern analysis results in the advanced memory system.

        Args:
            analysis_type: Type of pattern analysis (e.g., "authoritarian_trends", "actor_analysis")
            analysis_data: Analysis data

        Returns:
            ID of the stored item
        """
        try:
            # Generate ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            item_id = f"pattern_{analysis_type}_{timestamp}"

            # Convert to string representation
            import json
            text_content = json.dumps(analysis_data, indent=2)

            # Create metadata
            metadata = {
                "type": "pattern_analysis",
                "analysis_type": analysis_type,
                "timestamp": timestamp,
                "risk_level": analysis_data.get("risk_level", "Unknown"),
                "risk_score": analysis_data.get("aggregate_authoritarian_risk", 0)
            }

            # Store in advanced store
            self.store.add_item(
                collection="patterns",
                item_id=item_id,
                text=text_content,
                metadata=metadata
            )

            return item_id
        except Exception as e:
            self.logger.error(f"Error storing pattern analysis: {str(e)}")
            return ""

    def store_intelligence_report(self, report_type: str, report_content: str,
                                 report_data: Dict[str, Any] = None) -> str:
        """
        Store an intelligence report in the advanced memory system.

        Args:
            report_type: Type of report (e.g., "weekly", "actor", "topic")
            report_content: Report text content
            report_data: Additional report data

        Returns:
            ID of the stored item
        """
        try:
            # Generate ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            item_id = f"report_{report_type}_{timestamp}"

            # Create metadata
            metadata = {
                "type": "intelligence_report",
                "report_type": report_type,
                "timestamp": timestamp
            }

            # Add additional data if provided
            if report_data:
                metadata["report_data"] = report_data

            # Store in advanced store
            self.store.add_item(
                collection="reports",
                item_id=item_id,
                text=report_content,
                metadata=metadata
            )

            return item_id
        except Exception as e:
            self.logger.error(f"Error storing intelligence report: {str(e)}")
            return ""

    def find_similar_analyses(self, query: str, limit: int = 5,
                             filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Find analyses similar to the query with enhanced filtering.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters to apply

        Returns:
            List of similar analyses
        """
        # For backwards compatibility, use legacy search first
        legacy_results = self.legacy_memory.find_similar_analyses(query, limit)

        # Then try the advanced search if available
        try:
            advanced_results = self.store.search(
                collection="analyses",
                query=query,
                limit=limit,
                filters=filters
            )

            # Merge results intelligently
            if advanced_results:
                return advanced_results
            else:
                return legacy_results
        except Exception as e:
            self.logger.error(f"Error in advanced search: {str(e)}")
            return legacy_results

    def strategic_query(self, query_type: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a specialized strategic query against the memory system.

        Args:
            query_type: Type of query (e.g., "actor_influence", "topic_evolution")
            parameters: Query parameters

        Returns:
            Query results
        """
        parameters = parameters or {}

        if query_type == "actor_influence":
            # Analyze an actor's influence and authoritarian patterns
            actor_name = parameters.get("actor", "")
            if not actor_name:
                return {"error": "Actor name required"}

            # Get actor network from knowledge graph
            actor_network = self.knowledge_graph.get_actor_network(actor_name)

            # Get actor's authoritarian indicators
            actor_analyses = self.pattern_recognition.analyze_authoritarian_actors()
            actor_data = actor_analyses.get("actor_patterns", {}).get(actor_name, {})

            return {
                "actor": actor_name,
                "network": actor_network,
                "authoritarian_patterns": actor_data,
                "timestamp": datetime.now().isoformat()
            }

        elif query_type == "topic_evolution":
            # Analyze how a topic has evolved over time
            topic = parameters.get("topic", "")
            if not topic:
                return {"error": "Topic required"}

            # Find analyses related to this topic
            topic_analyses = self.find_similar_analyses(
                query=topic,
                limit=20
            )

            # Sort by timestamp
            topic_analyses.sort(
                key=lambda x: x.get("metadata", {}).get("timestamp", ""),
                reverse=False  # Oldest first
            )

            # Extract manipulation and authoritarian scores over time
            timeline = []
            for analysis in topic_analyses:
                metadata = analysis.get("metadata", {})
                timestamp = metadata.get("timestamp", "")

                timeline.append({
                    "timestamp": timestamp,
                    "manipulation_score": metadata.get("manipulation_score", 0),
                    "authoritarian_score": metadata.get("authoritarian_score", 0),
                    "source": metadata.get("source", "Unknown"),
                    "title": metadata.get("title", "Untitled")
                })

            # Get topic-related entities from knowledge graph
            topic_id = self.knowledge_graph.find_or_create_entity(topic, self.knowledge_graph.TOPIC)
            topic_network = self.knowledge_graph.get_topic_narrative_network(topic)

            return {
                "topic": topic,
                "timeline": timeline,
                "narratives": topic_network.get("narratives", []),
                "actors": topic_network.get("actors", []),
                "timestamp": datetime.now().isoformat()
            }

        elif query_type == "authoritarian_trends":
            # Analyze authoritarian trends over time
            lookback_days = parameters.get("lookback_days", 90)

            # Run the existing trend analysis
            trends = self.pattern_recognition.analyze_authoritarian_trend_patterns(lookback_days)

            # Add prediction if requested
            if parameters.get("include_prediction", False):
                prediction = self.pattern_recognition.predict_authoritarian_escalation()
                trends["prediction"] = prediction

            return trends

        elif query_type == "democratic_vulnerability":
            # Analyze vulnerability to authoritarian patterns
            democratic_strength = {}

            # Analyze each indicator area
            for indicator in self.pattern_recognition.authoritarian_indicators:
                # Calculate vulnerability based on indicator presence
                pattern_data = self.pattern_recognition.analyze_authoritarian_trend_patterns()
                indicator_data = pattern_data.get("trend_analysis", {}).get(indicator, {})

                trend_strength = indicator_data.get("trend_strength", 0)
                vulnerability = trend_strength * 10  # 0-10 scale

                democratic_strength[indicator] = {
                    "vulnerability_score": vulnerability,
                    "risk_level": "High" if vulnerability > 7 else
                                "Moderate" if vulnerability > 4 else
                                "Low",
                    "trend_strength": trend_strength,
                    "count": indicator_data.get("count", 0)
                }

            # Calculate overall democratic resilience score (inverse of vulnerability)
            vulnerability_scores = [data["vulnerability_score"] for data in democratic_strength.values()]
            avg_vulnerability = sum(vulnerability_scores) / len(vulnerability_scores) if vulnerability_scores else 0
            resilience_score = 10 - avg_vulnerability

            return {
                "democratic_vulnerability": democratic_strength,
                "overall_vulnerability_score": avg_vulnerability,
                "democratic_resilience_score": resilience_score,
                "resilience_level": "Strong" if resilience_score > 7 else
                                   "Moderate" if resilience_score > 4 else
                                   "Weak",
                "timestamp": datetime.now().isoformat()
            }

        else:
            return {"error": f"Unsupported query type: {query_type}"}

    def get_recent_analyses(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get analyses from the last N days with enhanced metadata.

        Args:
            days: Number of days to look back

        Returns:
            List of recent analyses
        """
        # For backwards compatibility, use legacy function first
        legacy_results = self.legacy_memory.get_recent_analyses(days)

        # Try advanced search if available
        try:
            # Calculate cutoff date
            cutoff = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff.isoformat()

            # Search with time filter
            advanced_results = self.store.search(
                collection="analyses",
                query="",  # Empty query to match all
                limit=100,  # Higher limit for time filtering
                filters={"timestamp": {"gte": cutoff_str}}
            )

            # Return whichever has more results
            if len(advanced_results) >= len(legacy_results):
                return advanced_results
            else:
                return legacy_results
        except Exception as e:
            self.logger.error(f"Error in advanced recent analyses: {str(e)}")
            return legacy_results

    def search_all(self, query: str, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search all content in the memory system with enhanced categorization.

        Args:
            query: Search query
            limit: Maximum number of results per category

        Returns:
            Dictionary of results by type
        """
        # Use legacy search for backwards compatibility
        legacy_results = self.legacy_memory.search_all(query, limit)

        # Run advanced search across collections
        try:
            # Initialize results
            advanced_results = {
                "article_analyses": [],
                "counter_narratives": [],
                "intelligence_reports": [],
                "patterns": [],
                "entities": []
            }

            # Search in each collection
            for collection in ["analyses", "counter_narratives", "reports", "patterns", "entities"]:
                results = self.store.search(
                    collection=collection,
                    query=query,
                    limit=limit
                )

                # Map to appropriate category
                if collection == "analyses":
                    advanced_results["article_analyses"] = results
                elif collection == "counter_narratives":
                    advanced_results["counter_narratives"] = results
                elif collection == "reports":
                    advanced_results["intelligence_reports"] = results
                elif collection == "patterns":
                    advanced_results["patterns"] = results
                elif collection == "entities":
                    advanced_results["entities"] = results

            # Merge with legacy results if needed
            for category in legacy_results:
                if category in advanced_results and not advanced_results[category]:
                    advanced_results[category] = legacy_results[category]

            return advanced_results
        except Exception as e:
            self.logger.error(f"Error in advanced search: {str(e)}")
            return legacy_results

    def generate_intelligence_brief(self, focus: str = "general",
                                   lookback_days: int = 7) -> Dict[str, Any]:
        """
        Generate an intelligence brief based on recent analyses and patterns.

        Args:
            focus: Focus area for the brief (e.g., "general", "actor", "indicator")
            lookback_days: Days to look back

        Returns:
            Intelligence brief data
        """
        # Get recent analyses
        recent_analyses = self.get_recent_analyses(lookback_days)

        # Get recent pattern analyses
        pattern_analyses = self.store.search(
            collection="patterns",
            query="",  # Empty query to match all
            limit=10,
            filters={"type": "pattern_analysis"}
        )

        # Sort by timestamp (newest first)
        pattern_analyses.sort(
            key=lambda x: x.get("metadata", {}).get("timestamp", ""),
            reverse=True
        )

        # Basic brief structure
        brief = {
            "timestamp": datetime.now().isoformat(),
            "lookback_days": lookback_days,
            "focus": focus,
            "analysis_count": len(recent_analyses),
            "pattern_count": len(pattern_analyses),
            "key_findings": []
        }

        # Add key findings based on focus
        if focus == "general":
            # Run trend analysis
            trends = self.pattern_recognition.analyze_authoritarian_trend_patterns(lookback_days)

            # Add aggregate risk
            brief["aggregate_risk"] = trends.get("aggregate_authoritarian_risk", 0)
            brief["risk_level"] = trends.get("risk_level", "Unknown")

            # Add top indicators
            top_indicators = []
            for indicator, data in trends.get("trend_analysis", {}).items():
                if indicator != "authoritarian_score" and data.get("count", 0) > 0:
                    top_indicators.append({
                        "indicator": indicator,
                        "trend_strength": data.get("trend_strength", 0),
                        "count": data.get("count", 0)
                    })

            # Sort by trend strength
            top_indicators.sort(key=lambda x: x["trend_strength"], reverse=True)
            brief["top_indicators"] = top_indicators[:5]  # Top 5

            # Add authoritarian actors
            actors = self.pattern_recognition.analyze_authoritarian_actors(lookback_days)
            brief["top_actors"] = actors.get("top_actors", [])

            # Add brief summary
            brief["summary"] = f"Analysis of {len(recent_analyses)} articles over the past {lookback_days} days " \
                              f"shows an aggregate authoritarian risk level of {brief['risk_level']} " \
                              f"({brief['aggregate_risk']:.1f}/10). "

            if top_indicators:
                brief["summary"] += f"The most concerning indicators are {top_indicators[0]['indicator'].replace('_', ' ')} " \
                                  f"and {top_indicators[1]['indicator'].replace('_', ' ')} " if len(top_indicators) > 1 else ""

            if brief["top_actors"]:
                brief["summary"] += f"Key actors to monitor include {', '.join(brief['top_actors'][:3])}."

            # Add key findings
            for indicator in top_indicators[:3]:
                indicator_name = indicator["indicator"].replace("_", " ").title()
                trend_data = trends.get("trend_analysis", {}).get(indicator["indicator"], {})
                examples = trend_data.get("examples", [])

                if examples:
                    brief["key_findings"].append({
                        "type": "indicator",
                        "indicator": indicator_name,
                        "trend_strength": indicator["trend_strength"],
                        "examples": examples[:2]  # Top 2 examples
                    })

            # Add actor findings
            for actor_name, actor_data in actors.get("actor_patterns", {}).items():
                if actor_data.get("authoritarian_pattern_score", 0) > 6:  # High risk actors
                    brief["key_findings"].append({
                        "type": "actor",
                        "actor": actor_name,
                        "risk_score": actor_data.get("authoritarian_pattern_score", 0),
                        "risk_level": actor_data.get("risk_level", "Unknown"),
                        "indicators": actor_data.get("indicator_counts", {}),
                        "examples": actor_data.get("examples", [])[:2]  # Top 2 examples
                    })

            # Add predictive insight
            prediction = self.pattern_recognition.predict_authoritarian_escalation()
            brief["prediction"] = {
                "risk_trajectory": prediction.get("risk_trajectory", "stable"),
                "predicted_risk": prediction.get("predicted_risk", 0),
                "predicted_risk_level": prediction.get("predicted_risk_level", "Unknown"),
                "key_indicators_to_monitor": [
                    {"indicator": k, "probability": v.get("escalation_probability", 0)}
                    for k, v in prediction.get("indicator_escalation", {}).items()
                    if v.get("escalation_probability", 0) > 0.5
                ][:3]  # Top 3 indicators to monitor
            }

        return brief

    def save(self, path: str) -> bool:
        """
        Save the advanced memory system to disk.

        Args:
            path: Path to save to

        Returns:
            True if save was successful
        """
        # First save legacy system for backward compatibility
        legacy_result = self.legacy_memory.save(path)

        # Save additional components
        base_dir = os.path.dirname(path)
        advanced_path = os.path.join(base_dir, "advanced_memory")
        os.makedirs(advanced_path, exist_ok=True)

        try:
            # Save vector store
            store_path = os.path.join(advanced_path, "vector_store.pkl")
            store_result = self.store.save(store_path)

            # Save knowledge graph
            graph_path = os.path.join(advanced_path, "knowledge_graph.pkl")
            graph_result = self.knowledge_graph.save(graph_path)

            # Save configuration
            config_path = os.path.join(advanced_path, "config.json")
            import json
            with open(config_path, 'w') as f:
                json.dump({
                    "version": "1.0",
                    "timestamp": datetime.now().isoformat(),
                    "store_path": store_path,
                    "graph_path": graph_path,
                    "legacy_path": path
                }, f, indent=2)

            self.logger.info(f"Saved advanced memory system to {advanced_path}")
            return legacy_result and store_result and graph_result
        except Exception as e:
            self.logger.error(f"Error saving advanced memory system: {str(e)}")
            return False

    def load(self, path: str) -> bool:
        """
        Load the advanced memory system from disk.

        Args:
            path: Path to load from

        Returns:
            True if load was successful
        """
        # First load legacy system for backward compatibility
        legacy_result = self.legacy_memory.load(path)

        # Check for advanced components
        base_dir = os.path.dirname(path)
        advanced_path = os.path.join(base_dir, "advanced_memory")
        config_path = os.path.join(advanced_path, "config.json")

        if not os.path.exists(config_path):
            self.logger.warning(f"Advanced memory configuration not found at {config_path}")
            return legacy_result

        try:
            # Load configuration
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Load vector store
            store_path = config.get("store_path")
            if store_path and os.path.exists(store_path):
                store_result = self.store.load(store_path)
            else:
                self.logger.warning(f"Vector store not found at {store_path}")
                store_result = False

            # Load knowledge graph
            graph_path = config.get("graph_path")
            if graph_path and os.path.exists(graph_path):
                graph_result = self.knowledge_graph.load(graph_path)
            else:
                self.logger.warning(f"Knowledge graph not found at {graph_path}")
                graph_result = False

            self.logger.info(f"Loaded advanced memory system from {advanced_path}")
            return legacy_result and store_result and graph_result
        except Exception as e:
            self.logger.error(f"Error loading advanced memory system: {str(e)}")
            return legacy_result