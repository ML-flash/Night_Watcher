"""
Night_watcher Analysis Registry
Registry for tracking analytical products and their derivation history with enhanced provenance.
"""

import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Union

# Configure logging
logger = logging.getLogger(__name__)

class AnalysisRegistry:
    """Registry for tracking analytical products and their derivation history with enhanced capabilities"""
    
    def __init__(self, base_dir: str = "data/analysis_registry"):
        """
        Initialize analysis registry
        
        Args:
            base_dir: Directory for registry data
        """
        self.base_dir = base_dir
        self.registry_file = os.path.join(base_dir, "registry.json")
        self.registry = {}  # Registry of all analyses
        self.source_index = {}  # Maps source_id to dependent analysis_ids
        self.type_index = {}  # Maps analysis_type to analysis_ids
        self.timestamp_index = {}  # Maps date string (YYYY-MM-DD) to analysis_ids
        self.version_index = {}  # Maps analysis_id to list of version_ids
        self._ensure_directories()
        self._load_registry()
        
    def _ensure_directories(self):
        """Ensure registry directories exist"""
        os.makedirs(self.base_dir, exist_ok=True)
        
    def _load_registry(self):
        """Load existing registry"""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.registry = data.get("registry", {})
                    self.source_index = data.get("source_index", {})
                    self.type_index = data.get("type_index", {})
                    self.timestamp_index = data.get("timestamp_index", {})
                    self.version_index = data.get("version_index", {})
                logger.info(f"Loaded registry with {len(self.registry)} analysis entries")
            except Exception as e:
                logger.error(f"Error loading registry: {str(e)}")
                self.registry = {}
                self.source_index = {}
                self.type_index = {}
                self.timestamp_index = {}
                self.version_index = {}
                
    def _save_registry(self):
        """Save registry to disk"""
        try:
            registry_data = {
                "registry": self.registry,
                "source_index": self.source_index,
                "type_index": self.type_index,
                "timestamp_index": self.timestamp_index,
                "version_index": self.version_index,
                "last_updated": datetime.now().isoformat(),
                "entry_count": len(self.registry)
            }
            
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved registry with {len(self.registry)} entries")
            return True
        except Exception as e:
            logger.error(f"Error saving registry: {str(e)}")
            return False
            
    def register_analysis(self, 
                          analysis_id: str, 
                          analysis_type: str,
                          source_ids: List[str],
                          metadata: Dict[str, Any],
                          version: str = "1.0.0") -> str:
        """
        Register an analytical product and its provenance
        
        Args:
            analysis_id: ID of the analytical product
            analysis_type: Type of analysis (e.g., 'entity_extraction', 'pattern_analysis')
            source_ids: IDs of source content used for this analysis
            metadata: Additional metadata about the analysis
            version: Version of the analysis
            
        Returns:
            Registered analysis ID
        """
        # Ensure analysis_id is unique or generate one if not provided
        if not analysis_id or analysis_id in self.registry:
            analysis_id = f"{analysis_type}_{uuid.uuid4().hex[:12]}"
            
        # Create timestamp
        timestamp = datetime.now().isoformat()
        date_str = timestamp.split("T")[0]  # YYYY-MM-DD
        
        # Create registry entry
        registry_entry = {
            "analysis_id": analysis_id,
            "analysis_type": analysis_type,
            "source_ids": source_ids.copy(),
            "registration_timestamp": timestamp,
            "date": date_str,
            "version": version,
            "metadata": metadata.copy(),
            "derived_analyses": [],  # Analyses derived from this one
            "citation_count": 0      # Number of times this analysis is cited
        }
        
        # Add to registry
        self.registry[analysis_id] = registry_entry
        
        # Update indexes
        for source_id in source_ids:
            if source_id not in self.source_index:
                self.source_index[source_id] = []
            self.source_index[source_id].append(analysis_id)
            
        if analysis_type not in self.type_index:
            self.type_index[analysis_type] = []
        self.type_index[analysis_type].append(analysis_id)
        
        if date_str not in self.timestamp_index:
            self.timestamp_index[date_str] = []
        self.timestamp_index[date_str].append(analysis_id)
        
        # Update source entries to include this as derived analysis
        for source_id in source_ids:
            if source_id in self.registry:
                # This source is itself an analysis
                self.registry[source_id]["derived_analyses"].append(analysis_id)
                self.registry[source_id]["citation_count"] += 1
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered analysis {analysis_id} of type {analysis_type}")
        return analysis_id
        
    def get_analysis_provenance(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Get provenance information for an analysis
        
        Args:
            analysis_id: ID of the analysis
            
        Returns:
            Provenance information if found, None otherwise
        """
        # Basic registry entry
        if analysis_id not in self.registry:
            return None
            
        entry = self.registry[analysis_id].copy()
        
        # Add enhanced provenance information
        
        # Get source information
        entry["sources"] = []
        for source_id in entry["source_ids"]:
            if source_id in self.registry:
                # Source is an analysis
                source_entry = self.registry[source_id]
                entry["sources"].append({
                    "id": source_id,
                    "type": source_entry["analysis_type"],
                    "timestamp": source_entry["registration_timestamp"],
                    "is_analysis": True
                })
            else:
                # Source is content
                entry["sources"].append({
                    "id": source_id,
                    "is_analysis": False
                })
        
        # Get derived analysis information
        entry["derived"] = []
        for derived_id in entry["derived_analyses"]:
            if derived_id in self.registry:
                derived_entry = self.registry[derived_id]
                entry["derived"].append({
                    "id": derived_id,
                    "type": derived_entry["analysis_type"],
                    "timestamp": derived_entry["registration_timestamp"]
                })
        
        # Get version history
        entry["versions"] = self.get_version_history(analysis_id)
        
        return entry
        
    def get_derived_analyses(self, source_id: str) -> List[Dict[str, Any]]:
        """
        Get all analyses derived from a specific source
        
        Args:
            source_id: ID of the source content
            
        Returns:
            List of derived analyses
        """
        if source_id not in self.source_index:
            return []
            
        derived = []
        for analysis_id in self.source_index[source_id]:
            if analysis_id in self.registry:
                derived.append(self.registry[analysis_id])
                
        return derived
        
    def find_analyses_by_type(self, analysis_type: str) -> List[Dict[str, Any]]:
        """
        Find all analyses of a specific type
        
        Args:
            analysis_type: Type of analysis to find
            
        Returns:
            List of matching analyses
        """
        if analysis_type not in self.type_index:
            return []
            
        return [self.registry[analysis_id] for analysis_id in self.type_index[analysis_type] 
                if analysis_id in self.registry]
                
    def find_analyses_by_metadata(self, key: str, value: Any) -> List[Dict[str, Any]]:
        """
        Find analyses by metadata value
        
        Args:
            key: Metadata key to match
            value: Value to match
            
        Returns:
            List of matching analyses
        """
        results = []
        
        for analysis_id, entry in self.registry.items():
            metadata = entry.get("metadata", {})
            if key in metadata and metadata[key] == value:
                results.append(entry)
                
        return results
                
    def register_new_version(self,
                            original_id: str,
                            new_analysis_id: str,
                            version: str,
                            metadata_updates: Dict[str, Any] = None) -> str:
        """
        Register a new version of an existing analysis
        
        Args:
            original_id: ID of the original analysis
            new_analysis_id: ID of the new analysis version
            version: Version identifier
            metadata_updates: Updates to metadata
            
        Returns:
            New analysis ID
        """
        if original_id not in self.registry:
            raise ValueError(f"Original analysis {original_id} not found")
            
        # Get original entry
        original = self.registry[original_id]
        
        # Create new entry based on original
        new_entry = original.copy()
        new_entry["analysis_id"] = new_analysis_id
        new_entry["version"] = version
        new_entry["registration_timestamp"] = datetime.now().isoformat()
        new_entry["date"] = new_entry["registration_timestamp"].split("T")[0]
        new_entry["previous_version"] = original_id
        new_entry["derived_analyses"] = []  # Reset derived analyses for new version
        new_entry["citation_count"] = 0      # Reset citation count
        
        # Update metadata if provided
        if metadata_updates:
            new_entry["metadata"].update(metadata_updates)
            
        # Add to registry
        self.registry[new_analysis_id] = new_entry
        
        # Update indexes
        for source_id in new_entry["source_ids"]:
            if source_id not in self.source_index:
                self.source_index[source_id] = []
            self.source_index[source_id].append(new_analysis_id)
            
        if new_entry["analysis_type"] not in self.type_index:
            self.type_index[new_entry["analysis_type"]] = []
        self.type_index[new_entry["analysis_type"]].append(new_analysis_id)
        
        if new_entry["date"] not in self.timestamp_index:
            self.timestamp_index[new_entry["date"]] = []
        self.timestamp_index[new_entry["date"]].append(new_analysis_id)
        
        # Update version index
        if original_id not in self.version_index:
            self.version_index[original_id] = []
        self.version_index[original_id].append(new_analysis_id)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered new version {version} of analysis {original_id} as {new_analysis_id}")
        return new_analysis_id
        
    def get_version_history(self, analysis_id: str) -> List[Dict[str, Any]]:
        """
        Get version history for an analysis
        
        Args:
            analysis_id: ID of the analysis
            
        Returns:
            List of analysis versions in chronological order
        """
        if analysis_id not in self.registry:
            return []
            
        # Find the root version
        current = self.registry[analysis_id].copy()
        root_id = analysis_id
        
        # Traverse to find root version
        while "previous_version" in current:
            previous_id = current["previous_version"]
            previous = self.registry.get(previous_id)
            if previous:
                root_id = previous_id
                current = previous.copy()
            else:
                break
        
        # Start from root and build history
        history = []
        if root_id in self.registry:
            history.append({
                "analysis_id": root_id,
                "version": self.registry[root_id]["version"],
                "timestamp": self.registry[root_id]["registration_timestamp"]
            })
            
            # Add all versions from version index
            for version_id in self.version_index.get(root_id, []):
                if version_id in self.registry:
                    history.append({
                        "analysis_id": version_id,
                        "version": self.registry[version_id]["version"],
                        "timestamp": self.registry[version_id]["registration_timestamp"]
                    })
            
            # Sort by timestamp
            history.sort(key=lambda x: x["timestamp"])
        
        return history
        
    def get_downstream_chain(self, content_id: str) -> Dict[str, List[str]]:
        """
        Get the full downstream analytical chain from a piece of content
        
        Args:
            content_id: ID of the content
            
        Returns:
            Dictionary mapping analysis types to lists of analysis IDs
        """
        # Find all analyses directly derived from this content
        direct_analyses = self.get_derived_analyses(content_id)
        
        # Group by analysis type
        result = {}
        processed = set()
        
        def process_analysis(analysis):
            if analysis["analysis_id"] in processed:
                return
                
            processed.add(analysis["analysis_id"])
            analysis_type = analysis["analysis_type"]
            
            if analysis_type not in result:
                result[analysis_type] = []
                
            result[analysis_type].append(analysis["analysis_id"])
            
            # Find analyses derived from this one
            derived = self.get_derived_analyses(analysis["analysis_id"])
            for d in derived:
                process_analysis(d)
                
        # Process each direct analysis
        for analysis in direct_analyses:
            process_analysis(analysis)
            
        return result
    
    def get_citation_network(self, max_depth: int = 3) -> Dict[str, Any]:
        """
        Get a network representation of analysis citations
        
        Args:
            max_depth: Maximum depth of citation chain to traverse
            
        Returns:
            Dictionary with nodes and links for network visualization
        """
        nodes = []
        links = []
        processed = set()
        
        # Process all analyses
        for analysis_id, entry in self.registry.items():
            # Add node
            nodes.append({
                "id": analysis_id,
                "type": entry["analysis_type"],
                "timestamp": entry["registration_timestamp"],
                "citations": entry["citation_count"]
            })
            
            # Add links to sources
            for source_id in entry["source_ids"]:
                if source_id in self.registry:
                    links.append({
                        "source": source_id,
                        "target": analysis_id,
                        "type": "source"
                    })
            
            # Add links to derived analyses
            for derived_id in entry["derived_analyses"]:
                links.append({
                    "source": analysis_id,
                    "target": derived_id,
                    "type": "derives"
                })
        
        return {
            "nodes": nodes,
            "links": links,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_analysis_file_path(self, analysis_id: str) -> str:
        """
        Get the file path for an analysis
        
        Args:
            analysis_id: ID of the analysis
            
        Returns:
            File path for the analysis
        """
        if analysis_id not in self.registry:
            raise ValueError(f"Analysis {analysis_id} not found")
            
        entry = self.registry[analysis_id]
        analysis_type = entry["analysis_type"]
        
        # Create nested directory structure based on type and date
        date_str = entry["date"]
        year, month, day = date_str.split("-")
        
        path = os.path.join(self.base_dir, "analyses", analysis_type, year, month, f"{analysis_id}.json")
        
        return path
    
    def store_analysis_file(self, analysis_id: str, data: Dict[str, Any]) -> bool:
        """
        Store analysis data in the filesystem
        
        Args:
            analysis_id: ID of the analysis
            data: Analysis data to store
            
        Returns:
            True if successful, False otherwise
        """
        if analysis_id not in self.registry:
            logger.error(f"Analysis {analysis_id} not found in registry")
            return False
            
        try:
            file_path = self.get_analysis_file_path(analysis_id)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Store data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Stored analysis file for {analysis_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing analysis file for {analysis_id}: {str(e)}")
            return False
    
    def load_analysis_file(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Load analysis data from the filesystem
        
        Args:
            analysis_id: ID of the analysis
            
        Returns:
            Analysis data if found, None otherwise
        """
        if analysis_id not in self.registry:
            logger.warning(f"Analysis {analysis_id} not found in registry")
            return None
            
        try:
            file_path = self.get_analysis_file_path(analysis_id)
            
            if not os.path.exists(file_path):
                logger.warning(f"Analysis file not found: {file_path}")
                return None
                
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading analysis file for {analysis_id}: {str(e)}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics
        
        Returns:
            Dictionary with registry statistics
        """
        analysis_types = {}
        dates = {}
        
        for entry in self.registry.values():
            analysis_type = entry["analysis_type"]
            date = entry["date"]
            
            analysis_types[analysis_type] = analysis_types.get(analysis_type, 0) + 1
            dates[date] = dates.get(date, 0) + 1
        
        # Sort dates chronologically
        sorted_dates = sorted(dates.items())
        date_history = [{"date": date, "count": count} for date, count in sorted_dates]
        
        # Get citation statistics
        citation_stats = {
            "total_citations": sum(entry["citation_count"] for entry in self.registry.values()),
            "most_cited": []
        }
        
        # Get most cited analyses
        sorted_by_citation = sorted(
            self.registry.values(), 
            key=lambda x: x["citation_count"], 
            reverse=True
        )[:10]  # Top 10
        
        citation_stats["most_cited"] = [
            {
                "id": entry["analysis_id"],
                "type": entry["analysis_type"],
                "citations": entry["citation_count"]
            } 
            for entry in sorted_by_citation if entry["citation_count"] > 0
        ]
        
        return {
            "total_analyses": len(self.registry),
            "analysis_types": analysis_types,
            "date_history": date_history,
            "citation_stats": citation_stats,
            "timestamp": datetime.now().isoformat()
        }
