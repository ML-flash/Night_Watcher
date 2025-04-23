"""
Night_watcher Analysis Registry
Registry for tracking analytical products and their derivation history.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Set


class AnalysisRegistry:
    """Registry for tracking analytical products and their derivation history"""
    
    def __init__(self, base_dir: str = "data/analysis_registry"):
        """
        Initialize analysis registry
        
        Args:
            base_dir: Directory for registry data
        """
        self.base_dir = base_dir
        self.registry = {}
        self._ensure_directories()
        self._load_registry()
        
    def _ensure_directories(self):
        """Ensure registry directories exist"""
        os.makedirs(self.base_dir, exist_ok=True)
        
    def _load_registry(self):
        """Load existing registry"""
        registry_path = os.path.join(self.base_dir, "registry.json")
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r', encoding='utf-8') as f:
                    self.registry = json.load(f)
            except Exception as e:
                print(f"Error loading registry: {str(e)}")
                self.registry = {}
                
    def _save_registry(self):
        """Save registry to disk"""
        registry_path = os.path.join(self.base_dir, "registry.json")
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2)
            
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
        timestamp = datetime.now().isoformat()
        registry_entry = {
            "analysis_id": analysis_id,
            "analysis_type": analysis_type,
            "source_ids": source_ids,
            "registration_timestamp": timestamp,
            "version": version,
            "metadata": metadata
        }
        
        self.registry[analysis_id] = registry_entry
        self._save_registry()
        
        return analysis_id
        
    def get_analysis_provenance(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Get provenance information for an analysis
        
        Args:
            analysis_id: ID of the analysis
            
        Returns:
            Provenance information if found, None otherwise
        """
        return self.registry.get(analysis_id)
        
    def get_derived_analyses(self, source_id: str) -> List[Dict[str, Any]]:
        """
        Get all analyses derived from a specific source
        
        Args:
            source_id: ID of the source content
            
        Returns:
            List of derived analyses
        """
        derived = []
        for analysis_id, entry in self.registry.items():
            if source_id in entry.get("source_ids", []):
                derived.append(entry)
                
        return derived
        
    def find_analyses_by_type(self, analysis_type: str) -> List[Dict[str, Any]]:
        """
        Find all analyses of a specific type
        
        Args:
            analysis_type: Type of analysis to find
            
        Returns:
            List of matching analyses
        """
        return [entry for entry in self.registry.values() 
                if entry.get("analysis_type") == analysis_type]
                
    def find_analyses_by_metadata(self, key: str, value: Any) -> List[Dict[str, Any]]:
        """
        Find analyses by metadata value
        
        Args:
            key: Metadata key to match
            value: Value to match
            
        Returns:
            List of matching analyses
        """
        return [entry for entry in self.registry.values() 
                if entry.get("metadata", {}).get(key) == value]
                
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
        original = self.registry.get(original_id)
        if not original:
            raise ValueError(f"Original analysis {original_id} not found")
            
        # Create new entry based on original
        new_entry = original.copy()
        new_entry["analysis_id"] = new_analysis_id
        new_entry["version"] = version
        new_entry["registration_timestamp"] = datetime.now().isoformat()
        new_entry["previous_version"] = original_id
        
        # Update metadata if provided
        if metadata_updates:
            new_entry["metadata"].update(metadata_updates)
            
        # Register new version
        self.registry[new_analysis_id] = new_entry
        self._save_registry()
        
        return new_analysis_id
        
    def get_version_history(self, analysis_id: str) -> List[Dict[str, Any]]:
        """
        Get version history for an analysis
        
        Args:
            analysis_id: ID of the analysis
            
        Returns:
            List of analysis versions in chronological order
        """
        # Find the root version
        current = self.registry.get(analysis_id)
        if not current:
            return []
            
        # Find earlier versions by traversing previous_version links
        history = [current]
        while "previous_version" in current:
            previous_id = current["previous_version"]
            previous = self.registry.get(previous_id)
            if previous:
                history.append(previous)
                current = previous
            else:
                break
                
        # Reverse to get chronological order
        history.reverse()
        
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
