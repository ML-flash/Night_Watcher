# analysis_provenance.py - Provenance tracking for analysis processes

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from provenance import ProvenanceTracker

class AnalysisProvenanceTracker:
    """
    Tracks provenance for analysis processes, linking analysis results
    back to source documents with cryptographic verification.
    """
    
    def __init__(self, base_dir: str = "data/analysis_provenance",
                 provenance_tracker: Optional[ProvenanceTracker] = None,
                 dev_passphrase: str = None,
                 dev_mode: bool = True):
        """
        Initialize the analysis provenance tracker.
        
        Args:
            base_dir: Directory for storing analysis provenance records
            provenance_tracker: Existing ProvenanceTracker instance or None to create new
            dev_passphrase: Development passphrase if creating new tracker
            dev_mode: Whether to use development cryptographic mode
        """
        self.base_dir = base_dir
        self.records_dir = os.path.join(base_dir, "records")
        
        # Create directories if they don't exist
        os.makedirs(self.records_dir, exist_ok=True)
        
        # Use provided tracker or create new one
        if provenance_tracker:
            self.provenance = provenance_tracker
        else:
            provenance_dir = os.path.join(base_dir, "provenance")
            self.provenance = ProvenanceTracker(
                base_dir=provenance_dir,
                dev_passphrase=dev_passphrase,
                dev_mode=dev_mode
            )
    
    def create_analysis_record(self, 
                               analysis_id: str,
                               document_ids: List[str],
                               analysis_type: str,
                               analysis_parameters: Dict[str, Any],
                               results: Dict[str, Any],
                               analyzer_version: str) -> Dict[str, Any]:
        """
        Create a provenance record for an analysis process.
        
        Args:
            analysis_id: Unique ID for the analysis
            document_ids: List of document IDs that were analyzed
            analysis_type: Type of analysis performed
            analysis_parameters: Parameters used for the analysis
            results: Results of the analysis
            analyzer_version: Version of the analyzer component
            
        Returns:
            Analysis provenance record
        """
        # Create analysis record
        timestamp = datetime.now().isoformat()
        
        # Generate result hash
        result_hash = hashlib.sha256(
            json.dumps(results, sort_keys=True).encode("utf-8")
        ).hexdigest()
        
        analysis_record = {
            "analysis_id": analysis_id,
            "timestamp": timestamp,
            "document_ids": document_ids,
            "analysis_type": analysis_type,
            "analysis_parameters": analysis_parameters,
            "result_hash": result_hash,
            "analyzer_version": analyzer_version
        }
        
        # Sign the analysis record
        signature_record = self.provenance.sign_content(
            json.dumps(analysis_record, sort_keys=True),
            {
                "type": "analysis_record",
                "analysis_id": analysis_id,
                "timestamp": timestamp
            }
        )
        
        # Create complete provenance record
        provenance_record = {
            "analysis_record": analysis_record,
            "signature": signature_record
        }
        
        # Save to file
        record_path = os.path.join(self.records_dir, f"{analysis_id}.json")
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(provenance_record, f, indent=2, ensure_ascii=False)
        
        return provenance_record
    
    def verify_analysis_record(self, analysis_id: str) -> Dict[str, Any]:
        """
        Verify an analysis record's integrity.
        
        Args:
            analysis_id: ID of the analysis to verify
            
        Returns:
            Verification result dictionary
        """
        record_path = os.path.join(self.records_dir, f"{analysis_id}.json")
        
        if not os.path.exists(record_path):
            return {
                "verified": False,
                "reason": "Analysis record not found"
            }
        
        try:
            # Load record
            with open(record_path, "r", encoding="utf-8") as f:
                provenance_record = json.load(f)
            
            analysis_record = provenance_record.get("analysis_record", {})
            signature_record = provenance_record.get("signature", {})
            
            # Verify signature
            is_valid = self.provenance.verify_signature(
                json.dumps(analysis_record, sort_keys=True),
                signature_record
            )
            
            if not is_valid:
                return {
                    "verified": False,
                    "reason": "Analysis record signature verification failed"
                }
            
            return {
                "verified": True,
                "analysis_id": analysis_id,
                "timestamp": analysis_record.get("timestamp"),
                "document_count": len(analysis_record.get("document_ids", [])),
                "analysis_type": analysis_record.get("analysis_type")
            }
            
        except Exception as e:
            return {
                "verified": False,
                "reason": f"Error verifying analysis record: {str(e)}"
            }
