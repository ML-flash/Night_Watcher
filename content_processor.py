"""
Night_watcher Content Processor
Processes raw content into standardized format for analysis.
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

from content_repository import ContentRepository


class ContentProcessor:
    """Processes raw content into standardized format for analysis"""
    
    def __init__(self, repository: ContentRepository, processed_dir: str = "data/processed"):
        """
        Initialize content processor
        
        Args:
            repository: Content repository for raw content
            processed_dir: Directory for processed content
        """
        self.repository = repository
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def process_content(self, content_id: str) -> str:
        """
        Process raw content into standardized format
        
        Args:
            content_id: ID of content to process
            
        Returns:
            Processed content ID
        """
        # Retrieve raw content
        raw_content = self.repository.get_content(content_id)
        metadata = self.repository.get_metadata(content_id)
        
        if not raw_content or not metadata:
            raise ValueError(f"Content not found: {content_id}")
        
        # Process based on source type
        source_type = metadata.get("source_type")
        processed_content = None
        
        if source_type == "rss":
            processed_content = self._process_rss_article(raw_content)
        elif source_type == "document":
            processed_content = self._process_document(raw_content)
        elif source_type == "api":
            processed_content = self._process_api_response(raw_content)
        else:
            # Default processing
            processed_content = self._process_generic(raw_content)
            
        # Generate processed content ID
        processed_id = f"proc_{content_id}"
        
        # Create processing metadata
        proc_metadata = {
            "raw_content_id": content_id,
            "processed_id": processed_id,
            "processing_timestamp": datetime.now().isoformat(),
            "processor_version": "1.0.0",
            "processing_type": f"{source_type}_standard"
        }
        
        # Combine with processed content
        output = {
            "content": processed_content,
            "metadata": proc_metadata
        }
        
        # Store processed content
        output_path = os.path.join(self.processed_dir, f"{processed_id}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
            
        return processed_id
        
    def _process_rss_article(self, raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """Process RSS article content"""
        # Extract core content, clean HTML, normalize formatting
        content = raw_content.get("content", "")
        
        # Basic HTML cleaning (a more robust solution would use a proper HTML parser)
        content = re.sub(r'<[^>]+>', ' ', content)
        content = re.sub(r'\s+', ' ', content).strip()
        
        return {
            "title": raw_content.get("title", ""),
            "source": raw_content.get("source", ""),
            "url": raw_content.get("url", ""),
            "publication_date": raw_content.get("published", ""),
            "content": content,
            "content_type": "article",
            "bias_label": raw_content.get("bias_label", "unknown")
        }
        
    def _process_document(self, raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """Process document content"""
        # Would handle different document types (PDF, DOCX, etc.)
        # This is a placeholder implementation
        return {
            "title": raw_content.get("title", ""),
            "source": raw_content.get("source", ""),
            "content": raw_content.get("content", ""),
            "content_type": "document",
            "document_type": raw_content.get("document_type", "generic")
        }
        
    def _process_api_response(self, raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """Process API response content"""
        # Extract relevant information from API responses
        return {
            "title": raw_content.get("title", "API Response"),
            "source": raw_content.get("source", "API"),
            "content": raw_content.get("content", ""),
            "content_type": "api_response",
            "api_type": raw_content.get("api_type", "generic")
        }
        
    def _process_generic(self, raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """Generic content processing"""
        return {
            "title": raw_content.get("title", ""),
            "source": raw_content.get("source", ""),
            "content": raw_content.get("content", ""),
            "content_type": "generic"
        }
        
    def get_processed_content(self, processed_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve processed content by ID"""
        processed_path = os.path.join(self.processed_dir, f"{processed_id}.json")
        if not os.path.exists(processed_path):
            return None
            
        with open(processed_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def batch_process(self, content_ids: List[str]) -> List[str]:
        """
        Process multiple content items
        
        Args:
            content_ids: List of content IDs to process
            
        Returns:
            List of processed content IDs
        """
        processed_ids = []
        
        for content_id in content_ids:
            try:
                processed_id = self.process_content(content_id)
                processed_ids.append(processed_id)
            except Exception as e:
                print(f"Error processing content {content_id}: {str(e)}")
                
        return processed_ids
