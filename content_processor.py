"""
Night_watcher Content Processor
Processes raw content into standardized format for analysis with enhanced metadata and content extraction.
"""

import os
import json
import re
import logging
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

from content_repository import ContentRepository

# Configure logging
logger = logging.getLogger(__name__)

class ContentProcessor:
    """Processes raw content into standardized format for analysis with enhanced capabilities"""
    
    def __init__(self, repository: ContentRepository, processed_dir: str = "data/processed"):
        """
        Initialize content processor
        
        Args:
            repository: Content repository for raw content
            processed_dir: Directory for processed content
        """
        self.repository = repository
        self.processed_dir = processed_dir
        self.processed_metadata = {}  # Map of processed_id to metadata
        os.makedirs(self.processed_dir, exist_ok=True)
        self._load_processed_metadata()
        
    def _load_processed_metadata(self):
        """Load metadata for processed content"""
        metadata_path = os.path.join(self.processed_dir, "processed_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.processed_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.processed_metadata)} processed items")
            except Exception as e:
                logger.error(f"Error loading processed metadata: {str(e)}")
                self.processed_metadata = {}
    
    def _save_processed_metadata(self):
        """Save metadata for processed content"""
        metadata_path = os.path.join(self.processed_dir, "processed_metadata.json")
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_metadata, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving processed metadata: {str(e)}")
            return False
        
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
        raw_metadata = self.repository.get_metadata(content_id)
        
        if not raw_content or not raw_metadata:
            error_msg = f"Content not found: {content_id}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Process based on source type
        source_type = raw_metadata.get("source_type")
        processed_content = None
        processing_method = None
        
        if source_type == "rss":
            processed_content = self._process_rss_article(raw_content)
            processing_method = "rss_standard"
        elif source_type == "document":
            processed_content = self._process_document(raw_content)
            processing_method = "document_standard" 
        elif source_type == "api":
            processed_content = self._process_api_response(raw_content)
            processing_method = "api_standard"
        else:
            # Default processing
            processed_content = self._process_generic(raw_content)
            processing_method = "generic_standard"
            
        # Generate processed content ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_id = f"proc_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Create processing metadata
        proc_metadata = {
            "processed_id": processed_id,
            "raw_content_id": content_id,
            "processing_timestamp": datetime.now().isoformat(),
            "processor_version": "1.1.0",
            "processing_method": processing_method,
            "content_hash": self._hash_processed_content(processed_content),
            "source_name": raw_metadata.get("source_name", "Unknown"),
            "source_type": raw_metadata.get("source_type", "unknown"),
            "title": processed_content.get("title", "Untitled"),
            "publication_date": processed_content.get("publication_date", ""),
            "content_type": processed_content.get("content_type", "unknown")
        }
        
        # Combine with processed content
        output = {
            "content": processed_content,
            "metadata": proc_metadata
        }
        
        # Store processed content
        output_path = os.path.join(self.processed_dir, f"{processed_id}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        # Update metadata store
        self.processed_metadata[processed_id] = proc_metadata
        self._save_processed_metadata()
            
        logger.info(f"Processed content {content_id} to {processed_id}")
        return processed_id
    
    def _hash_processed_content(self, content: Dict[str, Any]) -> str:
        """Generate hash for processed content"""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()
        
    def _process_rss_article(self, raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process RSS article content with enhanced cleaning and extraction
        
        Args:
            raw_content: Raw article content
            
        Returns:
            Processed article content
        """
        # Extract core content, clean HTML, normalize formatting
        content = raw_content.get("content", "")
        
        # Enhanced HTML cleaning
        content = self._clean_html_content(content)
        
        # Extract publication date
        pub_date = raw_content.get("published", datetime.now().isoformat())
        
        # Extract key political entities if possible
        political_entities = self._extract_basic_entities(content)
        
        # Determine content focus (political, governmental, etc.)
        content_focus = self._determine_content_focus(
            raw_content.get("title", ""), 
            content
        )
        
        # Create standardized output
        return {
            "title": raw_content.get("title", ""),
            "source": raw_content.get("source", ""),
            "url": raw_content.get("url", ""),
            "publication_date": pub_date,
            "content": content,
            "content_type": "article",
            "bias_label": raw_content.get("bias_label", "unknown"),
            "political_entities": political_entities,
            "content_focus": content_focus,
            "word_count": len(content.split()),
            "processing_notes": []
        }
    
    def _clean_html_content(self, content: str) -> str:
        """
        Clean HTML content with advanced techniques
        
        Args:
            content: HTML content to clean
            
        Returns:
            Cleaned text content
        """
        # Try to use html2text if available for better HTML cleaning
        try:
            import html2text
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.ignore_tables = False
            h.ignore_emphasis = False
            cleaned = h.handle(content)
            
            # Further cleanup
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Normalize multiple newlines
            return cleaned.strip()
        except ImportError:
            # Fallback to basic cleaning if html2text not available
            # Remove HTML tags
            content = re.sub(r'<[^>]+>', ' ', content)
            # Normalize whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            # Fix common HTML entities
            content = content.replace('&nbsp;', ' ')
            content = content.replace('&amp;', '&')
            content = content.replace('&lt;', '<')
            content = content.replace('&gt;', '>')
            content = content.replace('&quot;', '"')
            content = content.replace('&#39;', "'")
            return content
        
    def _process_document(self, raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process document content with metadata extraction
        
        Args:
            raw_content: Raw document content
            
        Returns:
            Processed document content
        """
        # Extract document type
        doc_type = raw_content.get("document_type", "generic")
        
        # Process based on document type
        if doc_type == "pdf":
            return self._process_pdf_document(raw_content)
        elif doc_type == "docx":
            return self._process_word_document(raw_content)
        elif doc_type == "txt":
            return self._process_text_document(raw_content)
        else:
            # Generic document processing
            return {
                "title": raw_content.get("title", ""),
                "source": raw_content.get("source", ""),
                "content": raw_content.get("content", ""),
                "content_type": "document",
                "document_type": doc_type,
                "publication_date": raw_content.get("date", ""),
                "political_entities": self._extract_basic_entities(raw_content.get("content", "")),
                "content_focus": self._determine_content_focus(
                    raw_content.get("title", ""),
                    raw_content.get("content", "")
                ),
                "word_count": len(raw_content.get("content", "").split()),
                "processing_notes": []
            }
            
    def _process_pdf_document(self, raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """Process PDF document (placeholder implementation)"""
        # This would normally use a PDF processing library
        return {
            "title": raw_content.get("title", ""),
            "source": raw_content.get("source", ""),
            "content": raw_content.get("content", ""),
            "content_type": "document",
            "document_type": "pdf",
            "publication_date": raw_content.get("date", ""),
            "political_entities": self._extract_basic_entities(raw_content.get("content", "")),
            "word_count": len(raw_content.get("content", "").split()),
            "processing_notes": ["PDF processing simplified in this version"]
        }
            
    def _process_word_document(self, raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """Process Word document (placeholder implementation)"""
        # This would normally use a DOCX processing library
        return {
            "title": raw_content.get("title", ""),
            "source": raw_content.get("source", ""),
            "content": raw_content.get("content", ""),
            "content_type": "document",
            "document_type": "docx",
            "publication_date": raw_content.get("date", ""),
            "political_entities": self._extract_basic_entities(raw_content.get("content", "")),
            "word_count": len(raw_content.get("content", "").split()),
            "processing_notes": ["DOCX processing simplified in this version"]
        }
            
    def _process_text_document(self, raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """Process plain text document"""
        content = raw_content.get("content", "")
        return {
            "title": raw_content.get("title", ""),
            "source": raw_content.get("source", ""),
            "content": content,
            "content_type": "document",
            "document_type": "txt",
            "publication_date": raw_content.get("date", ""),
            "political_entities": self._extract_basic_entities(content),
            "word_count": len(content.split()),
            "processing_notes": []
        }
        
    def _process_api_response(self, raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process API response content
        
        Args:
            raw_content: Raw API response content
            
        Returns:
            Processed API response content
        """
        # Extract API type
        api_type = raw_content.get("api_type", "generic")
        
        # Extract any nested content
        content = raw_content.get("content", "")
        if isinstance(content, dict):
            # Handle nested JSON
            content = json.dumps(content, indent=2)
        elif isinstance(content, list):
            # Handle list of items
            content = json.dumps(content, indent=2)
            
        # Extract relevant information based on API type
        if api_type == "twitter":
            return self._process_twitter_api(raw_content)
        elif api_type == "government":
            return self._process_government_api(raw_content)
        else:
            # Generic API processing
            return {
                "title": raw_content.get("title", "API Response"),
                "source": raw_content.get("source", "API"),
                "content": content,
                "content_type": "api_response",
                "api_type": api_type,
                "publication_date": raw_content.get("timestamp", datetime.now().isoformat()),
                "political_entities": self._extract_basic_entities(content if isinstance(content, str) else ""),
                "processing_notes": []
            }
            
    def _process_twitter_api(self, raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """Process Twitter API response (placeholder implementation)"""
        # This would normally use Twitter-specific processing logic
        content = raw_content.get("content", "")
        if isinstance(content, dict) or isinstance(content, list):
            content = json.dumps(content, indent=2)
            
        return {
            "title": raw_content.get("title", "Twitter API Response"),
            "source": "Twitter",
            "content": content,
            "content_type": "api_response",
            "api_type": "twitter",
            "publication_date": raw_content.get("timestamp", datetime.now().isoformat()),
            "political_entities": self._extract_basic_entities(content if isinstance(content, str) else ""),
            "processing_notes": ["Twitter API processing simplified in this version"]
        }
            
    def _process_government_api(self, raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """Process government API response (placeholder implementation)"""
        # This would normally use government-specific processing logic
        content = raw_content.get("content", "")
        if isinstance(content, dict) or isinstance(content, list):
            content = json.dumps(content, indent=2)
            
        return {
            "title": raw_content.get("title", "Government API Response"),
            "source": raw_content.get("source", "Government API"),
            "content": content,
            "content_type": "api_response",
            "api_type": "government",
            "publication_date": raw_content.get("timestamp", datetime.now().isoformat()),
            "political_entities": self._extract_basic_entities(content if isinstance(content, str) else ""),
            "processing_notes": ["Government API processing simplified in this version"]
        }
        
    def _process_generic(self, raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generic content processing for unknown content types
        
        Args:
            raw_content: Raw content to process
            
        Returns:
            Processed generic content
        """
        content = raw_content.get("content", "")
        if not isinstance(content, str):
            if isinstance(content, (dict, list)):
                content = json.dumps(content, indent=2)
            else:
                content = str(content)
                
        return {
            "title": raw_content.get("title", ""),
            "source": raw_content.get("source", ""),
            "content": content,
            "content_type": "generic",
            "publication_date": raw_content.get("date", raw_content.get("timestamp", datetime.now().isoformat())),
            "political_entities": self._extract_basic_entities(content),
            "content_focus": self._determine_content_focus(raw_content.get("title", ""), content),
            "word_count": len(content.split()),
            "processing_notes": ["Generic content processing applied"]
        }
    
    def _extract_basic_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract basic political entities using regex patterns
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of entities with type and context
        """
        if not text or not isinstance(text, str):
            return []
            
        entities = []
        
        # Define entity patterns
        patterns = {
            "person": [
                r'(President|Senator|Representative|Gov\.|Governor|Secretary|Attorney General|Judge)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
                r'(Donald Trump|Joe Biden|Kamala Harris|Mike Pence|Nancy Pelosi|Mitch McConnell|Chuck Schumer|Kevin McCarthy)',
                r'(Justice|Chief Justice)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)'
            ],
            "institution": [
                r'(White House|Congress|Supreme Court|House of Representatives|Senate|Department of \w+|Pentagon|FBI|CIA)',
                r'(Democratic Party|Republican Party|GOP|Dems)',
                r'(Federal Reserve|SEC|FCC|FDA)'
            ],
            "location": [
                r'(Washington D\.C\.|Capitol Hill)',
                r'(United States|America|U\.S\.|USA)'
            ]
        }
        
        # Extract entities
        for entity_type, entity_patterns in patterns.items():
            for pattern in entity_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    # Extract the entity name
                    entity_name = match.group(0)
                    
                    # Get context (30 chars before and after)
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip()
                    
                    # Add entity if not already present
                    entity_exists = False
                    for entity in entities:
                        if entity["name"] == entity_name:
                            entity_exists = True
                            break
                            
                    if not entity_exists:
                        entities.append({
                            "name": entity_name,
                            "type": entity_type,
                            "context": context
                        })
        
        return entities
        
    def _determine_content_focus(self, title: str, content: str) -> List[str]:
        """
        Determine the focus/topic areas of the content
        
        Args:
            title: Content title
            content: Content text
            
        Returns:
            List of focus areas
        """
        focus_areas = []
        
        # Check for various focus areas using keyword detection
        focus_patterns = {
            "governmental": [
                r'(executive order|administration|white house|oval office|president|cabinet)',
                r'(federal|government|policy|regulation|agency)'
            ],
            "legislative": [
                r'(congress|senate|house of representatives|bill|law|legislation)',
                r'(committee|subcommittee|hearing|markup|floor vote|filibuster)'
            ],
            "judicial": [
                r'(supreme court|federal court|judge|justice|ruling|opinion|case|lawsuit)',
                r'(judiciary|appeal|district court|circuit court)'
            ],
            "electoral": [
                r'(election|campaign|vote|voter|polling|ballot|primary|caucus)',
                r'(candidate|presidential|gubernatorial|senate race|house race)'
            ],
            "international": [
                r'(foreign policy|diplomacy|international|global|treaty|alliance)',
                r'(state department|embassy|diplomatic|sanction|summit)'
            ],
            "economic": [
                r'(economy|economic|financial|fiscal|monetary|budget|deficit)',
                r'(tax|taxation|spending|debt|treasury|federal reserve)'
            ],
            "security": [
                r'(security|defense|military|intelligence|terrorism|cyber|threat)',
                r'(pentagon|dod|cia|fbi|nsa|homeland security)'
            ]
        }
        
        # Check title and content
        combined_text = (title + " " + content[:2000]).lower()
        
        for focus_area, patterns in focus_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    focus_areas.append(focus_area)
                    break  # Only add each focus area once
        
        return focus_areas
        
    def get_processed_content(self, processed_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve processed content by ID
        
        Args:
            processed_id: ID of processed content to retrieve
            
        Returns:
            Processed content or None if not found
        """
        processed_path = os.path.join(self.processed_dir, f"{processed_id}.json")
        if not os.path.exists(processed_path):
            return None
            
        try:
            with open(processed_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading processed content {processed_id}: {str(e)}")
            return None
        
    def batch_process(self, content_ids: List[str]) -> List[str]:
        """
        Process multiple content items
        
        Args:
            content_ids: List of content IDs to process
            
        Returns:
            List of processed content IDs
        """
        processed_ids = []
        failed_ids = []
        
        logger.info(f"Batch processing {len(content_ids)} content items")
        
        for content_id in content_ids:
            try:
                processed_id = self.process_content(content_id)
                processed_ids.append(processed_id)
                logger.info(f"Successfully processed {content_id} → {processed_id}")
            except Exception as e:
                failed_ids.append(content_id)
                logger.error(f"Error processing content {content_id}: {str(e)}")
                
        if failed_ids:
            logger.warning(f"Failed to process {len(failed_ids)} content items: {failed_ids}")
            
        return processed_ids
    
    def list_processed_content(self, 
                             raw_content_id: Optional[str] = None,
                             start_date: Optional[Union[str, datetime]] = None,
                             end_date: Optional[Union[str, datetime]] = None,
                             content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List processed content with filtering
        
        Args:
            raw_content_id: Optional filter by original content ID
            start_date: Optional filter for earliest processing date
            end_date: Optional filter for latest processing date
            content_type: Optional filter by content type
            
        Returns:
            List of processed content metadata matching filters
        """
        # Convert dates to ISO string format if they are datetime objects
        if isinstance(start_date, datetime):
            start_date = start_date.isoformat()
        if isinstance(end_date, datetime):
            end_date = end_date.isoformat()
        
        # Start with all metadata
        results = list(self.processed_metadata.values())
        
        # Apply filters
        if raw_content_id:
            results = [item for item in results if item.get("raw_content_id") == raw_content_id]
            
        if start_date:
            results = [item for item in results if item.get("processing_timestamp", "") >= start_date]
            
        if end_date:
            results = [item for item in results if item.get("processing_timestamp", "") <= end_date]
            
        if content_type:
            results = [item for item in results if item.get("content_type") == content_type]
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.get("processing_timestamp", ""), reverse=True)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processor statistics
        
        Returns:
            Dictionary with processor statistics
        """
        sources = {}
        content_types = {}
        dates = {}
        
        # Count by source and type
        for metadata in self.processed_metadata.values():
            source = metadata.get("source_name", "Unknown")
            content_type = metadata.get("content_type", "Unknown")
            
            # Get date (just the day part of the timestamp)
            timestamp = metadata.get("processing_timestamp", "")
            date = timestamp.split("T")[0] if timestamp and "T" in timestamp else "Unknown"
            
            # Update counts
            sources[source] = sources.get(source, 0) + 1
            content_types[content_type] = content_types.get(content_type, 0) + 1
            dates[date] = dates.get(date, 0) + 1
        
        return {
            "total_items": len(self.processed_metadata),
            "sources": sources,
            "content_types": content_types,
            "by_date": dates,
            "timestamp": datetime.now().isoformat()
        }
