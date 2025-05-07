# analyzer.py
"""
Night_watcher Content Analyzer with Multi-Round Prompting and KG Pipeline
Optimized to minimize context size by passing only necessary JSON between rounds.
"""

import logging
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from prompts import (
    FACT_EXTRACTION_PROMPT,
    ARTICLE_ANALYSIS_PROMPT,
    NODE_EXTRACTION_PROMPT,
    NODE_DEDUPLICATION_PROMPT,
    EDGE_EXTRACTION_PROMPT,
    EDGE_ENRICHMENT_PROMPT,
    PACKAGE_INGESTION_PROMPT
)

logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """Analyzer for political content with multi-round prompting and KG extraction."""

    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
        self.logger = logging.getLogger("ContentAnalyzer")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        articles = input_data.get("articles", [])
        if not articles:
            self.logger.warning("No articles provided for analysis")
            return {"analyses": []}

        analyses: List[Dict[str, Any]] = []
        for article in articles:
            analyses.append(self._analyze_content_multi_round(article))
        self.logger.info(f"Completed analysis of {len(analyses)} articles")
        return {"analyses": analyses}

    def _analyze_content_multi_round(self, article: Dict[str, Any]) -> Dict[str, Any]:
        title = article.get("title", "Untitled")
        content = article.get("content", "")
        
        # Optimize content length - cap full content at 4000 chars for initial rounds
        # This preserves the essential information while reducing token usage
        core_content = self._optimize_content(content, 4000)
        
        self.logger.info(f"Analyzing: {title}")
        prompt_chain: List[Dict[str, Any]] = []

        # Round 1: Fact Extraction (uses optimized article)
        r1p = FACT_EXTRACTION_PROMPT.format(article_content=core_content)
        r1r = self._get_llm_response(r1p, max_tokens=2500)
        prompt_chain.append({"round":1, "name":"Fact Extraction", "prompt":r1p, "response":r1r})
        raw_facts = self._extract_json(r1r) or {}
        facts = raw_facts if isinstance(raw_facts, dict) else {}
        facts_json = json.dumps(facts, indent=2)
        pub_date = facts.get("publication_date", datetime.now().strftime("%Y-%m-%d"))
        
        # Log success/failure for debugging
        if not facts:
            self.logger.warning(f"Failed to extract facts JSON from round 1 for: {title}")
        else:
            self.logger.info(f"Successfully extracted facts for: {title}")

        # Round 2: Article Analysis (uses optimized article)
        r2p = ARTICLE_ANALYSIS_PROMPT.format(article_content=core_content)
        r2r = self._get_llm_response(r2p, max_tokens=2500)
        prompt_chain.append({"round":2, "name":"Article Analysis", "prompt":r2p, "response":r2r})

        # ===== Context Management: Strategic Reduction =====
        # For Round 3, we prefer shorter but more focused content
        # Extract key sections that contain potential entities and facts
        entity_focused_content = self._extract_entity_sections(content, facts)
        
        # Round 3: Node Extraction with targeted context
        r3p = NODE_EXTRACTION_PROMPT.format(article_content=entity_focused_content)
        r3r = self._get_llm_response(r3p, max_tokens=3000)
        prompt_chain.append({"round":3, "name":"Node Extraction", "prompt":r3p, "response":r3r})
        nodes = self._extract_json(r3r) or []
        
        # Fallback if no nodes extracted
        if not nodes:
            self.logger.warning(f"Failed to extract nodes JSON from round 3 for: {title}. Using fallback.")
            # Try extracting with a more robust pattern
            nodes = self._extract_json_robust(r3r) or []
            if nodes:
                self.logger.info("Fallback node extraction successful")
            else:
                self.logger.warning("Fallback node extraction also failed")
                # Create a minimal valid nodes array to continue
                nodes = []
                
        nodes_json = json.dumps(nodes, indent=2)

        # ===== Context Management: Only Pass Essential Data =====
        # Subsequent rounds don't need the article content at all, just the derived data
        
        # Round 4: Node Deduplication - passing only node data, not article content
        r4p = NODE_DEDUPLICATION_PROMPT.format(nodes=nodes_json, publication_date=pub_date)
        r4r = self._get_llm_response(r4p, max_tokens=2500)
        prompt_chain.append({"round":4, "name":"Node Deduplication", "prompt":r4p, "response":r4r})
        unique_nodes = self._extract_json(r4r) or []
        
        # Fallback if no unique nodes
        if not unique_nodes and nodes:
            self.logger.warning(f"Failed to extract unique nodes from round 4. Using original nodes as fallback.")
            unique_nodes = self._add_ids_to_nodes(nodes)
            
        unique_nodes_json = json.dumps(unique_nodes, indent=2)

        # Round 5: Edge Extraction - passing only nodes and essential facts
        # Optimize facts_json to include only required information
        essential_facts = self._extract_essential_facts(facts)
        essential_facts_json = json.dumps(essential_facts, indent=2)
        
        r5p = EDGE_EXTRACTION_PROMPT.format(nodes=unique_nodes_json, facts=essential_facts_json)
        r5r = self._get_llm_response(r5p, max_tokens=3000)
        prompt_chain.append({"round":5, "name":"Edge Extraction", "prompt":r5p, "response":r5r})
        edges = self._extract_json(r5r) or []
        
        # Fallback for edge extraction
        if not edges and unique_nodes:
            self.logger.warning(f"Failed to extract edges from round 5. Using empty edges list.")
            edges = []
            
        edges_json = json.dumps(edges, indent=2)

        # Round 6: Edge Enrichment - only pass edges that need enrichment
        r6p = EDGE_ENRICHMENT_PROMPT.format(edges=edges_json, publication_date=pub_date)
        r6r = self._get_llm_response(r6p, max_tokens=3000)
        prompt_chain.append({"round":6, "name":"Edge Enrichment", "prompt":r6p, "response":r6r})
        enriched_edges = self._extract_json(r6r) or []
        
        # Fallback for edge enrichment
        if not enriched_edges and edges:
            self.logger.warning(f"Failed to extract enriched edges from round 6. Enriching edges manually.")
            enriched_edges = self._enrich_edges(edges, pub_date)
            
        enriched_edges_json = json.dumps(enriched_edges, indent=2)

        # Round 7: Package Ingestion - final compact package
        # Optimize node/edge structure to remove redundant fields for final package
        compact_nodes_json = self._create_compact_json(unique_nodes)
        compact_edges_json = self._create_compact_json(enriched_edges)
        
        r7p = PACKAGE_INGESTION_PROMPT.format(nodes=compact_nodes_json, edges=compact_edges_json)
        r7r = self._get_llm_response(r7p, max_tokens=3000)
        prompt_chain.append({"round":7, "name":"Package Ingestion", "prompt":r7p, "response":r7r})
        package = self._extract_json(r7r) or {"nodes": unique_nodes, "edges": enriched_edges}

        return {
            "article": self._create_article_summary(article),  # Only store essential article data
            "structured_facts": facts,
            "article_analysis": r2r,
            "prompt_chain": prompt_chain,
            "kg_payload": package,
            "timestamp": datetime.now().isoformat()
        }

    def _get_llm_response(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> str:
        try:
            # Append a clear instruction for JSON formatting to help models
            if "JSON" in prompt:
                prompt += "\n\nIMPORTANT: Format your entire response as valid JSON. Do not include explanations outside the JSON. The response should be properly formatted to be directly parsable by JSON.parse()."
            
            resp = self.llm_provider.complete(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            text = resp.get("choices", [{}])[0].get("text", "")
            # Strip off any chain-of-thought in <think> tags
            if "</think>" in text:
                return text.split("</think>", 1)[1].strip()
            return text.strip()
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return f"ERROR: {e}"

    def _extract_json(self, text: str) -> Optional[Any]:
        """Standard JSON extraction with improved error handling"""
        if not text:
            logger.warning("Empty text provided for JSON extraction")
            return None
            
        try:
            # First try to extract JSON from code blocks if present
            code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if code_blocks:
                for block in code_blocks:
                    try:
                        return json.loads(block.strip())
                    except json.JSONDecodeError:
                        continue

            # Then try to extract JSON from the whole text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Next try to find JSON arrays or objects
            candidates = re.findall(r'(?s)(\[.*?\]|\{.*?\})', text)
            for cand in candidates:
                try:
                    return json.loads(cand)
                except json.JSONDecodeError:
                    continue

            # If we still don't have JSON, log a sample of the text
            logger.warning(f"JSON parsing failed for text: {text[:100]}...")
            return None
        except Exception as e:
            logger.warning(f"JSON candidate extraction failed: {e}")
            return None

    def _extract_json_robust(self, text: str) -> Optional[Any]:
        """More aggressive JSON extraction for challenging cases"""
        try:
            # Try to find any JSON-like structure and fix common issues
            # 1. Look for array/object patterns even with some errors
            array_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
            if array_match:
                # Extract and clean the JSON array
                array_text = array_match.group(0)
                # Fix common issues like unquoted keys, extra commas, etc.
                corrected = re.sub(r'(\w+):', r'"\1":', array_text)  # Quote unquoted keys
                corrected = re.sub(r',\s*[\]}]', r'\g<0>', corrected)  # Fix trailing commas
                try:
                    return json.loads(corrected)
                except:
                    pass
                    
            # 2. Try line-by-line reconstruction for arrays
            if '[' in text and ']' in text:
                start_idx = text.find('[')
                end_idx = text.rfind(']') + 1
                if start_idx < end_idx:
                    array_text = text[start_idx:end_idx]
                    # Remove explanatory text inside the array
                    lines = []
                    in_object = False
                    for line in array_text.split('\n'):
                        line = line.strip()
                        if line.startswith('{'):
                            in_object = True
                            lines.append(line)
                        elif line.endswith('}') or line.endswith('},'):
                            in_object = False
                            lines.append(line)
                        elif in_object and ('"' in line or '{' in line or '}' in line):
                            lines.append(line)
                    reconstructed = '\n'.join(lines)
                    # Ensure it's a valid array
                    if not reconstructed.startswith('['):
                        reconstructed = '[' + reconstructed
                    if not reconstructed.endswith(']'):
                        reconstructed = reconstructed + ']'
                    try:
                        return json.loads(reconstructed)
                    except:
                        pass
            
            return None
        except Exception as e:
            logger.warning(f"Robust JSON extraction failed: {e}")
            return None
            
    def _add_ids_to_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add IDs to nodes as a fallback when deduplication fails"""
        result = []
        for i, node in enumerate(nodes, 1):
            node_copy = node.copy()
            node_copy['id'] = i
            result.append(node_copy)
        return result
        
    def _enrich_edges(self, edges: List[Dict[str, Any]], publication_date: str) -> List[Dict[str, Any]]:
        """Add enrichment fields to edges as a fallback"""
        result = []
        for edge in edges:
            edge_copy = edge.copy()
            # Add required fields if missing
            if 'severity' not in edge_copy:
                edge_copy['severity'] = 0.5  # Default medium severity
            if 'is_decayable' not in edge_copy:
                edge_copy['is_decayable'] = True  # Default to true
            if 'reasoning' not in edge_copy:
                edge_copy['reasoning'] = "Automatically enriched as fallback"
            if 'timestamp' not in edge_copy or edge_copy.get('timestamp') == 'N/A':
                edge_copy['timestamp'] = publication_date
            result.append(edge_copy)
        return result
        
    # ===== CONTEXT MANAGEMENT METHODS =====
    
    def _optimize_content(self, content: str, max_length: int = 4000) -> str:
        """Intelligently truncate content to preserve the most important parts"""
        if len(content) <= max_length:
            return content
            
        # Simple approach: Keep first 60% and last 40% of the content
        # This preserves the lead (which typically contains key info) and the conclusion
        first_part_len = int(max_length * 0.6)
        last_part_len = max_length - first_part_len
        
        first_part = content[:first_part_len]
        last_part = content[-last_part_len:] if last_part_len > 0 else ""
        
        return first_part + "..." + last_part
    
    def _extract_entity_sections(self, content: str, facts: Dict[str, Any]) -> str:
        """Extract sections of content likely to contain entities based on facts"""
        if len(content) <= 4000:
            return content
            
        # Extract sentences containing entities/events from facts
        important_sections = []
        
        # Get key phrases from facts to look for in the content
        key_phrases = []
        
        # Add event names
        for event in facts.get("events", []):
            if event.get("name"):
                key_phrases.append(event["name"])
                
        # Add direct quotes - these often mention entities
        for quote in facts.get("direct_quotes", [])[:3]:  # Limit to first 3 quotes
            if len(quote) > 15:  # Only use substantial quotes
                key_phrases.append(quote[:40])  # Use first part of quote
                
        # Add key facts as phrases
        for fact in facts.get("facts", [])[:5]:  # Limit to first 5 facts
            if len(fact) > 15:
                key_phrases.append(fact[:40])  # Use first part of fact
        
        # Split content into paragraphs
        paragraphs = content.split('\n\n')
        
        # Keep paragraphs containing key phrases
        for para in paragraphs:
            for phrase in key_phrases:
                if phrase and len(phrase) > 5 and phrase.lower() in para.lower():
                    important_sections.append(para)
                    break
        
        # Always include the first two paragraphs (typically contain key context)
        if len(paragraphs) > 0 and paragraphs[0] not in important_sections:
            important_sections.insert(0, paragraphs[0])
        if len(paragraphs) > 1 and paragraphs[1] not in important_sections:
            important_sections.insert(1, paragraphs[1])
            
        # Combine and truncate if still too long
        result = "\n\n".join(important_sections)
        if len(result) > 4000:
            return self._optimize_content(result, 4000)
            
        return result
        
    def _extract_essential_facts(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """Extract only the most essential facts needed for edge extraction"""
        if not facts:
            return {}
            
        # Create a streamlined version with only what's needed
        essential = {
            "publication_date": facts.get("publication_date", ""),
            "events": []
        }
        
        # Only keep the first 5 most relevant events
        events = facts.get("events", [])
        essential_events = []
        
        for event in events[:5]:  # Limit to first 5 events
            essential_events.append({
                "name": event.get("name", ""),
                "date": event.get("date", ""),
                "description": event.get("description", "")[:100]  # Truncate long descriptions
            })
            
        essential["events"] = essential_events
        
        # Include a small sample of facts if present
        if "facts" in facts and facts["facts"]:
            essential["facts"] = facts["facts"][:3]  # Just first 3 facts
            
        return essential
        
    def _create_compact_json(self, data: List[Dict[str, Any]]) -> str:
        """Create a compact JSON representation by removing redundant fields"""
        if not data:
            return "[]"
            
        # Use json.dumps with minimal whitespace
        return json.dumps(data, separators=(',', ':'))
        
    def _create_article_summary(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the article with only essential metadata"""
        return {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "source": article.get("source", ""),
            "published": article.get("published", ""),
            "bias_label": article.get("bias_label", ""),
            "content_length": len(article.get("content", "")),
            "id": article.get("id", f"article_{hash(article.get('title', ''))}")
        }
