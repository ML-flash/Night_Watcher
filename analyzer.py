"""
Night_watcher Content Analyzer - Enhanced
Uses JSON template files to run multi-round analysis pipelines with intelligent context management.
"""

import json
import logging
import re
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """Analyzer for political content with template-based prompting and intelligent token management."""

    def __init__(self, llm_provider, template_file: str = "standard_analysis.json"):
        """
        Initialize with an LLM provider and analysis template.

        Args:
            llm_provider: LLM provider instance for text completion
            template_file: Path to analysis template JSON file
        """
        self.llm_provider = llm_provider
        self.logger = logging.getLogger("ContentAnalyzer")

        # Load analysis template
        self.template_file = template_file
        self.template = self._load_template(template_file)

        # Create logs directory for JSON failures
        os.makedirs("data/logs", exist_ok=True)

        self.logger.info(f"Loaded template: {self.template.get('name')} (status: {self.template.get('status')})")

    def _load_template(self, template_file: str) -> Dict[str, Any]:
        """Load analysis template from JSON file."""
        if not os.path.exists(template_file):
            raise FileNotFoundError(f"Template file not found: {template_file}")

        with open(template_file, 'r') as f:
            template = json.load(f)

        # Validate template structure
        required_fields = ["name", "status", "rounds"]
        for field in required_fields:
            if field not in template:
                raise ValueError(f"Template missing required field: {field}")

        return template

    def get_document_id(self, content: str) -> str:
        """Generate a document ID using SHA-256 hash of the content."""
        import hashlib
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of articles with template-based analysis.

        Args:
            input_data: Dictionary with 'articles' list and 'document_ids' list

        Returns:
            Dictionary with 'analyses' list and metadata
        """
        articles = input_data.get("articles", [])
        document_ids = input_data.get("document_ids", [])

        if not articles:
            return {"analyses": [], "error": "No articles provided"}

        analyses = []

        for i, article in enumerate(articles):
            self.logger.info(f"Analyzing article {i+1}/{len(articles)}: {article.get('title', 'Untitled')[:50]}...")

            try:
                analysis = self.analyze_article(article)

                # Add metadata
                analysis["template_info"] = {
                    "name": self.template.get("name"),
                    "version": self.template.get("version"),
                    "status": self.template.get("status"),
                    "file": self.template_file
                }

                # Add document ID if provided
                if i < len(document_ids):
                    analysis["document_id"] = document_ids[i]

                analyses.append(analysis)

            except Exception as e:
                self.logger.error(f"Error analyzing article {i+1}: {e}")
                analyses.append({
                    "error": str(e),
                    "article_title": article.get("title", "Unknown"),
                    "document_id": document_ids[i] if i < len(document_ids) else None
                })

        return {
            "analyses": analyses,
            "template": self.template.get("name"),
            "processed_at": datetime.now().isoformat(),
            "total_articles": len(articles),
            "successful_analyses": len([a for a in analyses if "error" not in a])
        }

    def analyze_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single article using the template pipeline.

        Args:
            article: Article data with title, content, url, etc.

        Returns:
            Analysis results dictionary
        """
        # Optimize article content for context window
        content = self._optimize_content(article.get("content", ""))

        # Prepare initial data for rounds
        round_data = {
            "article_content": content,
            "article_title": article.get("title", ""),
            "article_url": article.get("url", ""),
            "article_source": article.get("source", ""),
            "published_date": article.get("published", ""),
            "document_id": article.get("document_id", "")  # Ensure this is set
        }

        # Ensure document_id is set
        if not round_data.get("document_id"):
            doc_id = self.get_document_id(article.get("content", ""))
            article["document_id"] = doc_id
            round_data["document_id"] = doc_id

        # Initialize analysis data
        analysis_data = {
            "article": {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "source": article.get("source", ""),
                "published": article.get("published", ""),
                "document_id": article.get("document_id"),
                "content_length": len(content)
            },
            "document_id": article.get("document_id"),
            "rounds": {},
            "analyzed_at": datetime.now().isoformat(),
            "citation_summary": {
                "total_citations": 0,
                "exact_matches": 0,
                "fuzzy_matches": 0,
                "unmatched": 0,
                "avg_confidence": 0.0
            }
        }

        # Track prompt chain for debugging
        prompt_chain = []

        # Execute each round in the template
        for round_config in self.template.get("rounds", []):
            round_name = round_config.get("name", "unknown")
            self.logger.debug(f"Executing round: {round_name}")

            prompt_template = round_config["prompt"]
            # Use template max_tokens, but let the provider calculate optimal tokens
            template_max_tokens = round_config.get("max_tokens", 2000)

            try:
                # Format prompt by replacing variables
                prompt = prompt_template.format(**round_data)

                # Get LLM response with intelligent token management
                response = self._get_llm_response(prompt, template_max_tokens)

                # For JSON-critical rounds, retry if extraction fails
                if round_name in ["node_extraction", "edge_extraction", "fact_extraction"]:
                    expected_type = list if "extraction" in round_name else dict
                    test_extraction = self._extract_json_with_recovery(response, expected_type)

                    if not test_extraction:
                        self.logger.warning(f"JSON extraction failed for {round_name}, retrying with clearer prompt")
                        retry_prompt = self._add_json_retry_prompt(prompt)
                        response = self._get_llm_response(retry_prompt, template_max_tokens)

                # Store round info
                prompt_chain.append({
                    "round": len(prompt_chain) + 1,
                    "name": round_name,
                    "prompt": prompt,
                    "response": response,
                    "template_max_tokens": template_max_tokens
                })

                # Process round-specific data for next rounds
                processed_data = self._process_round_output(round_name, response, round_data)
                round_data.update(processed_data)
                if processed_data.get("citation_summary"):
                    cs = processed_data["citation_summary"]
                    cs_total = analysis_data["citation_summary"]
                    prev_total = cs_total["total_citations"]
                    new_total = prev_total + cs.get("total_citations", 0)
                    cs_total["exact_matches"] += cs.get("exact_matches", 0)
                    cs_total["fuzzy_matches"] += cs.get("fuzzy_matches", 0)
                    cs_total["unmatched"] += cs.get("unmatched", 0)
                    if new_total > 0:
                        prev_conf_sum = cs_total.get("avg_confidence", 0) * prev_total
                        new_conf_sum = cs.get("avg_confidence", 0) * cs.get("total_citations", 0)
                        cs_total["avg_confidence"] = (prev_conf_sum + new_conf_sum) / new_total
                    cs_total["total_citations"] = new_total
                    analysis_data["citation_summary"] = cs_total

                # Store in analysis
                analysis_data["rounds"][round_name] = {
                    "response": response,
                    "processed_data": processed_data
                }

            except Exception as e:
                self.logger.error(f"Error in round {round_name}: {e}")
                analysis_data["rounds"][round_name] = {
                    "error": str(e),
                    "response": f"ERROR: {e}"
                }

        # Extract key results for backward compatibility
        self._extract_legacy_fields(analysis_data, round_data)

        # Add prompt chain for debugging
        analysis_data["prompt_chain"] = prompt_chain

        # Validate analysis
        validation_result = self._validate_analysis(analysis_data)
        analysis_data["validation"] = validation_result

        return analysis_data

    def _process_round_output(self, round_name: str, response: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process round output and prepare data for subsequent rounds."""
        processed = {}

        if round_name == "fact_extraction":
            facts = self._extract_json_with_recovery(response, dict)
            if facts and isinstance(facts, dict):
                processed["facts"] = json.dumps(facts, indent=2)
                processed["facts_data"] = facts
                processed["publication_date"] = facts.get("publication_date", "N/A")
            else:
                self._log_json_failure(round_name, response)

        elif round_name == "article_analysis":
            # Extract manipulation and authoritarian assessments from the analysis
            manipulation_score = self._extract_manipulation_assessment(response)
            auth_indicators, concern_level = self._extract_authoritarian_assessment(response)

            processed["manipulation_score"] = manipulation_score
            processed["authoritarian_indicators"] = auth_indicators
            processed["concern_level"] = concern_level
            processed["article_analysis_text"] = response

        elif round_name == "node_extraction":
            nodes = self._extract_json_with_recovery(response, list)
            if nodes and isinstance(nodes, list):
                processed["nodes"] = json.dumps(nodes, indent=2)
                processed["nodes_data"] = nodes
            else:
                self._log_json_failure(round_name, response)

        elif round_name == "node_deduplication":
            unique_nodes = self._extract_json_with_recovery(response, list)
            if unique_nodes and isinstance(unique_nodes, list):
                processed["unique_nodes"] = json.dumps(unique_nodes, indent=2)
                processed["unique_nodes_data"] = unique_nodes
                # Update nodes for edge extraction
                processed["nodes"] = processed["unique_nodes"]
                processed["nodes_data"] = unique_nodes
            else:
                self._log_json_failure(round_name, response)

        elif round_name == "edge_extraction":
            edges = self._extract_json_with_recovery(response, list)
            if edges and isinstance(edges, list):
                processed["edges"] = json.dumps(edges, indent=2)
                processed["edges_data"] = edges
            else:
                self._log_json_failure(round_name, response)

        elif round_name == "edge_enrichment":
            enriched_edges = self._extract_json_with_recovery(response, list)
            if enriched_edges and isinstance(enriched_edges, list):
                processed["edges"] = json.dumps(enriched_edges, indent=2)
                processed["edges_data"] = enriched_edges
            else:
                self._log_json_failure(round_name, response)

        elif round_name == "package_ingestion":
            kg_payload = self._extract_json_with_recovery(response, dict)
            if kg_payload and isinstance(kg_payload, dict):
                processed["kg_payload"] = kg_payload
            else:
                self._log_json_failure(round_name, response)

        # Add citation metadata if possible
        article_text = current_data.get("article_content", "")
        doc_id = current_data.get("document_id")
        if article_text and doc_id:
            processed = self.add_citations(processed, article_text, doc_id)

        return processed

    def _create_citation_at_position(self, idx: int, length: int, source_text: str, doc_id: str) -> Dict[str, Any]:
        """Create a citation object with context window."""
        para_num = source_text[:idx].count("\n\n") + 1
        sent_num = source_text[:idx].count(".") + 1
        context_start = max(0, idx - 50)
        context_end = min(len(source_text), idx + length + 50)
        return {
            "doc_id": doc_id,
            "source_text": source_text[idx: idx + length],
            "context": source_text[context_start:context_end],
            "paragraph_num": para_num,
            "sentence_num": sent_num,
            "char_start": idx,
            "char_end": idx + length,
            "match_confidence": 1.0,
            "match_type": "exact",
        }

    def _exact_match_citation(self, text: str, source_text: str, doc_id: str):
        idx = source_text.lower().find(text.lower())
        if idx >= 0:
            return self._create_citation_at_position(idx, len(text), source_text, doc_id)
        return None

    def _normalized_match_citation(self, text: str, source_text: str, doc_id: str):
        import re
        normalized_text = re.sub(r"[^\w\s]", "", text.lower()).strip()
        normalized_text = " ".join(normalized_text.split())
        words = [re.escape(w) for w in normalized_text.split()]
        if not words:
            return None
        pattern = r"\b" + r"\W*".join(words) + r"\b"
        match = re.search(pattern, source_text, flags=re.IGNORECASE)
        if match:
            return self._create_citation_at_position(match.start(), match.end() - match.start(), source_text, doc_id)
        return None

    def _partial_entity_match(self, entity_name: str, source_text: str, doc_id: str):
        name_parts = entity_name.split()
        if len(name_parts) > 1:
            last_name = name_parts[-1]
            if len(last_name) > 3:
                citation = self._exact_match_citation(last_name, source_text, doc_id)
                if citation:
                    return citation
        if len(name_parts) > 2:
            short_name = f"{name_parts[0]} {name_parts[-1]}"
            citation = self._exact_match_citation(short_name, source_text, doc_id)
            if citation:
                return citation
        return None

    def _fuzzy_quote_match(self, quote: str, source_text: str, doc_id: str):
        from difflib import SequenceMatcher
        cleaned = quote.strip('"\n ')
        length = len(cleaned)
        if length == 0:
            return None
        best_ratio = 0.0
        best_idx = -1
        for i in range(0, len(source_text) - length + 1):
            window = source_text[i:i + length]
            ratio = SequenceMatcher(None, cleaned.lower(), window.lower()).ratio()
            if ratio > 0.85 and ratio > best_ratio:
                best_ratio = ratio
                best_idx = i
                if ratio > 0.9:
                    break
        if best_idx >= 0:
            citation = self._create_citation_at_position(best_idx, length, source_text, doc_id)
            citation["match_type"] = "fuzzy"
            citation["match_confidence"] = best_ratio
            return citation
        return None

    def create_citation(self, text: str, source_text: str, doc_id: str, match_context: str = None):
        """Create a citation using multiple matching strategies."""
        if not text:
            return None
        citation = self._exact_match_citation(text, source_text, doc_id)
        if citation:
            return citation
        citation = self._normalized_match_citation(text, source_text, doc_id)
        if citation:
            citation["match_type"] = "normalized"
            citation["match_confidence"] = 0.9
            return citation
        if match_context == "entity_name":
            citation = self._partial_entity_match(text, source_text, doc_id)
            if citation:
                citation["match_type"] = "partial"
                citation["match_confidence"] = 0.8
                return citation
        if match_context == "quote":
            citation = self._fuzzy_quote_match(text, source_text, doc_id)
            if citation:
                return citation
        return None

    def find_all_citations(self, text: str, source_text: str, doc_id: str, max_citations: int = 5):
        citations = []
        search_start = 0
        while len(citations) < max_citations:
            idx = source_text.lower().find(text.lower(), search_start)
            if idx == -1:
                break
            citations.append(self._create_citation_at_position(idx, len(text), source_text, doc_id))
            search_start = idx + 1
        return citations

    def add_citations(self, processed_data: Dict[str, Any], source_text: str, doc_id: str) -> Dict[str, Any]:
        """Add citation metadata to extracted items using multiple matching strategies."""
        summary = {
            "total_citations": 0,
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "unmatched": 0,
            "avg_confidence": 0.0
        }
        conf_sum = 0.0

        def create_citation(text: str, context: str = None):
            citation = self.create_citation(text, source_text, doc_id, context)
            return citation

        # Process nodes
        for node in processed_data.get("nodes_data", []):
            citation = None
            if node.get("source_sentence"):
                citation = create_citation(node.get("source_sentence"), "quote")
            if not citation and node.get("name"):
                citation = create_citation(node.get("name"), "entity_name")
            summary["total_citations"] += 1
            if citation:
                node.setdefault("citations", []).append(citation)
                if citation.get("match_type") == "exact":
                    summary["exact_matches"] += 1
                else:
                    summary["fuzzy_matches"] += 1
                conf_sum += citation.get("match_confidence", 1.0)
            else:
                summary["unmatched"] += 1

        for node in processed_data.get("unique_nodes_data", []):
            citation = None
            if node.get("source_sentence"):
                citation = create_citation(node.get("source_sentence"), "quote")
            if not citation and node.get("name"):
                citation = create_citation(node.get("name"), "entity_name")
            summary["total_citations"] += 1
            if citation:
                node.setdefault("citations", []).append(citation)
                if citation.get("match_type") == "exact":
                    summary["exact_matches"] += 1
                else:
                    summary["fuzzy_matches"] += 1
                conf_sum += citation.get("match_confidence", 1.0)
            else:
                summary["unmatched"] += 1

        # Process edges
        for edge in processed_data.get("edges_data", []):
            citation = create_citation(edge.get("evidence_quote"), "quote")
            summary["total_citations"] += 1
            if citation:
                edge.setdefault("citations", []).append(citation)
                if citation.get("match_type") == "exact":
                    summary["exact_matches"] += 1
                else:
                    summary["fuzzy_matches"] += 1
                conf_sum += citation.get("match_confidence", 1.0)
            else:
                summary["unmatched"] += 1

        # Process facts
        facts = processed_data.get("facts_data")
        if isinstance(facts, dict):
            facts_citations = {}
            for k, v in facts.items():
                if isinstance(v, str):
                    citation = create_citation(v, "quote")
                    summary["total_citations"] += 1
                    if citation:
                        facts_citations[k] = citation
                        if citation.get("match_type") == "exact":
                            summary["exact_matches"] += 1
                        else:
                            summary["fuzzy_matches"] += 1
                        conf_sum += citation.get("match_confidence", 1.0)
                    else:
                        summary["unmatched"] += 1
            if facts_citations:
                processed_data["facts_citations"] = facts_citations

        if summary["total_citations"] > 0:
            summary["avg_confidence"] = conf_sum / summary["total_citations"]

        if summary["total_citations"] > 0:
            processed_data["citation_summary"] = summary

        return processed_data

    def _extract_events_from_analysis(self, analysis_data: Dict[str, Any]) -> List[Dict]:
        """Extract event data from analysis for event tracking."""
        events = []

        facts = analysis_data.get("facts_data", {})
        if isinstance(facts, dict):
            for event in facts.get("events", []):
                events.append({
                    "primary_actor": self._extract_primary_actor(event),
                    "action": event.get("name", ""),
                    "date": event.get("date", "N/A"),
                    "location": event.get("location", ""),
                    "description": event.get("description", ""),
                    "context": event.get("description", ""),
                    "citations": analysis_data.get("citation_summary", {})
                })

        kg_payload = analysis_data.get("kg_payload", {})
        for node in kg_payload.get("nodes", []):
            if node.get("node_type") == "event":
                events.append({
                    "primary_actor": self._extract_actor_from_event(node),
                    "action": node.get("name", ""),
                    "date": node.get("timestamp", "N/A"),
                    "location": node.get("attributes", {}).get("location", ""),
                    "description": node.get("name", ""),
                    "context": node.get("source_sentence", ""),
                    "citations": node.get("citations", [])
                })

        return events

    def _extract_primary_actor(self, event: Dict) -> str:
        actors = event.get("actors", [])
        if actors and isinstance(actors, list):
            return actors[0]
        desc = event.get("description", "")
        if " arrested " in desc:
            return desc.split(" arrested ")[0].strip()
        return "Unknown"

    def _extract_actor_from_event(self, node: Dict) -> str:
        attrs = node.get("attributes", {})
        actor = attrs.get("actor") or attrs.get("primary_actor")
        if isinstance(actor, list):
            return actor[0] if actor else "Unknown"
        return actor or "Unknown"

    def _extract_legacy_fields(self, analysis_data: Dict[str, Any], round_data: Dict[str, Any]):
        """Extract legacy fields for backward compatibility."""
        # Copy key fields to top level
        for key in ["facts_data", "nodes_data", "edges_data", "kg_payload"]:
            if key in round_data:
                analysis_data[key] = round_data[key]

    def _validate_analysis(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate analysis results and determine status.

        Args:
            analysis_data: Complete analysis data

        Returns:
            validation result with status and details.
        """
        template_status = self.template.get("status", "TESTING")

        # Basic validation checks
        validation_checks = {
            "has_article": bool(analysis_data.get("article")),
            "has_rounds": bool(analysis_data.get("rounds")),
            "json_parseable": True,  # Already succeeded if we got this far
            "required_fields": True
        }

        # Check for critical fields
        critical_rounds = ["fact_extraction", "article_analysis", "node_extraction"]
        for round_name in critical_rounds:
            if round_name not in analysis_data.get("rounds", {}):
                validation_checks["required_fields"] = False
                break
            if "error" in analysis_data["rounds"][round_name]:
                validation_checks["required_fields"] = False
                break

        # Check for obvious LLM errors
        error_indicators = ["I cannot", "I'm unable", "ERROR:", "FAILED:"]
        has_errors = False
        for round_data in analysis_data.get("rounds", {}).values():
            response = round_data.get("response", "")
            if any(indicator in response for indicator in error_indicators):
                has_errors = True
                break
        validation_checks["no_llm_errors"] = not has_errors

        # Determine status
        all_checks_pass = all(validation_checks.values())

        if template_status == "TESTING":
            # Testing templates always go to review
            status = "REVIEW"
            reason = "Testing template requires human review"
        elif not all_checks_pass:
            status = "FAILED"
            failed_checks = [k for k, v in validation_checks.items() if not v]
            reason = f"Failed validation checks: {', '.join(failed_checks)}"
        else:
            status = "VALID"
            reason = "Passed all validation checks"

        return {
            "status": status,
            "reason": reason,
            "checks": validation_checks,
            "template_status": template_status,
            "validated_at": datetime.now().isoformat()
        }

    def _get_llm_response(self, prompt: str, template_max_tokens: int = 2000, temperature: float = 0.1) -> str:
        """
        Get a response from the LLM with intelligent context management.

        Args:
            prompt: Prompt text
            template_max_tokens: Maximum tokens requested by template
            temperature: Temperature for generation

        Returns:
            LLM response text
        """
        try:
            # Check if provider supports intelligent token management
            if hasattr(self.llm_provider, 'complete') and hasattr(self.llm_provider, 'context_manager'):
                # Use enhanced provider with auto token calculation
                resp = self.llm_provider.complete(
                    prompt=prompt,
                    max_tokens=template_max_tokens,
                    temperature=temperature,
                    auto_adjust_tokens=True
                )
            else:
                # Fallback to basic provider
                resp = self.llm_provider.complete(
                    prompt=prompt,
                    max_tokens=template_max_tokens,
                    temperature=temperature
                )

            if "error" in resp:
                self.logger.error(f"LLM error: {resp['error']}")
                return f"ERROR: {resp.get('error')}"

            text = resp.get("choices", [{}])[0].get("text", "")

            # Strip off any chain-of-thought in <think> tags
            if "</think>" in text:
                return text.split("</think>", 1)[1].strip()

            return text.strip()
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            return f"ERROR: {e}"

    def _extract_json(self, text: str) -> Optional[Any]:
        """
        Extract JSON from text with error handling.

        Args:
            text: Text containing JSON

        Returns:
            Parsed JSON object or None if failed
        """
        if not text:
            return None

        try:
            # Try code blocks first
            code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if code_blocks:
                for block in code_blocks:
                    try:
                        return json.loads(block.strip())
                    except json.JSONDecodeError:
                        continue

            # Try the whole text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Try to find JSON structures
            candidates = re.findall(r'(?s)(\[.*?\]|\{.*?\})', text)
            for cand in candidates:
                try:
                    return json.loads(cand)
                except json.JSONDecodeError:
                    continue

            return None
        except Exception as e:
            self.logger.error(f"Error extracting JSON: {e}")
            return None

    def _extract_json_with_recovery(self, text: str, expected_type: type = None) -> Optional[Any]:
        """
        Enhanced JSON extraction with multiple recovery strategies.

        Args:
            text: Text containing JSON
            expected_type: Expected type (list or dict) for validation

        Returns:
            Parsed JSON object or None if all strategies fail
        """
        if not text:
            return None

        # Strategy 1: Try standard extraction first
        result = self._extract_json(text)
        if result and (expected_type is None or isinstance(result, expected_type)):
            return result

        # Strategy 2: Clean common LLM mistakes
        cleaned_text = text

        # Remove common prefixes LLMs add
        prefixes_to_remove = [
            "Here is the JSON:",
            "Here's the JSON:",
            "JSON:",
            "```json",
            "```"
        ]
        for prefix in prefixes_to_remove:
            if cleaned_text.strip().startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):].strip()

        # Remove trailing backticks
        cleaned_text = cleaned_text.rstrip('`').strip()

        # Fix unclosed arrays/objects
        if cleaned_text.count('[') > cleaned_text.count(']'):
            cleaned_text += ']' * (cleaned_text.count('[') - cleaned_text.count(']'))
        if cleaned_text.count('{') > cleaned_text.count('}'):
            cleaned_text += '}' * (cleaned_text.count('{') - cleaned_text.count('}'))

        # Try parsing cleaned text
        try:
            result = json.loads(cleaned_text)
            if expected_type is None or isinstance(result, expected_type):
                return result
        except:
            pass

        # Strategy 3: Extract JSON-like structures more aggressively
        if expected_type == list:
            # Find the largest valid array
            matches = re.findall(r'\[\s*\{[^]]+\]\s*\]', text, re.DOTALL)
            for match in sorted(matches, key=len, reverse=True):
                try:
                    result = json.loads(match)
                    if isinstance(result, list) and len(result) > 0:
                        return result
                except:
                    continue

        elif expected_type == dict:
            # Find the largest valid object
            matches = re.findall(r'\{[^{}]+\}', text, re.DOTALL)
            for match in sorted(matches, key=len, reverse=True):
                try:
                    result = json.loads(match)
                    if isinstance(result, dict):
                        return result
                except:
                    continue

        # Strategy 4: If we expect specific structure, try to build it
        if expected_type == list and "node_type" in text:
            # Try to extract node-like structures
            nodes = []
            node_pattern = r'"node_type"\s*:\s*"([^"]+)"[^}]*"name"\s*:\s*"([^"]+)"'
            for match in re.finditer(node_pattern, text):
                nodes.append({
                    "node_type": match.group(1),
                    "name": match.group(2),
                    "attributes": {}
                })
            if nodes:
                return nodes

        return None

    def _add_json_retry_prompt(self, original_prompt: str, error_msg: str = None) -> str:
        """
        Add a retry instruction to the prompt when JSON extraction fails.
        """
        retry_instruction = """

CRITICAL: Your previous response could not be parsed as valid JSON. 
You MUST provide ONLY a valid JSON response with no additional text.
- Start your response directly with { or [
- End your response with } or ]
- Do NOT use markdown code blocks (no ```)
- Do NOT include any explanatory text before or after the JSON
- Do NOT repeat content from the input
- Ensure all quotes are properly escaped
- Ensure all brackets are balanced
- Use only standard JSON syntax

STOP IMMEDIATELY after the closing bracket. Do not continue writing.
"""

        if error_msg:
            retry_instruction += f"\nError encountered: {error_msg}\n"

        return original_prompt + retry_instruction

    def _log_json_failure(self, round_name: str, response: str):
        """Log JSON extraction failures for pattern analysis."""
        try:
            failure_log = f"data/logs/json_failures_{round_name}.txt"
            with open(failure_log, "a", encoding="utf-8") as f:
                f.write(f"\n--- {datetime.now().isoformat()} ---\n")
                f.write(f"Length: {len(response)} chars\n")
                f.write("Response:\n")
                f.write(response[:1000])  # First 1000 chars
                if len(response) > 1000:
                    f.write("\n... (truncated) ...\n")
                f.write("\n")
            self.logger.warning(f"JSON extraction failed for {round_name}, logged to {failure_log}")
        except Exception as e:
            self.logger.error(f"Failed to log JSON failure: {e}")

    def _optimize_content(self, content: str, max_length: int = 12000) -> str:
        """
        Optimize content length for LLM processing with deduplication.
        Increased default length to better utilize larger context windows.
        """
        if len(content) <= max_length:
            # Check for repetitive content even in shorter texts
            content = self._deduplicate_content(content)
            return content

        # Try to find natural break points
        sentences = content.split('. ')

        # If we can fit complete sentences, use that
        current_length = 0
        optimized_sentences = []
        seen_sentences = set()  # Track to avoid repetition

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Skip if we've seen this sentence or very similar
            sentence_key = sentence.lower()[:50]  # First 50 chars for similarity check
            if sentence_key in seen_sentences:
                continue
            seen_sentences.add(sentence_key)

            if current_length + len(sentence) + 2 <= max_length:
                optimized_sentences.append(sentence)
                current_length += len(sentence) + 2
            else:
                break

        if optimized_sentences:
            result = '. '.join(optimized_sentences) + '.'
            return self._deduplicate_content(result)

        # Fallback to character truncation with deduplication
        truncated = content[:max_length] + "..."
        return self._deduplicate_content(truncated)

    def _deduplicate_content(self, content: str) -> str:
        """Remove repetitive sections from content."""
        # Split into paragraphs
        paragraphs = content.split('\n\n')

        seen_paragraphs = set()
        unique_paragraphs = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Create a signature for the paragraph (first 100 chars, normalized)
            signature = ''.join(para.lower().split())[:100]

            if signature not in seen_paragraphs:
                seen_paragraphs.add(signature)
                unique_paragraphs.append(para)
            else:
                # If we see repetition, add a note instead
                if len(unique_paragraphs) > 0 and "[Similar content repeated]" not in unique_paragraphs[-1]:
                    unique_paragraphs.append("[Similar content repeated - details omitted]")

        return '\n\n'.join(unique_paragraphs)

    def _extract_manipulation_assessment(self, text: str) -> float:
        """Extract manipulation score from analysis text."""
        # Look for numerical scores
        score_patterns = [
            r'manipulation.*?score.*?(\d+(?:\.\d+)?)',
            r'score.*?(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:out of|/)\s*10'
        ]

        for pattern in score_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[0])
                except:
                    continue

        return 0.0

    def _extract_authoritarian_assessment(self, text: str) -> Tuple[List[str], str]:
        """Extract authoritarian indicators and concern level."""
        indicators = []

        # Look for specific authoritarian patterns
        auth_patterns = [
            "elite capture", "power concentration", "institution capture",
            "narrative weaponization", "legitimacy attack", "opposition suppression",
            "democratic erosion", "authoritarian", "autocratic"
        ]

        for pattern in auth_patterns:
            if pattern.lower() in text.lower():
                indicators.append(pattern)

        # Determine concern level
        if len(indicators) >= 3:
            concern_level = "HIGH"
        elif len(indicators) >= 1:
            concern_level = "MEDIUM"
        else:
            concern_level = "LOW"

        return indicators, concern_level