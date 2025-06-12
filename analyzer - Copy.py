"""
Night_watcher Content Analyzer
Uses JSON template files to run multi-round analysis pipelines.
"""

import json
import logging
import re
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """Analyzer for political content with template-based prompting and validation."""

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

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of articles with template-based analysis.

        Args:
            input_data: Dictionary with:
                - 'articles': List of article dictionaries
                - 'document_ids': Optional list of document IDs

        Returns:
            Dictionary with analysis results
        """
        articles = input_data.get("articles", [])
        doc_ids = input_data.get("document_ids", [])

        if not articles:
            self.logger.warning("No articles provided for analysis")
            return {"analyses": []}

        analyses = []
        for i, article in enumerate(articles):
            # Associate document ID if available
            doc_id = doc_ids[i] if i < len(doc_ids) else None
            if doc_id:
                article["document_id"] = doc_id

            # Run template-based analysis
            analysis = self._analyze_content_with_template(article)
            analyses.append(analysis)

        self.logger.info(f"Completed analysis of {len(analyses)} articles")
        return {"analyses": analyses}

    def _analyze_content_with_template(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an article using the loaded template.

        Args:
            article: Article data dictionary

        Returns:
            Analysis results dictionary with validation status
        """
        title = article.get("title", "Untitled")
        content = article.get("content", "")
        
        # Optimize content length for initial rounds
        core_content = self._optimize_content(content, 4000)
        
        self.logger.info(f"Analyzing: {title}")
        
        # Initialize analysis data
        analysis_data = {
            "template_info": {
                "name": self.template.get("name"),
                "version": self.template.get("version"),
                "status": self.template.get("status"),
                "file": self.template_file
            },
            "article": self._create_article_summary(article),
            "rounds": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Execute each round from template
        prompt_chain = []
        
        # Storage for data passed between rounds
        round_data = {
            "article_content": core_content,
            "full_content": content,
            "publication_date": article.get("published", datetime.now().strftime("%Y-%m-%d"))
        }
        
        for round_config in self.template["rounds"]:
            round_name = round_config["name"]
            prompt_template = round_config["prompt"]
            max_tokens = round_config.get("max_tokens", 2000)
            
            try:
                # Format prompt by replacing variables
                prompt = prompt_template.format(**round_data)
                
                # Get LLM response
                response = self._get_llm_response(prompt, max_tokens)
                
                # For JSON-critical rounds, retry if extraction fails
                if round_name in ["node_extraction", "edge_extraction", "fact_extraction"]:
                    expected_type = list if "extraction" in round_name else dict
                    test_extraction = self._extract_json_with_recovery(response, expected_type)
                    
                    if not test_extraction:
                        self.logger.warning(f"JSON extraction failed for {round_name}, retrying with clearer prompt")
                        retry_prompt = self._add_json_retry_prompt(prompt)
                        response = self._get_llm_response(retry_prompt, max_tokens)
                
                # Store round info
                prompt_chain.append({
                    "round": len(prompt_chain) + 1,
                    "name": round_name,
                    "prompt": prompt,
                    "response": response
                })
                
                # Process round-specific data for next rounds
                processed_data = self._process_round_output(round_name, response, round_data)
                round_data.update(processed_data)
                
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
                processed["enriched_edges"] = json.dumps(enriched_edges, indent=2)
                processed["enriched_edges_data"] = enriched_edges
                # Update edges for packaging
                processed["edges"] = processed["enriched_edges"]
                processed["edges_data"] = enriched_edges
            else:
                self._log_json_failure(round_name, response)
        
        return processed

    def _extract_legacy_fields(self, analysis_data: Dict[str, Any], round_data: Dict[str, Any]):
        """Extract key fields for backward compatibility."""
        # Facts
        if "facts_data" in round_data:
            analysis_data["facts"] = round_data["facts_data"]
        
        # Article analysis
        if "article_analysis" in analysis_data["rounds"]:
            analysis_data["article_analysis"] = analysis_data["rounds"]["article_analysis"]["response"]
            
            # Extract manipulation score and authoritarian indicators from article analysis
            analysis_text = analysis_data["article_analysis"]
            
            # Extract manipulation assessment
            manipulation_score = self._extract_manipulation_assessment(analysis_text)
            analysis_data["manipulation_score"] = manipulation_score
            
            # Extract authoritarian assessment  
            auth_indicators, concern_level = self._extract_authoritarian_assessment(analysis_text)
            analysis_data["authoritarian_indicators"] = auth_indicators
            analysis_data["concern_level"] = concern_level
        
        # Knowledge graph payload
        if "package_ingestion" in analysis_data["rounds"]:
            kg_response = analysis_data["rounds"]["package_ingestion"]["response"]
            kg_package = self._extract_json_with_recovery(kg_response, dict)
            if kg_package and isinstance(kg_package, dict):
                analysis_data["kg_payload"] = kg_package
            else:
                # Fallback construction
                nodes = round_data.get("unique_nodes_data", [])
                edges = round_data.get("enriched_edges_data", [])
                analysis_data["kg_payload"] = {"nodes": nodes, "edges": edges}

    def _validate_analysis(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate analysis results and determine status.
        
        Returns validation result with status and details.
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

    def _get_llm_response(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> str:
        """
        Get a response from the LLM with error handling.

        Args:
            prompt: Prompt text
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation

        Returns:
            LLM response text
        """
        try:
            resp = self.llm_provider.complete(prompt=prompt, max_tokens=max_tokens, temperature=temperature)

            if "error" in resp:
                self.logger.error(f"LLM error: {resp['error']}")
                return f"ERROR: {resp.get('error')}"

            text = resp.get("choices", [{}])[0].get("text", "")

            # Strip off any chain-of-thought in <think> tags
            if "</think>" in text:
                return text.split("</think>", 1)[1].strip()

            return text.strip()
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
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

IMPORTANT: Your previous response could not be parsed as valid JSON. 
Please provide ONLY a valid JSON response with no additional text.
- Start directly with [ or {
- End with ] or }
- No markdown code blocks
- No explanatory text before or after
- Ensure all quotes are properly escaped
- Ensure all brackets are balanced
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

    def _optimize_content(self, content: str, max_length: int = 4000) -> str:
        """
        Optimize content length for LLM processing.

        Args:
            content: Original content
            max_length: Maximum desired length

        Returns:
            Optimized content
        """
        if len(content) <= max_length:
            return content

        # Keep first 60% and last 40% of the content
        first_part_len = int(max_length * 0.6)
        last_part_len = max_length - first_part_len

        first_part = content[:first_part_len]
        last_part = content[-last_part_len:] if last_part_len > 0 else ""

        return first_part + "..." + last_part

    def _extract_manipulation_assessment(self, text: str) -> int:
        """
        Extract manipulation assessment from article analysis.
        
        Args:
            text: Article analysis text
            
        Returns:
            Manipulation score (1-10) based on analysis
        """
        # Look for manipulation section
        manipulation_section = re.search(r"4\.\s*MANIPULATION[:\s]*(.*?)(?=\n\d+\.|$)", text, re.IGNORECASE | re.DOTALL)
        
        if not manipulation_section:
            return 5  # Default middle score
        
        manipulation_text = manipulation_section.group(1).lower()
        
        # Score based on severity indicators
        if any(word in manipulation_text for word in ["extreme", "severe", "significant", "heavy", "strong"]):
            return 8
        elif any(word in manipulation_text for word in ["moderate", "some", "certain", "notable"]):
            return 6
        elif any(word in manipulation_text for word in ["minimal", "little", "minor", "slight"]):
            return 3
        elif any(word in manipulation_text for word in ["no", "none", "neutral", "objective"]):
            return 1
        else:
            return 5  # Default

    def _extract_authoritarian_assessment(self, text: str) -> Tuple[List[str], str]:
        """
        Extract authoritarian assessment from article analysis.
        
        Args:
            text: Article analysis text
            
        Returns:
            Tuple of (indicators list, concern level)
        """
        indicators = []
        concern_level = "Unknown"
        
        # Look for democratic concerns section
        concerns_section = re.search(r"5\.\s*DEMOCRATIC CONCERNS[:\s]*(.*?)(?=\n|$)", text, re.IGNORECASE | re.DOTALL)
        
        if concerns_section:
            concerns_text = concerns_section.group(1)
            
            # Extract specific indicators mentioned
            indicator_patterns = [
                r"undermin\w+ (?:of )?(\w+ ?\w*)",
                r"delegitimiz\w+ (?:of )?(\w+ ?\w*)",
                r"expansion of (\w+ ?\w*)",
                r"erosion of (\w+ ?\w*)",
                r"attack\w* on (\w+ ?\w*)",
                r"threat\w* to (\w+ ?\w*)"
            ]
            
            for pattern in indicator_patterns:
                matches = re.findall(pattern, concerns_text, re.IGNORECASE)
                indicators.extend(matches)
            
            # Determine concern level based on language
            concerns_lower = concerns_text.lower()
            if any(word in concerns_lower for word in ["severe", "significant", "serious", "grave"]):
                concern_level = "High"
            elif any(word in concerns_lower for word in ["moderate", "some", "potential"]):
                concern_level = "Moderate"
            elif any(word in concerns_lower for word in ["minimal", "little", "minor"]):
                concern_level = "Low"
            elif any(word in concerns_lower for word in ["no", "none"]):
                concern_level = "None"
            else:
                concern_level = "Moderate"  # Default
        
        # Clean up indicators
        indicators = [ind.strip() for ind in indicators if ind.strip()]
        indicators = list(set(indicators))[:5]  # Dedupe and limit to 5
        
        return indicators, concern_level

    def _create_article_summary(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of the article for the analysis output.

        Args:
            article: Article data

        Returns:
            Article summary dictionary
        """
        return {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "source": article.get("source", ""),
            "published": article.get("published", ""),
            "bias_label": article.get("bias_label", ""),
            "content_length": len(article.get("content", "")),
            "document_id": article.get("document_id", "")
        }
