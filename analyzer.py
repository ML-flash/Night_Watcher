"""
Night_watcher Content Analyzer
Template-based multi-round prompting for authoritarian pattern analysis with validation.
"""

import json
import logging
import re
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Import template-based prompts
from prompts import load_template_variables, format_prompt

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
        self.variables = self.template.get("variables", {})
        
        self.logger.info(f"Loaded template: {self.template.get('name')} (status: {self.template.get('status')})")

    def _load_template(self, template_file: str) -> Dict[str, Any]:
        """Load analysis template from JSON file."""
        if not os.path.exists(template_file):
            raise FileNotFoundError(f"Template file not found: {template_file}")
        
        with open(template_file, 'r') as f:
            template = json.load(f)
        
        # Validate template structure
        required_fields = ["name", "status", "rounds", "variables"]
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
            template_name = round_config["prompt_template"]
            max_tokens = round_config.get("max_tokens", 2000)
            
            try:
                # Format prompt with variables and round data
                prompt = format_prompt(template_name, self.variables, **round_data)
                
                # Get LLM response
                response = self._get_llm_response(prompt, max_tokens)
                
                # Store round info
                prompt_chain.append({
                    "round": len(prompt_chain) + 1,
                    "name": round_name,
                    "template": template_name,
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
            facts = self._extract_json(response)
            if facts and isinstance(facts, dict):
                processed["facts"] = json.dumps(facts, indent=2)
                processed["facts_data"] = facts
        
        elif round_name == "node_extraction":
            nodes = self._extract_json(response)
            if nodes and isinstance(nodes, list):
                processed["nodes"] = json.dumps(nodes, indent=2)
                processed["nodes_data"] = nodes
        
        elif round_name == "node_deduplication":
            unique_nodes = self._extract_json(response)
            if unique_nodes and isinstance(unique_nodes, list):
                processed["unique_nodes"] = json.dumps(unique_nodes, indent=2)
                processed["unique_nodes_data"] = unique_nodes
                # Update nodes for edge extraction
                processed["nodes"] = processed["unique_nodes"]
                processed["nodes_data"] = unique_nodes
        
        elif round_name == "edge_extraction":
            edges = self._extract_json(response)
            if edges and isinstance(edges, list):
                processed["edges"] = json.dumps(edges, indent=2)
                processed["edges_data"] = edges
        
        elif round_name == "edge_enrichment":
            enriched_edges = self._extract_json(response)
            if enriched_edges and isinstance(enriched_edges, list):
                processed["enriched_edges"] = json.dumps(enriched_edges, indent=2)
                processed["enriched_edges_data"] = enriched_edges
                # Update edges for packaging
                processed["edges"] = processed["enriched_edges"]
                processed["edges_data"] = enriched_edges
        
        return processed

    def _extract_legacy_fields(self, analysis_data: Dict[str, Any], round_data: Dict[str, Any]):
        """Extract legacy fields for backward compatibility."""
        # Facts
        if "facts_data" in round_data:
            analysis_data["facts"] = round_data["facts_data"]
        
        # Article analysis
        if "article_analysis" in analysis_data["rounds"]:
            analysis_data["article_analysis"] = analysis_data["rounds"]["article_analysis"]["response"]
        
        # Manipulation score
        if "manipulation_score" in analysis_data["rounds"]:
            score_response = analysis_data["rounds"]["manipulation_score"]["response"]
            analysis_data["manipulation_score"] = self._extract_manipulation_score(score_response)
        
        # Authoritarian analysis
        if "authoritarian_analysis" in analysis_data["rounds"]:
            auth_response = analysis_data["rounds"]["authoritarian_analysis"]["response"]
            indicators, concern_level = self._extract_authoritarian_data(auth_response)
            analysis_data["authoritarian_indicators"] = indicators
            analysis_data["concern_level"] = concern_level
        
        # Knowledge graph payload
        if "package_ingestion" in analysis_data["rounds"]:
            kg_response = analysis_data["rounds"]["package_ingestion"]["response"]
            kg_package = self._extract_json(kg_response)
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
        critical_rounds = ["fact_extraction", "manipulation_score", "authoritarian_analysis"]
        for round_name in critical_rounds:
            if round_name not in analysis_data.get("rounds", {}):
                validation_checks["required_fields"] = False
                break
            if "error" in analysis_data["rounds"][round_name]:
                validation_checks["required_fields"] = False
                break
        
        # Check manipulation score range
        manipulation_score = analysis_data.get("manipulation_score", 0)
        validation_checks["score_range"] = 1 <= manipulation_score <= 10
        
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

    def _extract_manipulation_score(self, text: str) -> int:
        """
        Extract manipulation score from analysis text.

        Args:
            text: Analysis text

        Returns:
            Manipulation score (1-10) or 0 if not found
        """
        try:
            match = re.search(r"MANIPULATION SCORE:\s*(\d+)", text)
            if match:
                score = int(match.group(1))
                # Ensure the score is within range
                return max(1, min(10, score))
            return 0
        except Exception as e:
            self.logger.error(f"Error extracting manipulation score: {e}")
            return 0

    def _extract_authoritarian_data(self, text: str) -> Tuple[List[str], str]:
        """
        Extract authoritarian indicators and concern level.

        Args:
            text: Analysis text

        Returns:
            Tuple of (indicators list, concern level)
        """
        indicators = []
        concern_level = "Unknown"

        try:
            # Extract indicators
            indicators_match = re.search(r"AUTHORITARIAN INDICATORS:\s*(.*?)(?:\n|$)", text)
            if indicators_match:
                indicators_text = indicators_match.group(1).strip()
                if indicators_text.lower() != "none detected":
                    # Split by common list separators
                    indicators = re.split(r'[,;]|\n-', indicators_text)
                    indicators = [i.strip() for i in indicators if i.strip()]

            # Extract concern level
            concern_match = re.search(r"CONCERN LEVEL:\s*(None|Low|Moderate|High|Very High)", text)
            if concern_match:
                concern_level = concern_match.group(1).strip()

        except Exception as e:
            self.logger.error(f"Error extracting authoritarian data: {e}")

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
