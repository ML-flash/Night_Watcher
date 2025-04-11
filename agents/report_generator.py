"""
Night_watcher Democratic Resilience Report Generator
Agent for generating comprehensive reports on democratic resilience.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Use relative import since base.py is in the same directory
from base import LLMProvider
# You might need to adjust these imports based on your actual project structure
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text import truncate_text
from memory.system import MemorySystem


class DemocraticResilienceReportGenerator:
    """Agent for generating comprehensive reports on democratic resilience"""

    def __init__(self, llm_provider: LLMProvider, memory_system: Optional[MemorySystem] = None):
        """Initialize with LLM provider and memory system"""
        self.llm_provider = llm_provider
        self.name = "DemocraticResilienceReportGenerator"
        self.logger = logging.getLogger(f"{self.name}")
        self.memory = memory_system

    def _call_llm(self, prompt: str, max_tokens: int = 1000,
                  temperature: float = 0.7, stop: Optional[List[str]] = None) -> str:
        """Helper method to call the LLM and extract text response"""
        response = self.llm_provider.complete(prompt, max_tokens, temperature, stop)

        if "error" in response:
            self.logger.error(f"LLM error: {response['error']}")
            return f"Error: {response['error']}"

        try:
            return response["choices"][0]["text"].strip()
        except (KeyError, IndexError) as e:
            self.logger.error(f"Error extracting text from LLM response: {str(e)}")
            return f"Error extracting response: {str(e)}"

    def _compile_indicators_summary(self, recent_analyses: List[Dict[str, Any]]) -> str:
        """
        Compile a summary of authoritarian indicators from recent analyses.
        
        Args:
            recent_analyses: List of recent analyses from memory
            
        Returns:
            Summary text of key indicators
        """
        # Key indicators to track
        indicators = {
            "institutional_undermining": [],
            "democratic_norm_violations": [],
            "media_delegitimization": [],
            "opposition_targeting": [],
            "power_concentration": [],
            "accountability_evasion": [],
            "rule_of_law_undermining": []
        }
        
        # Extract indicator examples from analyses
        for analysis in recent_analyses:
            metadata = analysis.get("metadata", {})
            auth_analysis = analysis.get("text", "")
            
            # If analysis contains authoritarian indicators, extract them
            if "AUTHORITARIAN" in auth_analysis:
                # Check each indicator
                for indicator in indicators.keys():
                    indicator_label = indicator.upper().replace("_", " ")
                    if indicator_label in auth_analysis:
                        # Extract the section
                        start_idx = auth_analysis.find(indicator_label)
                        if start_idx >= 0:
                            end_idx = auth_analysis.find("\n\n", start_idx)
                            if end_idx < 0:  # If no double newline, go to end
                                end_idx = len(auth_analysis)
                                
                            section = auth_analysis[start_idx:end_idx].strip()
                            
                            # Skip if it indicates no evidence
                            negative_patterns = ["not present", "no evidence", "none found", "not identified"]
                            if not any(pat in section.lower() for pat in negative_patterns):
                                indicators[indicator].append({
                                    "source": metadata.get("source", "Unknown"),
                                    "title": metadata.get("title", "Unknown"),
                                    "date": metadata.get("analysis_timestamp", ""),
                                    "text": section
                                })
        
        # Build summary text
        summary = []
        summary.append("KEY AUTHORITARIAN INDICATORS SUMMARY:\n")
        
        for indicator, examples in indicators.items():
            if examples:
                indicator_name = indicator.replace("_", " ").title()
                summary.append(f"{indicator_name} ({len(examples)} instances):")
                
                # Sort by date, newest first
                examples.sort(key=lambda x: x.get("date", ""), reverse=True)
                
                # Include top 3 examples
                for i, example in enumerate(examples[:3]):
                    summary.append(f"- {example['title']} ({example['source']})")
                
                summary.append("")
        
        return "\n".join(summary)
    
    def _summarize_patterns(self, pattern_findings: Dict[str, Any]) -> str:
        """
        Summarize pattern analysis findings.
        
        Args:
            pattern_findings: Dictionary of pattern analysis results
            
        Returns:
            Summary text of pattern analysis
        """
        summary = []
        summary.append("PATTERN ANALYSIS SUMMARY:\n")
        
        # Authoritarian trend patterns
        if "trend_analysis" in pattern_findings:
            trends = pattern_findings["trend_analysis"]
            summary.append("Authoritarian Trend Patterns:")
            
            # Sort indicators by trend strength
            sorted_indicators = sorted(
                [(k, v) for k, v in trends.items() if k != "authoritarian_score"],
                key=lambda x: x[1].get("trend_strength", 0),
                reverse=True
            )
            
            for indicator, data in sorted_indicators[:5]:  # Top 5 trends
                strength = data.get("trend_strength", 0)
                if strength > 0:
                    indicator_name = indicator.replace("_", " ").title()
                    count = data.get("count", 0)
                    trend_desc = "Strong" if strength > 0.6 else "Moderate" if strength > 0.3 else "Weak"
                    summary.append(f"- {indicator_name}: {trend_desc} trend ({count} instances)")
            
            # Overall risk
            if "aggregate_authoritarian_risk" in pattern_findings:
                risk = pattern_findings["aggregate_authoritarian_risk"]
                risk_level = pattern_findings.get("risk_level", "Unknown")
                summary.append(f"\nOverall Authoritarian Risk: {risk:.1f}/10 ({risk_level})")
            
            summary.append("")
        
        # Actor patterns
        if "actor_patterns" in pattern_findings:
            actors = pattern_findings["actor_patterns"]
            summary.append("Key Actor Risk Assessment:")
            
            # Sort actors by pattern score
            sorted_actors = sorted(
                actors.items(),
                key=lambda x: x[1].get("authoritarian_pattern_score", 0),
                reverse=True
            )
            
            for actor, data in sorted_actors[:5]:  # Top 5 actors
                score = data.get("authoritarian_pattern_score", 0)
                risk_level = data.get("risk_level", "Unknown")
                indicator = data.get("most_frequent_indicator", {}).get("indicator", "unknown")
                indicator_name = indicator.replace("_", " ").title()
                summary.append(f"- {actor}: {score:.1f}/10 ({risk_level} risk), Primary concern: {indicator_name}")
            
            summary.append("")
        
        # Topic analysis
        if "recurring_topics" in pattern_findings:
            topics = pattern_findings["recurring_topics"]
            summary.append("Top Recurring Topics:")
            
            # Sort topics by auth score
            sorted_topics = sorted(
                topics.items(),
                key=lambda x: x[1].get("average_auth_score", 0),
                reverse=True
            )
            
            for topic, data in sorted_topics[:5]:  # Top 5 topics
                auth_score = data.get("average_auth_score", 0)
                if auth_score > 0:
                    count = data.get("count", 0)
                    summary.append(f"- {topic}: {auth_score:.1f}/10 avg. authoritarian score ({count} instances)")
            
            summary.append("")
        
        return "\n".join(summary)

    def generate_weekly_report(self, recent_analyses: List[Dict[str, Any]], pattern_findings: Dict[str, Any]) -> str:
        """
        Generate a comprehensive weekly report on democratic resilience.
        
        Args:
            recent_analyses: List of recent analyses from memory
            pattern_findings: Pattern analysis findings
            
        Returns:
            Complete democratic resilience report
        """
        # Compile key information from analyses and patterns
        indicators_summary = self._compile_indicators_summary(recent_analyses)
        patterns_summary = self._summarize_patterns(pattern_findings)
        
        # Extract overall risk level and score
        risk_level = pattern_findings.get("risk_level", "Unknown")
        risk_score = pattern_findings.get("aggregate_authoritarian_risk", 0)
        
        prompt = f"""
        Create a comprehensive "Democratic Resilience Report" based on the following data:
        
        KEY INDICATORS SUMMARY:
        {indicators_summary}
        
        PATTERN ANALYSIS:
        {patterns_summary}
        
        Overall Risk Level: {risk_level} ({risk_score:.1f}/10)
        
        Create a detailed report with these sections:
        
        1. EXECUTIVE SUMMARY: Key findings and overall assessment of the state of democratic resilience (2-3 paragraphs)
        
        2. CRITICAL INDICATORS: The most significant authoritarian indicators observed this week, with specific examples
        
        3. TREND ANALYSIS: How patterns have evolved over time, identifying acceleration or deceleration in key areas
        
        4. INSTITUTIONAL IMPACTS: How specific democratic institutions (courts, agencies, press, elections, etc.) are being affected
        
        5. CROSS-PARTISAN CONCERNS: Issues that should concern Americans regardless of political affiliation
        
        6. HISTORICAL CONTEXT: Similar patterns from history and their outcomes, with focus on how democracies have been undermined
        
        7. RESILIENCE RECOMMENDATIONS: Concrete actions for strengthening democratic resilience, sorted by:
           - For individuals
           - For civic organizations
           - For media organizations
           - For elected officials who support democratic norms
        
        Format this as a formal, evidence-based report suitable for civic organizations, journalists, and concerned
        citizens across the political spectrum. Focus on factual analysis while emphasizing the non-partisan nature
        of protecting democratic institutions and norms.
        """
        
        self.logger.info("Generating democratic resilience report...")
        report = self._call_llm(prompt, max_tokens=2500, temperature=0.4)
        
        return report
    
    def generate_actor_report(self, actor: str, actor_data: Dict[str, Any], recent_analyses: List[Dict[str, Any]]) -> str:
        """
        Generate a focused report on a specific political actor's authoritarian patterns.
        
        Args:
            actor: Name of the political actor
            actor_data: Data about the actor's authoritarian patterns
            recent_analyses: List of recent analyses from memory
            
        Returns:
            Actor-focused report
        """
        # Extract key information about the actor
        score = actor_data.get("authoritarian_pattern_score", 0)
        risk_level = actor_data.get("risk_level", "Unknown")
        
        # Extract indicator frequencies
        indicators = actor_data.get("indicator_frequencies", {})
        sorted_indicators = sorted(indicators.items(), key=lambda x: x[1], reverse=True)
        
        # Format indicator list
        indicator_list = []
        for indicator, count in sorted_indicators:
            if count > 0:
                indicator_name = indicator.replace("_", " ").title()
                indicator_list.append(f"{indicator_name}: {count} instances")
        
        # Extract examples
        examples = actor_data.get("examples", [])
        examples_text = []
        for i, example in enumerate(examples[:5]):
            examples_text.append(f"Example {i+1}: {example.get('example', 'No text')}")
            examples_text.append(f"Source: {example.get('source', 'Unknown')}, Article: {example.get('article_title', 'Unknown')}")
            examples_text.append("")
        
        prompt = f"""
        Create a focused analytical report on authoritarian governance patterns associated with {actor}.
        
        AUTHORITARIAN RISK ASSESSMENT:
        - Overall Pattern Score: {score:.1f}/10
        - Risk Level: {risk_level}
        
        KEY INDICATORS:
        {chr(10).join(f"- {item}" for item in indicator_list)}
        
        EXAMPLES:
        {chr(10).join(examples_text)}
        
        Create a detailed analysis with these sections:
        
        1. EXECUTIVE SUMMARY: Assessment of {actor}'s actions and rhetoric related to democratic norms (2 paragraphs)
        
        2. PATTERN ANALYSIS: Key patterns in {actor}'s approach to democratic institutions and norms
        
        3. HISTORICAL PARALLELS: Historical comparison to similar approaches by political figures
        
        4. DEMOCRATIC IMPACT: How these patterns could impact democratic institutions
        
        5. CROSS-PARTISAN FRAMING: How to discuss these concerns in ways that resonate across political divides
        
        6. RECOMMENDED RESPONSES: Evidence-based approaches to strengthen democratic resilience in response
        
        Format this as an analytical, factual assessment that avoids partisan framing while clearly
        addressing specific threats to democratic institutions and norms. Focus on actions and patterns
        rather than character or personality.
        """
        
        self.logger.info(f"Generating actor report for {actor}...")
        report = self._call_llm(prompt, max_tokens=1800, temperature=0.4)
        
        return report

    def generate_topic_report(self, topic: str, topic_data: Dict[str, Any], recent_analyses: List[Dict[str, Any]]) -> str:
        """
        Generate a focused report on a specific recurring topic with democratic implications.
        
        Args:
            topic: The recurring topic
            topic_data: Data about the topic's patterns
            recent_analyses: List of recent analyses from memory
            
        Returns:
            Topic-focused report
        """
        # Extract key information about the topic
        avg_auth_score = topic_data.get("average_auth_score", 0)
        count = topic_data.get("count", 0)
        
        # Extract related indicators
        related_indicators = topic_data.get("related_indicators", {})
        sorted_indicators = sorted(related_indicators.items(), key=lambda x: x[1], reverse=True)
        
        # Format indicator list
        indicator_list = []
        for indicator, count in sorted_indicators:
            if count > 0:
                indicator_name = indicator.replace("_", " ").title()
                indicator_list.append(f"{indicator_name}: {count} instances")
        
        # Extract examples
        examples = topic_data.get("examples", [])
        examples_text = []
        for i, example in enumerate(examples[:3]):
            examples_text.append(f"Example {i+1}: {example.get('title', 'Unknown')}")
            examples_text.append(f"Source: {example.get('source', 'Unknown')}")
            examples_text.append("")
        
        # Extract timeline if available
        timeline = topic_data.get("timeline", [])
        timeline_text = []
        
        if timeline:
            timeline_sorted = sorted(timeline, key=lambda x: x.get("timestamp", ""), reverse=True)
            for i, item in enumerate(timeline_sorted[:5]):
                auth_score = item.get("auth_score", 0)
                if auth_score > 0:
                    timeline_text.append(f"- {item.get('title', 'Unknown article')}: Auth. Score {auth_score}/10")
        
        prompt = f"""
        Create a focused analytical report on the topic of "{topic}" and its democratic implications.
        
        TOPIC ASSESSMENT:
        - Average Authoritarian Score: {avg_auth_score:.1f}/10
        - Occurrences: {count} articles
        
        RELATED AUTHORITARIAN INDICATORS:
        {chr(10).join(f"- {item}" for item in indicator_list)}
        
        EXAMPLES:
        {chr(10).join(examples_text)}
        
        RECENT TIMELINE:
        {chr(10).join(timeline_text)}
        
        Create a detailed analysis with these sections:
        
        1. EXECUTIVE SUMMARY: Assessment of how this topic relates to democratic resilience (2 paragraphs)
        
        2. DEMOCRATIC IMPLICATIONS: How this topic impacts democratic institutions and norms
        
        3. NARRATIVE ANALYSIS: How the topic is being framed and potential manipulation
        
        4. CROSS-PARTISAN PERSPECTIVE: How different political perspectives view this topic
        
        5. BRIDGING OPPORTUNITIES: Finding common ground on this topic to strengthen democratic dialogue
        
        6. RECOMMENDED MESSAGING: Evidence-based approaches to discuss this topic in ways that strengthen
           rather than undermine democratic values
        
        Format this as an analytical, factual assessment that provides context and clarity on this topic
        while identifying opportunities to build democratic resilience through constructive engagement.
        """
        
        self.logger.info(f"Generating topic report for '{topic}'...")
        report = self._call_llm(prompt, max_tokens=1800, temperature=0.4)
        
        return report
    
    def generate_action_kit(self, risk_level: str, pattern_findings: Dict[str, Any]) -> str:
        """
        Generate a practical action kit for citizens based on the current risk assessment.
        
        Args:
            risk_level: Current democratic risk level
            pattern_findings: Pattern analysis findings
            
        Returns:
            Democratic resilience action kit
        """
        # Extract key information about risks
        risk_score = pattern_findings.get("aggregate_authoritarian_risk", 0)
        
        # Extract trending indicators
        trend_analysis = pattern_findings.get("trend_analysis", {})
        sorted_indicators = sorted(
            [(k, v) for k, v in trend_analysis.items() if k != "authoritarian_score"],
            key=lambda x: x[1].get("trend_strength", 0),
            reverse=True
        )
        
        # Format top indicators
        top_indicators = []
        for indicator, data in sorted_indicators[:3]:
            indicator_name = indicator.replace("_", " ").title()
            count = data.get("count", 0)
            top_indicators.append(f"{indicator_name} ({count} instances)")
        
        prompt = f"""
        Create a practical "Democratic Resilience Action Kit" for citizens based on the current risk assessment.
        
        CURRENT DEMOCRATIC RISK ASSESSMENT:
        - Risk Level: {risk_level}
        - Risk Score: {risk_score:.1f}/10
        
        TOP CONCERNS:
        {chr(10).join(f"- {item}" for item in top_indicators)}
        
        Create a practical, action-oriented guide with these sections:
        
        1. UNDERSTANDING THE RISK: Brief explanation of the current risk level and what it means (1 paragraph)
        
        2. PERSONAL ACTIONS: Five specific actions individuals can take to strengthen democratic resilience, such as:
           - Media literacy practices
           - Civic engagement opportunities
           - Community dialogue suggestions
           - Information verification habits
           - Democratic skills development
        
        3. FAMILY & COMMUNITY ACTIONS: Five actions to build democratic resilience in local communities:
           - Bridging activities across political divides
           - Supporting local democratic institutions
           - Promoting democratic values in community spaces
           - Having constructive conversations with those who disagree
           - Building democratic capacity in local organizations
        
        4. DIGITAL CITIZENSHIP: Four specific practices for strengthening democracy in digital spaces:
           - Countering manipulation without increasing polarization
           - Sharing information responsibly
           - Engaging constructively across divides
           - Supporting reliable information sources
        
        5. DISCUSSION GUIDE: Three specific conversation prompts that can help people discuss democratic
           values with those across political divides
        
        6. RESOURCES: Suggestions for organizations, books, and tools that build democratic resilience
        
        Format this as a practical, accessible guide written for ordinary citizens. Use clear, direct language 
        and concrete suggestions that anyone could implement. Avoid partisan framing and focus on democracy-
        strengthening actions that would appeal across the political spectrum.
        """
        
        self.logger.info("Generating democratic resilience action kit...")
        action_kit = self._call_llm(prompt, max_tokens=2000, temperature=0.5)
        
        return action_kit

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data to generate comprehensive democratic resilience reports.
        
        Args:
            input_data: Input data containing analysis results and parameters
            
        Returns:
            Generated reports
        """
        report_type = input_data.get("report_type", "weekly")
        lookback_days = input_data.get("lookback_days", 7)
        
        # Get recent analyses from memory system
        if self.memory:
            recent_analyses = self.memory.get_recent_analyses(lookback_days)
        else:
            recent_analyses = input_data.get("recent_analyses", [])
            
        # Get pattern findings
        pattern_findings = input_data.get("pattern_findings", {})
        
        # Generate appropriate report based on type
        if report_type == "weekly":
            report = self.generate_weekly_report(recent_analyses, pattern_findings)
            return {
                "report_type": "weekly",
                "report": report,
                "lookback_days": lookback_days,
                "timestamp": datetime.now().isoformat(),
                "risk_level": pattern_findings.get("risk_level", "Unknown"),
                "risk_score": pattern_findings.get("aggregate_authoritarian_risk", 0)
            }
        
        elif report_type == "actor":
            actor = input_data.get("actor", "")
            actor_data = input_data.get("actor_data", {})
            
            if not actor or not actor_data:
                return {"error": "Actor or actor data missing for actor report"}
                
            report = self.generate_actor_report(actor, actor_data, recent_analyses)
            return {
                "report_type": "actor",
                "actor": actor,
                "report": report,
                "timestamp": datetime.now().isoformat(),
                "risk_level": actor_data.get("risk_level", "Unknown"),
                "risk_score": actor_data.get("authoritarian_pattern_score", 0)
            }
        
        elif report_type == "topic":
            topic = input_data.get("topic", "")
            topic_data = input_data.get("topic_data", {})
            
            if not topic or not topic_data:
                return {"error": "Topic or topic data missing for topic report"}
                
            report = self.generate_topic_report(topic, topic_data, recent_analyses)
            return {
                "report_type": "topic",
                "topic": topic,
                "report": report,
                "timestamp": datetime.now().isoformat(),
                "authoritarian_score": topic_data.get("average_auth_score", 0)
            }
        
        elif report_type == "action_kit":
            risk_level = pattern_findings.get("risk_level", "Unknown")
            
            action_kit = self.generate_action_kit(risk_level, pattern_findings)
            return {
                "report_type": "action_kit",
                "action_kit": action_kit,
                "timestamp": datetime.now().isoformat(),
                "risk_level": risk_level,
                "risk_score": pattern_findings.get("aggregate_authoritarian_risk", 0)
            }
        
        else:
            return {"error": f"Unsupported report type: {report_type}"}


# This allows the module to be run directly for testing
if __name__ == "__main__":
    print("Democratic Resilience Report Generator module")
    print("This module provides report generation capabilities for the Night_watcher framework.")
    print("It should be imported and used as part of the Night_watcher workflow.")