"""
Night_watcher Workflow Orchestrator - Partial Update
Handle scenarios with no available LLM provider
"""

def run(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run the Night_watcher workflow with focus on intelligence analysis"""
    if config is None:
        config = {}

    article_limit = config.get("article_limit", 5)
    sources = config.get("sources", None)
    pattern_analysis_days = config.get("pattern_analysis_days", 30)
    llm_provider_available = config.get("llm_provider_available", True)
    
    # Use provided date range if available
    start_date = config.get("start_date")
    end_date = config.get("end_date")

    self.logger.info(f"Starting Night_watcher workflow with timestamp {self.timestamp}")

    # 1. Collect articles
    self.logger.info("Collecting articles with focus on government/political content...")
    collection_params = {"limit": article_limit}
    if sources:
        collection_params["sources"] = sources
    
    # Add date range if provided
    if start_date:
        collection_params["start_date"] = start_date
    if end_date:
        collection_params["end_date"] = end_date

    collection_result = self.collector.process(collection_params)
    articles = collection_result.get("articles", [])
    self.logger.info(f"Collected {len(articles)} articles")

    # Save collected articles
    save_to_file(articles, f"{self.output_dir}/collected/articles_{self.timestamp}.json")
    
    # If no LLM provider is available, stop here
    if not llm_provider_available or self.llm_provider is None:
        self.logger.warning("No LLM provider available. Stopping after content collection.")
        return {
            "timestamp": self.timestamp,
            "output_dir": self.output_dir,
            "articles_collected": len(articles),
            "date_range": {
                "start_date": start_date.isoformat() if hasattr(start_date, 'isoformat') else start_date,
                "end_date": end_date.isoformat() if hasattr(end_date, 'isoformat') else end_date
            }
        }

    # 2. Analyze content for both divisive elements and authoritarian patterns
    self.logger.info("Analyzing articles for divisive content and authoritarian patterns...")
    analysis_result = self.analyzer.process({"articles": articles})
    analyses = analysis_result.get("analyses", [])
    auth_analyses = analysis_result.get("authoritarian_analyses", [])

    # Save individual analyses and store in memory system
    for i, analysis in enumerate(analyses):
        if "article" in analysis:
            article_slug = create_slug(analysis['article']['title'])
            save_to_file(analysis, f"{self.output_dir}/analyzed/analysis_{article_slug}_{self.timestamp}.json")

            # Store in memory system
            analysis_id = self.memory.store_article_analysis(analysis)

            # If we have corresponding authoritarian analysis, save it
            if i < len(auth_analyses):
                auth_analysis = auth_analyses[i]
                save_to_file(auth_analysis,
                             f"{self.output_dir}/analyzed/auth_analysis_{article_slug}_{self.timestamp}.json")

                # Store authoritarian analysis in memory
                auth_meta = {
                    "type": "authoritarian_analysis",
                    "title": analysis['article']['title'],
                    "source": analysis['article']['source'],
                    "url": analysis['article'].get('url', ''),
                    "parent_id": analysis_id
                }

                # Extract authoritarian score if available
                if "structured_elements" in auth_analysis:
                    auth_meta["authoritarian_score"] = auth_analysis["structured_elements"].get(
                        "authoritarian_score", 0)

                self.memory.store.add_item(
                    f"auth_{analysis_id}",
                    auth_analysis.get("authoritarian_analysis", ""),
                    auth_meta
                )

    # 3. Run pattern analysis to identify authoritarian trends
    self.logger.info(f"Running pattern analysis over the last {pattern_analysis_days} days...")

    # Run all pattern analyses
    analysis_count = 0
    
    # Authoritarian trend analysis
    auth_trends = self.pattern_recognition.analyze_source_bias_patterns(pattern_analysis_days)
    save_to_file(
        auth_trends,
        f"{self.analysis_dir}/authoritarian_trends_{self.timestamp}.json"
    )
    analysis_count += 1

    # Topic analysis
    topic_analysis = self.pattern_recognition.identify_recurring_topics()
    save_to_file(
        topic_analysis,
        f"{self.analysis_dir}/topic_analysis_{self.timestamp}.json"
    )
    analysis_count += 1

    # Actor analysis
    actor_analysis = self.pattern_recognition.analyze_authoritarian_actors(pattern_analysis_days)
    save_to_file(
        actor_analysis,
        f"{self.analysis_dir}/actor_analysis_{self.timestamp}.json"
    )
    analysis_count += 1

    self.logger.info(f"Processing complete. All outputs saved in {self.output_dir}")

    return {
        "timestamp": self.timestamp,
        "output_dir": self.output_dir,
        "articles_collected": len(articles),
        "articles_analyzed": len(analyses),
        "pattern_analyses_generated": analysis_count,
        "date_range": {
            "start_date": start_date.isoformat() if hasattr(start_date, 'isoformat') else start_date,
            "end_date": end_date.isoformat() if hasattr(end_date, 'isoformat') else end_date
        }
    }
