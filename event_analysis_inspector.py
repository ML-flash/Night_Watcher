#!/usr/bin/env python3
"""
Event Analysis Inspector
Standalone script to inspect analyzed files and count events.
"""

import os
import json
from file_utils import safe_json_load
import glob
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any

def inspect_analyses(analyzed_dir: str = "data/analyzed") -> Dict[str, Any]:
    """
    Inspect all analysis files to understand event extraction.
    
    Args:
        analyzed_dir: Directory containing analysis files
        
    Returns:
        Dictionary with inspection results
    """
    stats = {
        "total_files": 0,
        "files_with_kg_payload": 0,
        "files_with_events": 0,
        "total_event_nodes": 0,
        "event_types": defaultdict(int),
        "node_types": defaultdict(int),
        "sources": defaultdict(int),
        "dates": defaultdict(int),
        "sample_events": [],
        "sample_non_event_analyses": [],
        "files_without_kg": [],
        "files_without_events": [],
        "error_files": [],
        "template_usage": defaultdict(int)
    }
    
    # Find all analysis files
    pattern = os.path.join(analyzed_dir, "analysis_*.json")
    files = glob.glob(pattern)
    stats["total_files"] = len(files)
    
    print(f"Found {len(files)} analysis files in {analyzed_dir}")
    print("-" * 80)
    
    for i, filepath in enumerate(files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(files)}...")
            
        try:
            analysis = safe_json_load(filepath, default=None)
            if analysis is None:
                continue
            
            # Get basic info
            article = analysis.get("article", {})
            source = article.get("source", "Unknown")
            doc_id = article.get("document_id", "Unknown")
            
            # Track template used
            template = analysis.get("template_file", "Unknown")
            stats["template_usage"][template] += 1
            
            stats["sources"][source] += 1
            
            # Check for KG payload
            kg_payload = analysis.get("kg_payload", {})
            if not kg_payload:
                stats["files_without_kg"].append(os.path.basename(filepath))
                continue
                
            stats["files_with_kg_payload"] += 1
            
            # Get nodes
            nodes = kg_payload.get("nodes", [])
            if not nodes:
                stats["files_without_events"].append(os.path.basename(filepath))
                continue
            
            # Count node types
            event_count = 0
            for node in nodes:
                node_type = node.get("node_type", "unknown")
                stats["node_types"][node_type] += 1
                
                if node_type == "event":
                    event_count += 1
                    stats["total_event_nodes"] += 1
                    
                    # Get event details
                    event_name = node.get("name", "Unnamed")
                    event_date = node.get("timestamp", "N/A")
                    event_attrs = node.get("attributes", {})
                    
                    # Track event types
                    event_type = event_attrs.get("event_type", "unspecified")
                    stats["event_types"][event_type] += 1
                    
                    # Track dates
                    if event_date != "N/A":
                        try:
                            date_obj = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
                            year = date_obj.year
                            stats["dates"][year] += 1
                        except:
                            stats["dates"]["invalid"] += 1
                    else:
                        stats["dates"]["N/A"] += 1
                    
                    # Collect sample events
                    if len(stats["sample_events"]) < 10:
                        stats["sample_events"].append({
                            "name": event_name,
                            "date": event_date,
                            "type": event_type,
                            "source": source,
                            "attributes": event_attrs,
                            "file": os.path.basename(filepath)
                        })
            
            if event_count > 0:
                stats["files_with_events"] += 1
            else:
                # Sample analyses without events
                if len(stats["sample_non_event_analyses"]) < 5:
                    stats["sample_non_event_analyses"].append({
                        "file": os.path.basename(filepath),
                        "source": source,
                        "template": template,
                        "node_types_found": list(set(n.get("node_type", "unknown") for n in nodes)),
                        "node_count": len(nodes),
                        "sample_nodes": nodes[:3] if nodes else []
                    })
                
        except Exception as e:
            stats["error_files"].append({
                "file": os.path.basename(filepath),
                "error": str(e)
            })
    
    return stats

def print_report(stats: Dict[str, Any]):
    """Print a formatted report of the inspection results."""
    print("\n" + "="*80)
    print("EVENT ANALYSIS INSPECTION REPORT")
    print("="*80)
    
    print(f"\n📊 FILE STATISTICS:")
    print(f"  Total analysis files: {stats['total_files']}")
    print(f"  Files with KG payload: {stats['files_with_kg_payload']} ({stats['files_with_kg_payload']/stats['total_files']*100:.1f}%)")
    print(f"  Files with event nodes: {stats['files_with_events']} ({stats['files_with_events']/stats['total_files']*100:.1f}%)")
    print(f"  Files with errors: {len(stats['error_files'])}")
    
    print(f"\n📈 EVENT STATISTICS:")
    print(f"  Total event nodes found: {stats['total_event_nodes']}")
    if stats['files_with_events'] > 0:
        print(f"  Average events per file (with events): {stats['total_event_nodes']/stats['files_with_events']:.1f}")
    
    print(f"\n🏷️ NODE TYPE DISTRIBUTION:")
    for node_type, count in sorted(stats['node_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {node_type}: {count}")
    
    print(f"\n📅 EVENT DATE DISTRIBUTION:")
    # Sort dates, handling both strings and integers
    date_items = list(stats['dates'].items())
    date_items.sort(key=lambda x: (isinstance(x[0], str), x[0]))
    for year, count in date_items:
        print(f"  {year}: {count} events")
    
    print(f"\n🎯 EVENT TYPE DISTRIBUTION:")
    for event_type, count in sorted(stats['event_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {event_type}: {count}")
    
    print(f"\n📰 TOP SOURCES:")
    for source, count in sorted(stats['sources'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {source}: {count} analyses")
    
    print(f"\n🔍 SAMPLE EVENTS:")
    for i, event in enumerate(stats['sample_events'], 1):
        print(f"\n  Event {i}:")
        print(f"    Name: {event['name']}")
        print(f"    Date: {event['date']}")
        print(f"    Type: {event['type']}")
        print(f"    Source: {event['source']}")
        if event['attributes']:
            print(f"    Attributes: {json.dumps(event['attributes'], indent=6)}")
    
    if stats['sample_non_event_analyses']:
        print(f"\n🔎 SAMPLE ANALYSES WITHOUT EVENTS:")
        for analysis in stats['sample_non_event_analyses']:
            print(f"\n  File: {analysis['file']}")
            print(f"  Source: {analysis['source']}")
            print(f"  Template: {analysis['template']}")
            print(f"  Node types found: {', '.join(analysis['node_types_found'])}")
            print(f"  Total nodes: {analysis['node_count']}")
            if analysis['sample_nodes']:
                print(f"  Sample node: {analysis['sample_nodes'][0].get('node_type')} - {analysis['sample_nodes'][0].get('name')}")
    
    print(f"\n📋 TEMPLATE USAGE:")
    for template, count in sorted(stats['template_usage'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {template}: {count} analyses")
    
    if stats['files_without_kg']:
        print(f"\n⚠️ FILES WITHOUT KG PAYLOAD: {len(stats['files_without_kg'])}")
        print(f"  (First 5): {stats['files_without_kg'][:5]}")
    
    if stats['files_without_events']:
        print(f"\n⚠️ FILES WITH KG BUT NO EVENTS: {len(stats['files_without_events'])}")
        print(f"  (First 5): {stats['files_without_events'][:5]}")
    
    if stats['error_files']:
        print(f"\n❌ ERROR FILES:")
        for err in stats['error_files'][:5]:
            print(f"  {err['file']}: {err['error']}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect Night_watcher event analyses")
    parser.add_argument("--dir", default="data/analyzed", help="Directory containing analysis files")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    # Run inspection
    print(f"Inspecting analyses in: {args.dir}")
    stats = inspect_analyses(args.dir)
    
    if args.json:
        # Convert defaultdicts to regular dicts for JSON serialization
        stats["event_types"] = dict(stats["event_types"])
        stats["node_types"] = dict(stats["node_types"])
        stats["sources"] = dict(stats["sources"])
        stats["dates"] = dict(stats["dates"])
        stats["template_usage"] = dict(stats["template_usage"])
        print(json.dumps(stats, indent=2))
    else:
        print_report(stats)

if __name__ == "__main__":
    main()