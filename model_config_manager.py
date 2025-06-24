#!/usr/bin/env python3
"""
Night_watcher Model Configuration Manager
Utility to manage model-specific configurations and test context windows.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Import our enhanced providers
from providers import LMStudioProvider, ModelConfig, ContextWindowManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_model_context(model_name: str, host: str = "http://localhost:1234"):
    """Test a model's context window with progressively larger prompts."""

    provider = LMStudioProvider(host, model_name)
    model_config = ModelConfig()

    # Load current config
    config = model_config.load_config(model_name)
    logger.info(f"Current config for {model_name}: {config}")

    # Test with different prompt sizes
    test_prompts = [
        ("Small", "Analyze this short text: Hello world. Respond with JSON."),
        ("Medium", "Analyze this article: " + "This is a test article. " * 100 + " Respond with JSON."),
        ("Large", "Analyze this article: " + "This is a test article. " * 500 + " Respond with JSON."),
        ("Very Large", "Analyze this article: " + "This is a test article. " * 1000 + " Respond with JSON.")
    ]

    results = {}

    for size, prompt in test_prompts:
        logger.info(f"Testing {size} prompt ({len(prompt)} chars)...")

        # Calculate optimal tokens
        optimal_tokens, calc_info = provider.context_manager.calculate_optimal_tokens(
            model_name, prompt, 2000
        )

        logger.info(f"  Estimated tokens: {calc_info['prompt_tokens']}")
        logger.info(f"  Optimal output tokens: {optimal_tokens}")
        logger.info(f"  Context utilization: {calc_info['utilization']:.1%}")

        # Try the actual completion
        try:
            response = provider.complete(prompt, max_tokens=min(optimal_tokens, 100), temperature=0.1)

            if "error" in response:
                logger.error(f"  Error: {response['error']}")
                results[size] = {"status": "error", "error": response["error"]}
            else:
                usage = response.get("usage", {})
                logger.info(f"  Success! Actual tokens: prompt={usage.get('prompt_tokens', 'N/A')}, "
                            f"completion={usage.get('completion_tokens', 'N/A')}")
                results[size] = {
                    "status": "success",
                    "calc_info": calc_info,
                    "actual_usage": usage
                }
        except Exception as e:
            logger.error(f"  Exception: {e}")
            results[size] = {"status": "exception", "error": str(e)}

    return results


def configure_model(model_name: str, **kwargs):
    """Configure a model with specific parameters."""

    model_config = ModelConfig()

    # Load existing config
    config = model_config.load_config(model_name)

    # Update with provided parameters
    for key, value in kwargs.items():
        if value is not None:
            config[key] = value

    # Add timestamp
    config["updated_at"] = "manual_configuration"

    # Save config
    model_config.save_config(model_name, config)

    logger.info(f"Updated configuration for {model_name}:")
    logger.info(json.dumps(config, indent=2))


def list_model_configs():
    """List all saved model configurations."""

    model_config = ModelConfig()
    config_dir = model_config.config_dir

    if not config_dir.exists():
        logger.info("No model configurations found.")
        return

    configs = list(config_dir.glob("*.json"))

    if not configs:
        logger.info("No model configurations found.")
        return

    logger.info(f"Found {len(configs)} model configurations:")

    for config_file in sorted(configs):
        model_name = config_file.stem.replace("_", "/")

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            logger.info(f"\n{model_name}:")
            logger.info(f"  Context window: {config.get('context_window', 'unknown')}")
            logger.info(f"  Max tokens: {config.get('max_tokens', 'unknown')}")
            logger.info(f"  Temperature: {config.get('temperature', 'unknown')}")
            logger.info(f"  Created: {config.get('created_at', 'unknown')}")

        except Exception as e:
            logger.error(f"  Error reading config: {e}")


def auto_detect_context_window(model_name: str, host: str = "http://localhost:1234"):
    """Attempt to auto-detect a model's context window by testing progressively larger prompts."""

    provider = LMStudioProvider(host, model_name)

    # Binary search for context window
    min_tokens = 1000
    max_tokens = 200000  # Start with a very high number

    # Test prompt base
    base_prompt = "Respond with 'OK': "

    logger.info(f"Auto-detecting context window for {model_name}...")

    last_working_size = min_tokens

    # Start with doubling approach
    test_size = min_tokens
    while test_size <= max_tokens:
        # Create prompt of specific token size
        repeat_text = "This is a test sentence. "
        target_chars = test_size * 4  # Rough token to char conversion
        repeats = max(1, target_chars // len(repeat_text))
        test_prompt = base_prompt + (repeat_text * repeats)

        logger.info(f"  Testing {test_size} tokens (~{len(test_prompt)} chars)...")

        try:
            response = provider.complete(test_prompt, max_tokens=50, temperature=0.1, auto_adjust_tokens=False)

            if "error" in response:
                if "context" in response["error"].lower() or "token" in response["error"].lower():
                    logger.info(f"    Context limit reached at {test_size} tokens")
                    break
                else:
                    logger.warning(f"    Other error: {response['error']}")
            else:
                logger.info(f"    Success at {test_size} tokens")
                last_working_size = test_size

        except Exception as e:
            logger.error(f"    Exception at {test_size} tokens: {e}")
            break

        test_size *= 2

    # Refine with binary search between last_working_size and test_size
    if test_size > last_working_size * 2:
        logger.info(f"Refining between {last_working_size} and {test_size} tokens...")

        low = last_working_size
        high = test_size

        for _ in range(5):  # Max 5 iterations
            mid = (low + high) // 2

            repeat_text = "This is a test sentence. "
            target_chars = mid * 4
            repeats = max(1, target_chars // len(repeat_text))
            test_prompt = base_prompt + (repeat_text * repeats)

            logger.info(f"  Testing {mid} tokens...")

            try:
                response = provider.complete(test_prompt, max_tokens=50, temperature=0.1, auto_adjust_tokens=False)

                if "error" in response and (
                        "context" in response["error"].lower() or "token" in response["error"].lower()):
                    high = mid
                else:
                    low = mid
                    last_working_size = mid

            except:
                high = mid

    logger.info(f"Detected context window: approximately {last_working_size} tokens")

    # Update the model configuration
    model_config = ModelConfig()
    config = model_config.load_config(model_name)
    config["context_window"] = last_working_size
    config["max_tokens"] = min(4096, last_working_size // 3)  # Conservative output allocation
    config["auto_detected"] = True
    config["detected_at"] = "auto_detection"

    model_config.save_config(model_name, config)

    return last_working_size


def main():
    """Main CLI interface."""

    parser = argparse.ArgumentParser(description="Night_watcher Model Configuration Manager")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List command
    list_parser = subparsers.add_parser('list', help='List all model configurations')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test model context window')
    test_parser.add_argument('model_name', help='Model name to test')
    test_parser.add_argument('--host', default='http://localhost:1234', help='LM Studio host')

    # Configure command
    config_parser = subparsers.add_parser('config', help='Configure model parameters')
    config_parser.add_argument('model_name', help='Model name to configure')
    config_parser.add_argument('--context-window', type=int, help='Context window size')
    config_parser.add_argument('--max-tokens', type=int, help='Maximum output tokens')
    config_parser.add_argument('--temperature', type=float, help='Temperature')
    config_parser.add_argument('--top-p', type=float, help='Top-p value')
    config_parser.add_argument('--top-k', type=int, help='Top-k value')
    config_parser.add_argument('--repeat-penalty', type=float, help='Repeat penalty')

    # Auto-detect command
    detect_parser = subparsers.add_parser('detect', help='Auto-detect context window')
    detect_parser.add_argument('model_name', help='Model name to detect')
    detect_parser.add_argument('--host', default='http://localhost:1234', help='LM Studio host')

    args = parser.parse_args()

    if args.command == 'list':
        list_model_configs()

    elif args.command == 'test':
        results = test_model_context(args.model_name, args.host)
        logger.info("\nTest Results Summary:")
        for size, result in results.items():
            logger.info(f"  {size}: {result['status']}")

    elif args.command == 'config':
        config_params = {
            'context_window': args.context_window,
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k,
            'repeat_penalty': args.repeat_penalty
        }
        configure_model(args.model_name, **config_params)

    elif args.command == 'detect':
        context_size = auto_detect_context_window(args.model_name, args.host)
        logger.info(f"Auto-detected context window: {context_size} tokens")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()