"""
Example client for Fiftynet API.

Demonstrates how to interact with the Fiftynet REST API.
"""

import requests
import json


def generate_text(
    prompt: str,
    api_url: str = "http://localhost:8000",
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
):
    """
    Generate text using the Fiftynet API.

    Args:
        prompt: Input text prompt
        api_url: Base URL of the API server
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling (0 to disable)
        top_p: Nucleus sampling threshold

    Returns:
        Generated text
    """
    endpoint = f"{api_url}/generate"

    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()

        result = response.json()
        return result

    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None


def check_health(api_url: str = "http://localhost:8000"):
    """Check API health status."""
    endpoint = f"{api_url}/health"

    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error checking health: {e}")
        return None


def get_model_info(api_url: str = "http://localhost:8000"):
    """Get information about the loaded model."""
    endpoint = f"{api_url}/model-info"

    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting model info: {e}")
        return None


def main():
    """Example usage of the API client."""
    api_url = "http://localhost:8000"

    print("=== Fiftynet API Client Example ===\n")

    # Check health
    print("1. Checking API health...")
    health = check_health(api_url)
    if health:
        print(f"   Status: {health['status']}")
        print(f"   Model loaded: {health['model_loaded']}")
        print(f"   Uptime: {health['uptime']:.2f}s\n")
    else:
        print("   API is not available!\n")
        return

    # Get model info
    print("2. Getting model information...")
    info = get_model_info(api_url)
    if info:
        print(f"   Model: {info['model_name']}")
        print(f"   Vocab size: {info['vocab_size']}")
        print(f"   Dimensions: {info['dim']}")
        print(f"   Blocks: {info['num_blocks']}")
        print(f"   Device: {info['device']}\n")

    # Generate text - greedy
    print("3. Generating text (greedy)...")
    result = generate_text(
        prompt="The future of AI is",
        max_new_tokens=20,
        temperature=0.0,  # Greedy
    )
    if result:
        print(f"   Prompt: {result['prompt']}")
        print(f"   Generated: {result['generated_text']}")
        print(f"   Tokens: {result['tokens_generated']}")
        print(f"   Time: {result['inference_time']:.3f}s\n")

    # Generate text - sampling
    print("4. Generating text (nucleus sampling)...")
    result = generate_text(
        prompt="Once upon a time",
        max_new_tokens=30,
        temperature=0.9,
        top_p=0.95,
    )
    if result:
        print(f"   Prompt: {result['prompt']}")
        print(f"   Generated: {result['generated_text']}")
        print(f"   Tokens: {result['tokens_generated']}")
        print(f"   Time: {result['inference_time']:.3f}s\n")

    # Generate text - top-k sampling
    print("5. Generating text (top-k sampling)...")
    result = generate_text(
        prompt="In a world where",
        max_new_tokens=25,
        temperature=1.0,
        top_k=50,
    )
    if result:
        print(f"   Prompt: {result['prompt']}")
        print(f"   Generated: {result['generated_text']}")
        print(f"   Tokens: {result['tokens_generated']}")
        print(f"   Time: {result['inference_time']:.3f}s\n")


if __name__ == "__main__":
    main()
