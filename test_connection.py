"""
Test script for verifying LLM connectivity.

Usage:
    python test_connection.py              # test default model
    python test_connection.py qwen3-32b    # test specific model
    python test_connection.py --all        # test all models + embedding
"""

import sys
import time
from openai import OpenAI

from config import MODELS, EMBEDDING, DEFAULT_MODEL_KEY, get_model_cfg


def test_chat_model(model_key: str) -> bool:
    """Tests chat completion for a single model."""
    cfg = get_model_cfg(model_key)
    print(f"\n{'='*60}")
    print(f"  Model:    {cfg['name']}")
    print(f"  URL:      {cfg['base_url']}")
    print(f"  Context:  {cfg.get('context_window', '?')}")
    print(f"{'='*60}")

    client = OpenAI(
        api_key=cfg.get("api_key", "none"),
        base_url=cfg["base_url"],
    )

    # 1. Check /models endpoint
    print("\n[1/3] GET /models ... ", end="", flush=True)
    try:
        available = client.models.list()
        model_ids = [m.id for m in available.data]
        print(f"OK  ({len(model_ids)} models: {model_ids})")
    except Exception as e:
        print(f"FAIL\n      {e}")
        return False

    # 2. Simple completion
    print("[2/3] Chat completion ... ", end="", flush=True)
    try:
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=cfg["name"],
            messages=[
                {"role": "user", "content": "Say 'hello' in one word."},
            ],
            max_tokens=16,
            temperature=0,
        )
        duration = time.perf_counter() - t0
        reply = response.choices[0].message.content.strip()
        usage = response.usage
        tokens_info = ""
        if usage:
            tokens_info = (
                f"  (prompt: {usage.prompt_tokens}, "
                f"completion: {usage.completion_tokens})"
            )
        print(f"OK  [{duration:.2f}s]{tokens_info}")
        print(f"      Reply: \"{reply}\"")
    except Exception as e:
        print(f"FAIL\n      {e}")
        return False

    # 3. JSON mode (used by parse_task_type)
    print("[3/3] JSON mode ... ", end="", flush=True)
    try:
        response = client.chat.completions.create(
            model=cfg["name"],
            messages=[
                {"role": "system", "content": "Return a JSON object with key 'status' and value 'ok'."},
                {"role": "user", "content": "test"},
            ],
            response_format={"type": "json_object"},
            max_tokens=32,
            temperature=0,
        )
        reply = response.choices[0].message.content.strip()
        print(f"OK")
        print(f"      Reply: {reply}")
    except Exception as e:
        print(f"FAIL\n      {e}")
        return False

    print(f"\n  >> {model_key}: ALL TESTS PASSED")
    return True


def test_embedding() -> bool:
    """Tests the embedding model."""
    if not EMBEDDING:
        print("\n[embedding] Not configured, skipping.")
        return True

    print(f"\n{'='*60}")
    print(f"  Embedding: {EMBEDDING['name']}")
    print(f"  URL:       {EMBEDDING['base_url']}")
    print(f"{'='*60}")

    client = OpenAI(
        api_key=EMBEDDING.get("api_key", "none"),
        base_url=EMBEDDING["base_url"],
    )

    print("\n[1/1] Create embedding ... ", end="", flush=True)
    try:
        t0 = time.perf_counter()
        response = client.embeddings.create(
            model=EMBEDDING["name"],
            input="test embedding",
        )
        duration = time.perf_counter() - t0
        dim = len(response.data[0].embedding)
        print(f"OK  [{duration:.2f}s]  dim={dim}")
    except Exception as e:
        print(f"FAIL\n      {e}")
        return False

    print(f"\n  >> embedding: PASSED")
    return True


def main():
    args = sys.argv[1:]

    if "--all" in args:
        # Test all models + embedding
        keys_to_test = list(MODELS.keys())
        test_emb = True
    elif args:
        keys_to_test = [a for a in args if a != "--all"]
        test_emb = False
    else:
        keys_to_test = [DEFAULT_MODEL_KEY]
        test_emb = False

    results = {}

    for key in keys_to_test:
        results[key] = test_chat_model(key)

    if test_emb:
        results["embedding"] = test_embedding()

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name:20s}  {status}")

    total_ok = all(results.values())
    print(f"\n  {'All tests passed!' if total_ok else 'Some tests FAILED.'}")
    sys.exit(0 if total_ok else 1)


if __name__ == "__main__":
    main()
