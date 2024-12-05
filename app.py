import os
from collections import defaultdict


def parse_model_names(cache_dir: str) -> dict[str, list[str]]:
    """
    Parse model names from Hugging Face cache directory and organize them by organization.

    Args:
        cache_dir: Path to Hugging Face cache directory

    Returns:
        Dictionary mapping organizations to their models
    """
    models_by_org = defaultdict(list)

    # List all directories in cache
    for item in os.listdir(cache_dir):
        if item.startswith("models--"):
            # Split the model name into components
            parts = item.split("--")
            if len(parts) >= 3:
                org = parts[1]
                model = parts[2]
                models_by_org[org].append(model)

    return dict(models_by_org)


def main():
    # Retrieve the Hugging Face cache directory
    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"Hugging Face model cache directory: {cache_dir}")
    huggingface_hub_cache_dir = cache_dir + "/hub"

    # List all models in the Hugging Face cache directory
    models = os.listdir(huggingface_hub_cache_dir)
    print("Models stored in Hugging Face cache:")
    for model in models:
        print(f"{model}")

    models = parse_model_names(huggingface_hub_cache_dir)

    print("\nHugging Face Models by Organization:")
    print("===================================")
    for org, model_list in sorted(models.items()):
        print(f"\n{org}:")
        for model in sorted(model_list):
            print(f"  - {model}")


if __name__ == "__main__":
    main()
