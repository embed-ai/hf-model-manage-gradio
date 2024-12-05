import os
from collections import defaultdict
import math


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


def get_directory_size(path: str) -> int:
    """
    Calculate the total size of a directory in bytes.

    Args:
        path: Path to the directory

    Returns:
        Total size in bytes
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if not os.path.islink(file_path):  # Skip symbolic links
                total_size += os.path.getsize(file_path)
    return total_size


def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string with appropriate unit
    """
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    exponent = int(math.log(size_bytes, 1024))
    unit = units[min(exponent, len(units) - 1)]
    size = size_bytes / (1024**exponent)
    return f"{size:.2f} {unit}"


def main():
    # Retrieve the Hugging Face cache directory
    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"Hugging Face model cache directory: {cache_dir}")
    huggingface_hub_cache_dir = cache_dir + "/hub"

    # Get models organized by organization
    models = parse_model_names(huggingface_hub_cache_dir)

    print("\nHugging Face Models by Organization and Size:")
    print("===========================================")
    total_size = 0

    for org, model_list in sorted(models.items()):
        print(f"\n{org}:")
        org_size = 0
        for model in sorted(model_list):
            model_dir = os.path.join(huggingface_hub_cache_dir, f"models--{org}--{model}")
            size = get_directory_size(model_dir)
            org_size += size
            total_size += size
            print(f"  - {model}: {format_size(size)}")
        print(f"  Total organization size: {format_size(org_size)}")

    print(f"\nTotal disk space used by all models: {format_size(total_size)}")


if __name__ == "__main__":
    main()
