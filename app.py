import os
from collections import defaultdict
import math
import gradio as gr
import pandas as pd
# from typing import Tuple, List


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
    Format size in bytes to GB.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string in GB
    """
    if size_bytes == 0:
        return "0 GB"

    size_gb = size_bytes / (1024**3)  # Convert to GB
    return f"{size_gb:.2f} GB"


def get_models_data() -> tuple[pd.DataFrame, str]:
    """
    Get models data as a pandas DataFrame and total size.

    Returns:
        Tuple of (DataFrame with model info, total size string)
    """
    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    huggingface_hub_cache_dir = cache_dir + "/hub"

    # Get models organized by organization
    models = parse_model_names(huggingface_hub_cache_dir)

    # Prepare data for DataFrame
    data = []
    total_size = 0

    for org, model_list in sorted(models.items()):
        for model in sorted(model_list):
            model_dir = os.path.join(huggingface_hub_cache_dir, f"models--{org}--{model}")
            size = get_directory_size(model_dir)
            total_size += size
            data.append(
                {
                    "Organization": org,
                    "Model": model,
                    "Size": format_size(size),
                    "Raw Size": size,  # For sorting
                }
            )

    df = pd.DataFrame(data)
    return df, format_size(total_size)


def filter_models(organization: str, df: pd.DataFrame) -> pd.DataFrame:
    """Filter models by organization"""
    if organization == "All Organizations":
        return df
    return df[df["Organization"] == organization]


def create_interface():
    # Get initial data
    df, total_size = get_models_data()

    def refresh_data():
        new_df, new_total = get_models_data()
        orgs = ["All Organizations"] + sorted(new_df["Organization"].unique().tolist())
        return new_df, new_total, gr.Dropdown(choices=orgs, value="All Organizations")

    def update_table(org: str, df: pd.DataFrame):
        filtered_df = filter_models(org, df)
        org_total = format_size(filtered_df["Raw Size"].sum())
        return filtered_df[["Organization", "Model", "Size"]], org_total

    with gr.Blocks(title="Hugging Face Model Cache Viewer") as interface:
        gr.Markdown("# 🤗 Hugging Face Model Cache Viewer")

        with gr.Row():
            with gr.Column():
                refresh_btn = gr.Button("🔄 Refresh Data", variant="primary")
                total_size_text = gr.Textbox(label="Total Cache Size", value=total_size, interactive=False)
                org_dropdown = gr.Dropdown(
                    choices=["All Organizations"] + sorted(df["Organization"].unique().tolist()),
                    value="All Organizations",
                    label="Filter by Organization",
                )
                org_size_text = gr.Textbox(label="Selected Organization(s) Size", value=total_size, interactive=False)

        table = gr.DataFrame(
            value=df[["Organization", "Model", "Size"]],
            headers=["Organization", "Model", "Size"],
            datatype=["str", "str", "str"],
            col_count=(3, "fixed"),
            interactive=False,
        )

        # Hidden state for the full DataFrame
        state = gr.State(df)

        # Set up event handlers
        refresh_btn.click(refresh_data, outputs=[state, total_size_text, org_dropdown]).then(
            update_table, inputs=[org_dropdown, state], outputs=[table, org_size_text]
        )

        org_dropdown.change(update_table, inputs=[org_dropdown, state], outputs=[table, org_size_text])

    return interface


interface = create_interface()


def main():
    interface.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()
