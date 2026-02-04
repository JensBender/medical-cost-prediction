from pathlib import Path

def get_root_directory(anchor_files: str | list[str] = [".git", "Readme.md", "requirements.txt"]) -> Path:
    """
    Finds the project root directory by searching upwards for anchor files.
    Works from any script within the project's directory structure.
    """
    # Standardize inputs to list[str]
    if isinstance(anchor_files, str):
        anchor_files = [anchor_files]

    # Get absolute path to this file (src/utils.py)
    file_path = Path(__file__).resolve()

    # Iterate over each parent directory
    for parent in file_path.parents:
        # Iterate over each anchor file
        for anchor_file in anchor_files:
            # Check if anchor file exists in parent directory
            if (parent / anchor_file).exists():
                return parent
                
    raise FileNotFoundError(f"Root directory not found: None of the anchor files '{anchor_files}' were found in any parent directory.")
