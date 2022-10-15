import os
import tarfile


def untar(archive_path: str, output_path: str):
    """Untar archive."""
    with tarfile.open(archive_path) as tar_file:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tar_file.extractall(output_path)
