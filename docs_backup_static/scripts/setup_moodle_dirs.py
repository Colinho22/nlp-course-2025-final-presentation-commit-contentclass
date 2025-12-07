"""
Setup script for Moodle GitHub Pages directory structure.

Creates the required directory structure for integrating Moodle content
into the GitHub Pages documentation site.

Author: Claude Code
Date: 2025-12-07
"""

from pathlib import Path


def create_directory(path: Path) -> bool:
    """
    Create a directory if it doesn't exist.

    Args:
        path: Path object for the directory to create

    Returns:
        True if directory was created, False if it already existed
    """
    if path.exists():
        print(f"[EXISTS] {path}")
        return False
    else:
        path.mkdir(parents=True, exist_ok=True)
        print(f"[CREATED] {path}")
        return True


def main():
    """Create the Moodle directory structure."""
    print("Setting up Moodle GitHub Pages directory structure...\n")

    # Get the project root (2 levels up from this script)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    # Define directory structure
    directories = [
        project_root / "docs" / "moodle",
        project_root / "docs" / "moodle" / "assets",
        project_root / "docs" / "moodle" / "assets" / "pdfs",
        project_root / "docs" / "moodle" / "assets" / "notebooks",
    ]

    # Create directories
    created_count = 0
    for directory in directories:
        if create_directory(directory):
            created_count += 1

    # Summary
    print(f"\n--- Summary ---")
    print(f"Total directories: {len(directories)}")
    print(f"Newly created: {created_count}")
    print(f"Already existed: {len(directories) - created_count}")

    # Verify structure
    print(f"\n--- Verification ---")
    base_path = project_root / "docs" / "moodle"
    if base_path.exists():
        print(f"Base path exists: {base_path}")
        print("\nDirectory tree:")
        for item in sorted(base_path.rglob("*")):
            if item.is_dir():
                indent = "  " * (len(item.relative_to(base_path).parts) - 1)
                print(f"{indent}- {item.name}/")
    else:
        print(f"ERROR: Base path does not exist: {base_path}")

    print("\nSetup complete!")


if __name__ == "__main__":
    main()
