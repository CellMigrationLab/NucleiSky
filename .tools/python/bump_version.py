#!/usr/bin/env python3
import argparse
import pathlib
import re
import shutil
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONSTRUCT = ROOT / "construct.yaml"

# Match version line with optional quotes and optional trailing comment
VERSION_LINE_RE = re.compile(
    r'^(version:\s*)(?P<quote>["\']?)(?P<version>\d+\.\d+\.\d+)(?P=quote)(?P<trailing>\s*(?:#.*)?)$',
    re.MULTILINE,
)
THANKS_LINE_RE = re.compile(r'(Thank you! You have successfully installed [^\n]*)(\d+\.\d+\.\d+)(!)')
PKG_VERSION_RE = re.compile(r'^(\s*SET\s+"PKG_VERSION=)(\d+\.\d+\.\d+)("\s*)$', re.MULTILINE)
POST_INSTALL = ROOT / "app" / "bash_bat_scripts" / "post_install.bat"


def replace_version_in_file(file_path: pathlib.Path, old_version: str, new_version: str) -> bool:
    """Replace all occurrences of old_version with new_version in a file.
    
    Works for markdown and Python files. Handles various version formats:
    - Bare versions: 1.2.3
    - With 'v' prefix: v1.2.3
    - In quotes: "1.2.3" or '1.2.3'
    - In version strings: version 1.2.3, __version__ = "1.2.3"
    
    Args:
        file_path: Path to the file to update
        old_version: Current version string (X.Y.Z format)
        new_version: New version string (X.Y.Z format)
    
    Returns:
        True if any replacements were made, False otherwise
    """
    if not file_path.exists():
        return False
    
    text = file_path.read_text(encoding="utf-8")
    original_text = text
    
    # Escape dots for regex
    old_escaped = re.escape(old_version)
    
    # Pattern matches version with optional surrounding characters but preserves them
    # Matches: v1.2.3, "1.2.3", '1.2.3', (1.2.3), [1.2.3], bare 1.2.3
    # Does NOT match partial versions like 1.2.34 or 11.2.3
    pattern = re.compile(
        r'(?<=["\'\[(\s>v]|^)' +  # Lookbehind: quote, bracket, paren, space, >, v, or start
        old_escaped + 
        r'(?=["\'\])\s<,;]|$)'  # Lookahead: quote, bracket, paren, space, <, comma, semicolon, or end
    )
    
    # Also match v-prefixed versions explicitly
    v_pattern = re.compile(r'(v)' + old_escaped + r'(?=["\'\])\s<,;!]|$)')
    
    # Replace v-prefixed versions
    text = v_pattern.sub(r'\g<1>' + new_version, text)
    
    # Replace other versions
    text = pattern.sub(new_version, text)
    
    if text != original_text:
        file_path.write_text(text, encoding="utf-8")
        return True
    return False


def read_current_version() -> tuple[str, str]:
    """Read construct.yaml as text and extract current SemVer.

    Returns (file_text, version_string). Exits if not found.
    """
    text = CONSTRUCT.read_text(encoding="utf-8")
    m = VERSION_LINE_RE.search(text)
    if not m:
        raise ValueError("Could not find version in construct.yaml")
    return text, m.group("version")


def bump_construct_text(text: str, old_version: str, new_version: str) -> str:
    """Return updated file text with bumped version, preserving YAML structure/comments.

    - Updates the `version: "X.Y.Z"` line.
    - Updates the version inside the 'Thank you for installing ... vX.Y.Z!' line
      within `conclusion_text` if present.
    """
    # 1) Update the explicit version line
    def _repl_version(m: re.Match) -> str:
        # Preserve original quoting and trailing comment/whitespace
        return f"{m.group(1)}{m.group('quote')}{new_version}{m.group('quote')}{m.group('trailing')}"

    text = VERSION_LINE_RE.sub(_repl_version, text, count=1)

    # 2) Update the thank-you line version (only first occurrence)
    text = THANKS_LINE_RE.sub(lambda m: f"{m.group(1)}{new_version}{m.group(3)}", text, count=1)
    return text


def bump_post_install_bat(new_version: str) -> None:
    if not POST_INSTALL.exists():
        return
    text = POST_INSTALL.read_text(encoding="utf-8")
    if not PKG_VERSION_RE.search(text):
        return
    updated = PKG_VERSION_RE.sub(lambda m: f"{m.group(1)}{new_version}{m.group(3)}", text, count=1)
    POST_INSTALL.write_text(updated, encoding="utf-8")
    print(f"Updated post_install.bat PKG_VERSION to {new_version}")

def bump_version_in_download_executable_md(new_version: str) -> None:
    download_md = ROOT / "docs" / "download_executable.md"
    if not download_md.exists():
        return
    # This file may contain the version in various formats, so we use the flexible replacement function
    if replace_version_in_file(download_md, new_version, new_version):
        print(f"Updated download_executable.md to version {new_version}")
    else:
        print("No version string found in download_executable.md to update.")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update project version across release artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Example:
              python .tools/python/bump_version.py 0.0.4
              python .tools/python/bump_version.py 0.0.4 --files README.md src/__init__.py
            """
        ),
    )
    parser.add_argument("new_version", help="SemVer version, e.g. 0.0.4")
    parser.add_argument(
        "--files", 
        nargs="+", 
        help="Additional files to update (e.g., README.md, __init__.py)"
    )
    args = parser.parse_args()

    if not re.fullmatch(r"\d+\.\d+\.\d+", args.new_version):
        sys.exit("Version must look like X.Y.Z")

    # On the first release replace download_executable.md with the template
    # (but only if it exists)
    if pathlib.Path(".tools/templates/download_executable_template.md").exists():
        # Remove existing download_executable.md if present
        if pathlib.Path(".tools/docs/download_executable.md").exists():
            pathlib.Path(".tools/docs/download_executable.md").unlink()
        # Copy the template to the docs folder using shutil for cross-platform support
        shutil.copy(
            ".tools/templates/download_executable_template.md",
            ".tools/docs/download_executable.md",
        )

    text, current = read_current_version()
    if args.new_version == current:
        print(f"Version already at {current}; nothing to update.")
        return

    updated_text = bump_construct_text(text, current, args.new_version)

    CONSTRUCT.write_text(updated_text, encoding="utf-8")
    print(f"Bumped {current} -> {args.new_version}")

    # Also bump PKG_VERSION in post_install.bat
    bump_post_install_bat(args.new_version)

    # Update download_executable.md if it exists
    bump_version_in_download_executable_md(args.new_version)

    # Update additional files if specified
    if args.files:
        for file_path in args.files:
            full_path = ROOT / file_path
            if replace_version_in_file(full_path, current, args.new_version):
                print(f"Updated {file_path}")
            else:
                print(f"No changes in {file_path}")


if __name__ == "__main__":
    main()
