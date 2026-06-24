from __future__ import annotations

from pathlib import Path
from typing import Any
import re


BUMP_CONSTRUCTOR_PATH = Path(".tools/python/bump_constructor.py")
BUMP_VERSION_PATH = Path(".tools/python/bump_version.py")


def detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def update_bump_constructor(repo_root: Path) -> bool:
    path = repo_root / BUMP_CONSTRUCTOR_PATH
    if not path.exists():
        print(f"Skipping missing file: {BUMP_CONSTRUCTOR_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)
    lines = original_text.splitlines(keepends=True)
    changed = False

    if "DEBUG_PREFIX = \"[bump_constructor]\"" not in original_text:
        for index, line in enumerate(lines):
            if line.rstrip("\r\n") == "import re":
                helper_lines = [
                    "",
                    'DEBUG_PREFIX = "[bump_constructor]"',
                    "",
                    "",
                    "def debug(message: str) -> None:",
                    '    print(f"{DEBUG_PREFIX} {message}")',
                    "",
                ]
                insertion = [entry + newline for entry in helper_lines]
                for offset, entry in enumerate(insertion, start=1):
                    lines.insert(index + offset, entry)
                changed = True
                break
        else:
            raise ValueError(f"Unable to find import insertion point in {BUMP_CONSTRUCTOR_PATH}")

    if 'debug(f"Using project folder: {project_folder}")' not in "".join(lines):
        for index, line in enumerate(lines):
            if line.rstrip("\r\n") == "    project_folder = extract_project_folder(extra_files)":
                lines.insert(index + 1, '    debug(f"Using project folder: {project_folder}")' + newline)
                changed = True
                break
        else:
            raise ValueError(f"Unable to find project folder insertion point in {BUMP_CONSTRUCTOR_PATH}")

    summary_marker = '    debug(f"Final extra_files entries: {len(normalized_items)}")'
    if summary_marker not in "".join(lines):
        summary_lines = [
            '    debug(f"Final extra_files entries: {len(normalized_items)}")',
            "    for item in normalized_items:",
            '        debug(f"extra_files entry: {item}")',
        ]
        for index, line in enumerate(lines):
            if line.rstrip("\r\n") == "    return ntbk_added, src_added":
                insertion = [entry + newline for entry in summary_lines]
                for offset, entry in enumerate(insertion):
                    lines.insert(index + offset, entry)
                changed = True
                break
        else:
            raise ValueError(f"Unable to find summary insertion point in {BUMP_CONSTRUCTOR_PATH}")

    if not changed:
        return False

    write_text(path, "".join(lines))
    print(f"Updated {BUMP_CONSTRUCTOR_PATH}")
    return True


def update_bump_version(repo_root: Path) -> bool:
    path = repo_root / BUMP_VERSION_PATH
    if not path.exists():
        print(f"Skipping missing file: {BUMP_VERSION_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)
    updated_text = original_text
    changed = False

    old_regex = 'THANKS_LINE_RE = re.compile(r\'(Thank you! You have successfully installed [^\\n]*)(\\d+\\.\\d+\\.\\d+)(!)\')'
    new_regex = newline.join(
        [
            "CONCLUSION_LINE_RE = re.compile(",
            '    r\'^(conclusion_text:\\s*)(?P<quote>["\\\'])(?P<body>.*?)(?P=quote)(?P<trailing>\\s*(?:#.*)?)$\',',
            "    re.MULTILINE,",
            ")",
        ]
    )
    if "CONCLUSION_LINE_RE = re.compile(" not in updated_text:
        if old_regex not in updated_text:
            raise ValueError(f"Unable to find conclusion regex in {BUMP_VERSION_PATH}")
        updated_text = updated_text.replace(old_regex, new_regex, 1)
        changed = True

    new_function = newline.join(
        [
            "def bump_construct_text(text: str, old_version: str, new_version: str) -> str:",
            '    """Return updated file text with bumped version, preserving YAML structure/comments.',
            "",
            "    - Updates the `version: \"X.Y.Z\"` line.",
            "    - Updates the version inside `conclusion_text`, replacing either",
            "      `VERSION_NUMBER` or the previous explicit version if present.",
            '    """',
            "    # 1) Update the explicit version line",
            "    def _repl_version(m: re.Match) -> str:",
            "        # Preserve original quoting and trailing comment/whitespace",
            "        return f\"{m.group(1)}{m.group('quote')}{new_version}{m.group('quote')}{m.group('trailing')}\"",
            "",
            "    text = VERSION_LINE_RE.sub(_repl_version, text, count=1)",
            "",
            "    # 2) Update the version inside conclusion_text if present.",
            "    def _repl_conclusion(m: re.Match) -> str:",
            "        body = m.group(\"body\")",
            "        if \"VERSION_NUMBER\" in body:",
            "            updated_body = body.replace(\"VERSION_NUMBER\", new_version)",
            "        else:",
            "            updated_body = body.replace(old_version, new_version)",
            "        return f\"{m.group(1)}{m.group('quote')}{updated_body}{m.group('quote')}{m.group('trailing')}\"",
            "",
            "    text = CONCLUSION_LINE_RE.sub(_repl_conclusion, text, count=1)",
            "    return text",
        ]
    ) + newline

    if "text = CONCLUSION_LINE_RE.sub(_repl_conclusion, text, count=1)" not in updated_text:
        start_marker = "def bump_construct_text(text: str, old_version: str, new_version: str) -> str:"
        end_marker = "def bump_post_install_bat(new_version: str) -> None:"
        start_index = updated_text.find(start_marker)
        end_index = updated_text.find(end_marker)
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            raise ValueError(f"Unable to replace bump_construct_text() in {BUMP_VERSION_PATH}")
        updated_text = updated_text[:start_index] + new_function + updated_text[end_index:]
        changed = True

    if not changed:
        return False

    write_text(path, updated_text)
    print(f"Updated {BUMP_VERSION_PATH}")
    return True


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    _ = context

    changed_any = False
    changed_any = update_bump_constructor(repo_root) or changed_any
    changed_any = update_bump_version(repo_root) or changed_any

    if not changed_any:
        print("No repository changes were required for this migration.")
