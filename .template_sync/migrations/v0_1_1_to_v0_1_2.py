from __future__ import annotations

from pathlib import Path
from typing import Any


SRC_WORKFLOW_PATH = Path(".github/workflows/update_on_src_change.yaml")
BUMP_CONSTRUCTOR_PATH = Path(".tools/python/bump_constructor.py")
WELCOME_TEMPLATE_PATH = Path(".tools/templates/Welcome_template.ipynb")
EXTERNAL_CODE_DOC_PATH = Path(".tools/docs/external_code_upload.md")


def detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def replace_text(
    repo_root: Path,
    relative_path: Path,
    old: str,
    new: str,
    *,
    already_updated: str | None = None,
) -> bool:
    path = repo_root / relative_path
    if not path.exists():
        print(f"Skipping missing file: {relative_path}")
        return False

    original_text = read_text(path)
    if old in original_text:
        updated_text = original_text.replace(old, new, 1)
        write_text(path, updated_text)
        print(f"Updated {relative_path}")
        return True

    if already_updated and already_updated in original_text:
        return False

    raise ValueError(f"Unable to find expected text in {relative_path}")


def update_src_change_workflow(repo_root: Path) -> bool:
    return replace_text(
        repo_root,
        SRC_WORKFLOW_PATH,
        "      - 'src/**/*.py'",
        "      - 'src/**'",
        already_updated="      - 'src/**'",
    )


def update_bump_constructor(repo_root: Path) -> bool:
    path = repo_root / BUMP_CONSTRUCTOR_PATH
    if not path.exists():
        print(f"Skipping missing file: {BUMP_CONSTRUCTOR_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)
    updated_text = original_text

    old_dst = '        dst = f"{project_folder}/src/{project_name}/{rel.replace(\'src/\', \'\')}"'
    new_dst = '        dst = f"{project_folder}/{rel}"'
    if old_dst in updated_text:
        updated_text = updated_text.replace(old_dst, new_dst, 1)
    elif new_dst not in updated_text:
        raise ValueError(f"Unable to find constructor src destination mapping in {BUMP_CONSTRUCTOR_PATH}")

    # Match the small formatting cleanup made in the template update.
    updated_text = updated_text.replace(
        f"import re{newline}{newline}{newline}def load_construct",
        f"import re{newline}{newline}def load_construct",
        1,
    )

    if updated_text == original_text:
        return False

    write_text(path, updated_text)
    print(f"Updated {BUMP_CONSTRUCTOR_PATH}")
    return True


def update_welcome_template(repo_root: Path) -> bool:
    return replace_text(
        repo_root,
        WELCOME_TEMPLATE_PATH,
        '    "        src_folder = Path(\\"..\\") / \\"src\\" / \\"PYTHON_PROJ_NAME\\"\\n",',
        '    "        src_folder = Path(\\"..\\") / \\"src\\"\\n",',
        already_updated='    "        src_folder = Path(\\"..\\") / \\"src\\"\\n",',
    )


def update_external_code_docs(repo_root: Path) -> bool:
    path = repo_root / EXTERNAL_CODE_DOC_PATH
    if not path.exists():
        print(f"Skipping missing file: {EXTERNAL_CODE_DOC_PATH}")
        return False

    original_text = read_text(path)
    newline = detect_newline(original_text)

    changed = False
    updated_text = original_text

    old_tree = "\n".join(
        [
            "```",
            "src",
            "|-- __init__.py",
            "|-- my_script.py",
            "|-- subpackage/",
            "    |-- __init__.py",
            "    |-- submodule1.py",
            "```",
        ]
    ).replace("\n", newline)
    new_tree = "\n".join(
        [
            "```text",
            "src/",
            "|-- PYTHON_PROJ_NAME/",
            "|   |-- __init__.py",
            "|   |-- my_script.py",
            "|   |-- subpackage/",
            "|       |-- __init__.py",
            "```",
        ]
    ).replace("\n", newline)

    if old_tree in updated_text:
        updated_text = updated_text.replace(old_tree, new_tree, 1)
        changed = True
    elif new_tree not in updated_text:
        raise ValueError(f"Unable to find directory tree block in {EXTERNAL_CODE_DOC_PATH}")

    replacements = [
        ("# src/__init__.py", "# src/PYTHON_PROJ_NAME/__init__.py"),
        ("from PYTHON_PROJ_NAME import subpackage", "from PYTHON_PROJ_NAME.subpackage import submodule1"),
    ]
    for old, new in replacements:
        if old in updated_text:
            updated_text = updated_text.replace(old, new, 1)
            changed = True
        elif new not in updated_text:
            raise ValueError(f"Unable to find expected text in {EXTERNAL_CODE_DOC_PATH}: {old}")

    if not changed:
        return False

    write_text(path, updated_text)
    print(f"Updated {EXTERNAL_CODE_DOC_PATH}")
    return True


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    _ = context

    changed_any = False
    changed_any = update_src_change_workflow(repo_root) or changed_any
    changed_any = update_bump_constructor(repo_root) or changed_any
    changed_any = update_welcome_template(repo_root) or changed_any
    changed_any = update_external_code_docs(repo_root) or changed_any

    if not changed_any:
        print("No repository changes were required for this migration.")
