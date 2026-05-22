from __future__ import annotations

from pathlib import Path
from typing import Any


WELCOME_TEMPLATE_PATH = Path(".tools/templates/Welcome_template.ipynb")
WELCOME_NOTEBOOK_PATH = Path("app/menuinst/Welcome.ipynb")

OLD_NOTEBOOK_URL = (
    '    "                                notebook_url = '
    'f\\"https://api.github.com/repos/{github_owner}/{github_repo_name}'
    '/contents/notebooks/{main_folder}/{subfolder}/{subfolder}.ipynb'
    '?ref={github_branch}\\"\\n",'
)
NEW_NOTEBOOK_URL = (
    '    "                                notebook_url = '
    'f\\"https://api.github.com/repos/{github_owner}/{github_repo_name}'
    '/contents/notebooks/{main_folder}/{subfolder}.ipynb'
    '?ref={github_branch}\\"\\n",'
)


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

    original_text = path.read_text(encoding="utf-8")
    if old in original_text:
        path.write_text(original_text.replace(old, new, 1), encoding="utf-8")
        print(f"Updated {relative_path}")
        return True

    if already_updated and already_updated in original_text:
        return False

    raise ValueError(f"Unable to find expected text in {relative_path}")


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    _ = context

    changed_any = False
    changed_any = replace_text(
        repo_root,
        WELCOME_TEMPLATE_PATH,
        OLD_NOTEBOOK_URL,
        NEW_NOTEBOOK_URL,
        already_updated=NEW_NOTEBOOK_URL,
    ) or changed_any
    changed_any = replace_text(
        repo_root,
        WELCOME_NOTEBOOK_PATH,
        OLD_NOTEBOOK_URL,
        NEW_NOTEBOOK_URL,
        already_updated=NEW_NOTEBOOK_URL,
    ) or changed_any

    if not changed_any:
        print("No repository changes were required for this migration.")
