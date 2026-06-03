"""Execute notebook-derived step scripts in original notebook order."""

from __future__ import annotations

from typing import Any

from .config import ProjectConfig, build_execution_context


def run_pipeline(context_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run each generated step script with a shared notebook-style context.

    Steps live as numbered ``*.py`` files under ``steps/``. They are executed
    with ``exec`` so later steps see names defined by earlier ones, matching
    the original notebook cell order.
    """
    config = ProjectConfig()
    context = build_execution_context(config=config, context_overrides=context_overrides)

    # Filenames are prefixed 01_, 02_, ... so lexical sort is run order.
    for step_path in sorted(config.steps_dir.glob('*.py')):
        context['__file__'] = str(step_path)
        source = step_path.read_text()
        exec(compile(source, str(step_path), 'exec'), context)

    return context
