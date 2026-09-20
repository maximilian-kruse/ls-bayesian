"""Shared helper for integration tests that execute a tutorial notebook and check its results.

Used by the `optimization`, `posterior`, and `spde_prior` test suites, each of which owns its own
notebook paths and execution timeout (tutorials differ widely in runtime, e.g. FEM assembly and
plotting vs. a plain optimization loop) but shares the same execution-and-probing logic.

Functions:
    execute_notebook_and_extract_values: Execute a tutorial notebook and evaluate expressions
        against its final namespace.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==================================================================================================
def execute_notebook_and_extract_values(
    notebook_path: Path, expressions: Mapping[str, str], *, timeout_seconds: float
) -> dict[str, Any]:
    """Execute a tutorial notebook and evaluate expressions against its final namespace.

    The notebook is executed unmodified in its own kernel, except for one appended code cell that
    evaluates the given expressions and serializes the results to stdout as JSON. Expressions
    reduce large results (e.g. vectors) to compact scalars, such as a norm, so that reference
    values stay small numeric literals in test code rather than stored array data.

    Args:
        notebook_path (Path): Path to the `.ipynb` file to execute.
        expressions (Mapping[str, str]): Mapping from a result key to a Python expression,
            evaluated in the notebook's namespace after all of its own cells have run. Expression
            results must be JSON-serializable.
        timeout_seconds (float): Maximum time to allow the notebook to run, in seconds. Tutorials
            vary widely in cost (e.g. FEM assembly and plotting vs. a plain optimization loop), so
            callers pass a value suited to their own notebook rather than sharing one constant.

    Returns:
        dict[str, Any]: Mapping from result key to its JSON-deserialized value.
    """
    notebook = nbformat.read(notebook_path, as_version=4)
    probe_source = (
        "import json as _json\n"
        f"_probe_values = {{key: eval(expr) for key, expr in {dict(expressions)!r}.items()}}\n"
        "print(_json.dumps(_probe_values))"
    )
    notebook.cells.append(nbformat.v4.new_code_cell(source=probe_source))

    client = NotebookClient(
        notebook,
        timeout=timeout_seconds,
        kernel_name=notebook.metadata["kernelspec"]["name"],
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()

    probe_outputs = notebook.cells[-1]["outputs"]
    stdout_text = "".join(
        output["text"] for output in probe_outputs if output.get("name") == "stdout"
    )
    return json.loads(stdout_text)
