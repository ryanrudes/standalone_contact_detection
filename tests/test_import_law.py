"""The import law, machine-checked: the estimator cannot see the oracle.

THEORY.md §9's protocol — truth is manufactured, withheld from the detector, and spent
only on scoring — is enforced here as a *structural* property of the package tree, the
same move DESIGN.md made for capabilities (an absent capability is a structural no-op,
not a tested convention):

1. No module in `contact/` references the `oracle` package, at any scope. The detector
   cannot import the thing that knows the answers.
2. At module scope, `contact/` imports only the method's substrate — stdlib, numpy,
   scipy, the vendored markovlib, and itself. The heavyweight apparatus (mujoco,
   matplotlib) and the opt-in solver stacks (cvxpy, coal) may appear only as
   function-local imports on the code paths that actually use them, so
   ``import contact`` stays light.
"""

from __future__ import annotations

import ast
import pathlib
import sys

CONTACT_DIR = pathlib.Path(__file__).resolve().parent.parent / "contact"

#: What the estimator may import at module scope (plus the standard library).
MODULE_SCOPE_ALLOWED = {"numpy", "scipy", "markovlib", "contact"}

#: Never importable from contact/, at any scope.
FORBIDDEN_EVERYWHERE = {"oracle"}


def _top_package(node: ast.Import | ast.ImportFrom, module_path: pathlib.Path) -> list[str]:
    """The top-level package name(s) an import node pulls in ('' for relative)."""
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if node.level and node.level > 0:  # relative import -> within contact/
        return ["contact"]
    return [node.module.split(".")[0]] if node.module else []


def _walk_imports(tree: ast.AST):
    """Yield (node, is_module_scope) for every import in the tree."""
    module_scope_ids = {id(n) for n in ast.iter_child_nodes(tree)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node, id(node) in module_scope_ids


def test_contact_never_references_oracle():
    offenders = []
    for path in sorted(CONTACT_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node, _ in _walk_imports(tree):
            for pkg in _top_package(node, path):
                if pkg in FORBIDDEN_EVERYWHERE:
                    offenders.append(f"{path.name}:{node.lineno} imports {pkg}")
    assert not offenders, "the estimator imported the oracle:\n" + "\n".join(offenders)


def test_contact_module_scope_imports_are_method_substrate_only():
    stdlib = set(sys.stdlib_module_names)
    offenders = []
    for path in sorted(CONTACT_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node, at_module_scope in _walk_imports(tree):
            if not at_module_scope:
                continue
            for pkg in _top_package(node, path):
                if pkg not in MODULE_SCOPE_ALLOWED and pkg not in stdlib:
                    offenders.append(f"{path.name}:{node.lineno} imports {pkg} at module scope")
    assert not offenders, (
        "contact/ grew a module-scope heavyweight import (make it function-local):\n"
        + "\n".join(offenders)
    )


def test_import_contact_loads_no_apparatus(fresh: bool = True):
    """Importing the estimator must not load the oracle stack into the process."""
    import subprocess

    code = (
        "import sys, contact; "
        "bad = [m for m in ('oracle','mujoco','matplotlib','cvxpy','coal') if m in sys.modules]; "
        "raise SystemExit(', '.join(bad) if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"import contact pulled in: {proc.stderr.strip() or proc.stdout}"
