"""The README's package map, machine-checked against the actual tree.

The repo's organization lives in the README's module tables; this locks them to reality
in both directions, so a module added, removed, or renamed without updating the map
fails the suite (the drift this repo once accumulated silently — six undocumented
modules — can't come back):

* every public module of `contact/` and `oracle/` is mentioned in the README;
* every `contact/*.py` / `oracle/*.py` path the README mentions actually exists.

Private modules (leading underscore) and `__init__.py` are exempt from the first check
but still verified to exist if mentioned.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()

MENTIONED = set(re.findall(r"\b(contact|oracle)/([a-z_]+\.py)", README))


def _public_modules(package: str) -> set[tuple[str, str]]:
    return {
        (package, p.name)
        for p in (ROOT / package).glob("*.py")
        if not p.name.startswith("_")
    }


def test_every_public_module_is_on_the_readme_map():
    missing = sorted(
        f"{pkg}/{mod}"
        for pkg, mod in (_public_modules("contact") | _public_modules("oracle"))
        if (pkg, mod) not in MENTIONED
    )
    assert not missing, "public modules absent from the README map: " + ", ".join(missing)


def test_every_readme_module_path_exists():
    stale = sorted(
        f"{pkg}/{mod}" for pkg, mod in MENTIONED if not (ROOT / pkg / mod).exists()
    )
    assert not stale, "README mentions modules that don't exist: " + ", ".join(stale)
