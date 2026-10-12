#!/usr/bin/env python3

"""
Repository verification script for spectra_pure.

Checks:
    ✓ Repository structure
    ✓ Python syntax
    ✓ Local imports
    ✓ Required runtime files
    ✓ Absolute paths
    ✓ Importability of core modules

Ignored:
    .git
    __pycache__
    archive
    *_delete
"""

from pathlib import Path
import ast
import py_compile
import importlib
import re
import sys

ROOT = Path(__file__).resolve().parent

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "archive",
    "models_delete",
    "utils_delete",
}

REQUIRED_DIRS = [
    "models",
    "utils",
    "scripts",
    "data",
    "checkpoints",
]

REQUIRED_FILES = [
    "checkpoints/best_model.pt",
    "checkpoints/tokenizer.pkl",
    "data/processed/curated_subset_named.parquet",
]

CORE_IMPORTS = [
    "models.molclip_model",
    "utils.image_retrieval_dataset",
    "utils.smiles_tokenizer",
]

ABSOLUTE_PATTERNS = [
    "/mnt/",
    "C:\\\\",
    "/Users/",
    "/home/",
]

python_files = []

for path in ROOT.rglob("*.py"):

    parts = set(path.parts)

    if parts & IGNORE_DIRS:
        continue

    python_files.append(path)

python_files = sorted(python_files)

print("=" * 70)
print("Verifying repository:", ROOT.name)
print("=" * 70)

# ==========================================================
# helper
# ==========================================================

errors = {
    "structure": [],
    "syntax": [],
    "imports": [],
    "files": [],
    "paths": [],
    "core": [],
}


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

# ==========================================================
# repository structure
# ==========================================================

section("Checking repository structure")

for d in REQUIRED_DIRS:

    if not (ROOT / d).exists():
        errors["structure"].append(d)

if errors["structure"]:
    print("Missing directories:")
    for d in errors["structure"]:
        print("   ", d)
else:
    print("✓ Structure OK")


# ==========================================================
# required runtime files
# ==========================================================

section("Checking required runtime files")

for f in REQUIRED_FILES:

    if not (ROOT / f).exists():
        errors["files"].append(f)

if errors["files"]:
    print("Missing files:")
    for f in errors["files"]:
        print("   ", f)
else:
    print("✓ Required files found")


# ==========================================================
# syntax
# ==========================================================

section("Checking Python syntax")

for file in python_files:

    try:
        py_compile.compile(
            str(file),
            doraise=True
        )

    except Exception as e:

        errors["syntax"].append(
            (file.relative_to(ROOT), str(e))
        )

if errors["syntax"]:

    for file, err in errors["syntax"]:
        print(file)
        print("   ", err)

else:
    print("✓ No syntax errors")


# ==========================================================
# local imports
# ==========================================================

section("Checking local imports")

for file in python_files:

    source = file.read_text(
        encoding="utf8",
        errors="ignore"
    )

    try:
        tree = ast.parse(source)

    except Exception:
        continue

    for node in ast.walk(tree):

        if isinstance(node, ast.ImportFrom):

            if node.module is None:
                continue

            module = node.module

            if module.startswith("models."):

                rel = Path(
                    *module.split(".")
                ).with_suffix(".py")

                if not (ROOT / rel).exists():

                    errors["imports"].append(
                        (
                            file.relative_to(ROOT),
                            module
                        )
                    )

            elif module.startswith("utils."):

                rel = Path(
                    *module.split(".")
                ).with_suffix(".py")

                if not (ROOT / rel).exists():

                    errors["imports"].append(
                        (
                            file.relative_to(ROOT),
                            module
                        )
                    )

if errors["imports"]:

    for file, mod in errors["imports"]:
        print(file, "->", mod)

else:
    print("✓ Local imports OK")


# ==========================================================
# absolute paths
# ==========================================================

section("Checking hard-coded absolute paths")

for file in python_files:

    txt = file.read_text(
        encoding="utf8",
        errors="ignore"
    )

    for pattern in ABSOLUTE_PATTERNS:

        if pattern in txt:

            errors["paths"].append(
                file.relative_to(ROOT)
            )

            break

if errors["paths"]:

    for f in sorted(set(errors["paths"])):
        print(f)

else:
    print("✓ No absolute paths found")


# ==========================================================
# importability
# ==========================================================

section("Importing core modules")

sys.path.insert(
    0,
    str(ROOT)
)

for module in CORE_IMPORTS:

    try:

        importlib.import_module(module)

        print("✓", module)

    except Exception as e:

        errors["core"].append(
            (
                module,
                str(e)
            )
        )

        print("✗", module)

# ==========================================================
# FINAL SUMMARY
# ==========================================================

section("SUMMARY")

print(f"Python files checked : {len(python_files)}")
print(f"Missing directories  : {len(errors['structure'])}")
print(f"Missing imports      : {len(errors['imports'])}")
print(f"Missing files        : {len(errors['files'])}")
print(f"Syntax errors        : {len(errors['syntax'])}")
print(f"Absolute paths       : {len(errors['paths'])}")
print(f"Core import failures : {len(errors['core'])}")

total_errors = sum(
    len(v)
    for v in errors.values()
)

print()

if total_errors == 0:

    print("✓ Repository is self-contained and ready for GitHub.")

else:

    print("❌ Repository is NOT yet self-contained.")

    if errors["structure"]:
        print("\nMissing directories:")
        for d in errors["structure"]:
            print("  -", d)

    if errors["files"]:
        print("\nMissing runtime files:")
        for f in errors["files"]:
            print("  -", f)

    if errors["imports"]:
        print("\nBroken local imports:")
        for file, mod in errors["imports"]:
            print(f"  - {file} -> {mod}")

    if errors["syntax"]:
        print("\nSyntax errors:")
        for file, err in errors["syntax"]:
            print(f"  - {file}")
            print(f"      {err}")

    if errors["paths"]:
        print("\nFiles containing absolute paths:")
        for f in sorted(set(errors["paths"])):
            print("  -", f)

    if errors["core"]:
        print("\nCore module import failures:")
        for mod, err in errors["core"]:
            print(f"  - {mod}")
            print(f"      {err}")

print("\nDone.")
