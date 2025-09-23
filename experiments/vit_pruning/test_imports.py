#!/usr/bin/env python3
"""
Test script to verify vit_pruning module can be imported correctly
"""
import sys
import pathlib
import importlib

# Setup project path
proj_root = pathlib.Path(__file__).parent.parent
if not (proj_root / 'src').exists():
    for p in proj_root.parents:
        if (p / 'src').exists():
            proj_root = p
            break

# Add project root to path
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))
    print(f"Added to sys.path: {proj_root}")

# Try different import methods
print("Python version:", sys.version)
print("Import paths:", sys.path)

print("\nTrying import src.vit_pruning:")
try:
    import src.vit_pruning
    print("SUCCESS: Module imported")
    print("__all__:", src.vit_pruning.__all__)
    print("Has prune_vit_attention_blocks:", hasattr(src.vit_pruning, "prune_vit_attention_blocks"))
except Exception as e:
    print(f"ERROR: {str(e)}")

print("\nTrying from src.vit_pruning import prune_vit_attention_blocks:")
try:
    from src.vit_pruning import prune_vit_attention_blocks
    print("SUCCESS: Function imported")
except Exception as e:
    print(f"ERROR: {str(e)}")

print("\nChecking module file location:")
try:
    import src.vit_pruning
    print("Module file:", src.vit_pruning.__file__)
except Exception as e:
    print(f"ERROR: {str(e)}")
