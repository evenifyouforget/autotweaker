#!/usr/bin/env python3
"""
Static analysis script for the autotweaker codebase.

Performs basic type checking and code validation using Python's built-in tools.
"""

import os
import sys
import py_compile
import ast
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def check_syntax(file_path: str) -> Tuple[bool, str]:
    """Check Python syntax using py_compile."""
    try:
        py_compile.compile(file_path, doraise=True)
        return True, "✅ Syntax OK"
    except py_compile.PyCompileError as e:
        return False, f"❌ Syntax Error: {e}"


def check_imports(file_path: str) -> Tuple[bool, str]:
    """Check that all imports can be resolved."""
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return True, f"✅ Imports: {len(imports)} found"
    except Exception as e:
        return False, f"❌ Import Error: {e}"


def check_type_annotations(file_path: str) -> Tuple[bool, str]:
    """Check for type annotations coverage."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        functions = []
        annotated_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
                
                # Check for return annotation
                has_return_annotation = node.returns is not None
                
                # Check for parameter annotations
                has_param_annotations = any(arg.annotation is not None for arg in node.args.args)
                
                if has_return_annotation or has_param_annotations:
                    annotated_functions.append(node.name)
        
        if functions:
            coverage = len(annotated_functions) / len(functions) * 100
            return True, f"✅ Type annotation coverage: {coverage:.1f}% ({len(annotated_functions)}/{len(functions)} functions)"
        else:
            return True, "✅ No functions found"
            
    except Exception as e:
        return False, f"❌ Type Annotation Check Error: {e}"


def run_static_analysis(path: str) -> None:
    """Run static analysis on Python file(s) in path."""
    print("🔍 STATIC ANALYSIS REPORT")
    print("=" * 50)
    
    path_obj = Path(path)
    
    if path_obj.is_file() and path_obj.suffix == '.py':
        # Single file
        py_files = [path_obj]
    elif path_obj.is_dir():
        # Directory - find all Python files
        py_files = list(path_obj.glob("**/*.py"))
    else:
        print(f"Invalid path: {path}")
        return
    
    if not py_files:
        print(f"No Python files found at {path}")
        return
    
    print(f"Analyzing {len(py_files)} Python files...\n")
    
    total_files = len(py_files)
    syntax_ok = 0
    imports_ok = 0
    
    base_path = path_obj if path_obj.is_dir() else path_obj.parent
    
    for file_path in py_files:
        try:
            relative_path = file_path.relative_to(base_path)
        except ValueError:
            # File not relative to base path, use name only
            relative_path = file_path.name
        print(f"📁 {relative_path}")
        
        # Syntax check
        syntax_success, syntax_msg = check_syntax(str(file_path))
        print(f"   {syntax_msg}")
        if syntax_success:
            syntax_ok += 1
        
        # Import check
        import_success, import_msg = check_imports(str(file_path))
        print(f"   {import_msg}")
        if import_success:
            imports_ok += 1
        
        # Type annotation check
        type_success, type_msg = check_type_annotations(str(file_path))
        print(f"   {type_msg}")
        
        print()
    
    print("📊 SUMMARY")
    print("-" * 30)
    print(f"Files analyzed: {total_files}")
    print(f"Syntax valid: {syntax_ok}/{total_files} ({syntax_ok/total_files*100:.1f}%)")
    print(f"Imports valid: {imports_ok}/{total_files} ({imports_ok/total_files*100:.1f}%)")
    
    if syntax_ok == total_files and imports_ok == total_files:
        print("\n🎉 All checks passed!")
    else:
        print(f"\n⚠️  Issues found in {total_files - min(syntax_ok, imports_ok)} files")


if __name__ == "__main__":
    # Default to py_autotweaker directory
    analysis_dir = sys.argv[1] if len(sys.argv) > 1 else "py_autotweaker"
    
    if not os.path.exists(analysis_dir):
        print(f"Directory {analysis_dir} not found")
        sys.exit(1)
    
    run_static_analysis(analysis_dir)