#!/usr/bin/env python3
"""
Test runner for Composite Technical Indicators.

This script provides an easy way to run all composite indicator tests
and generate a comprehensive report.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_tests(test_file: str, verbose: bool = True) -> dict:
    """Run tests for a specific file and return results."""
    print(f"\n{'='*60}")
    print(f"Running tests for: {test_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run pytest with the specified file
        cmd = [sys.executable, "-m", "pytest", test_file, "-v" if verbose else ""]
        cmd = [arg for arg in cmd if arg]  # Remove empty strings
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse results
        if result.returncode == 0:
            status = "PASSED"
            # Extract test summary from output
            lines = result.stdout.split('\n')
            summary_line = [line for line in lines if 'passed' in line.lower() and 'failed' in line.lower()]
            summary = summary_line[0] if summary_line else "All tests passed"
        else:
            status = "FAILED"
            summary = result.stderr if result.stderr else "Tests failed"
        
        return {
            'file': test_file,
            'status': status,
            'duration': duration,
            'summary': summary,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
        
    except Exception as e:
        return {
            'file': test_file,
            'status': 'ERROR',
            'duration': 0,
            'summary': f"Exception: {str(e)}",
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }

def main():
    """Main test runner function."""
    print("🧪 COMPOSITE INDICATORS TEST RUNNER")
    print("=" * 60)
    
    # Test files to run
    test_files = [
        "tests/technical_analysis/test_composite.py",
        "tests/technical_analysis/test_composite_pandas_validation.py"
    ]
    
    # Check if test files exist
    existing_tests = []
    for test_file in test_files:
        if os.path.exists(test_file):
            existing_tests.append(test_file)
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    if not existing_tests:
        print("❌ No test files found!")
        return 1
    
    print(f"Found {len(existing_tests)} test file(s)")
    
    # Run tests
    results = []
    total_duration = 0
    
    for test_file in existing_tests:
        result = run_tests(test_file)
        results.append(result)
        total_duration += result['duration']
        
        # Print immediate result
        status_icon = "✅" if result['status'] == 'PASSED' else "❌" if result['status'] == 'FAILED' else "💥"
        print(f"{status_icon} {result['file']}: {result['status']} ({result['duration']:.2f}s)")
        if result['status'] != 'PASSED':
            print(f"   Summary: {result['summary']}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for r in results if r['status'] == 'PASSED')
    failed = sum(1 for r in results if r['status'] == 'FAILED')
    errors = sum(1 for r in results if r['status'] == 'ERROR')
    
    print(f"📊 Results: {passed} passed, {failed} failed, {errors} errors")
    print(f"⏱️  Total duration: {total_duration:.2f} seconds")
    
    # Detailed results
    for result in results:
        status_icon = "✅" if result['status'] == 'PASSED' else "❌" if result['status'] == 'FAILED' else "💥"
        print(f"\n{status_icon} {result['file']}")
        print(f"   Status: {result['status']}")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Summary: {result['summary']}")
        
        if result['status'] != 'PASSED' and result['stderr']:
            print(f"   Error Details: {result['stderr'][:200]}...")
    
    print(f"\n{'='*60}")
    
    # Return appropriate exit code
    if failed > 0 or errors > 0:
        print("❌ Some tests failed!")
        return 1
    else:
        print("✅ All tests passed!")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
