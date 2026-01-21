#!/usr/bin/env python3
# scripts/run_dashboard_tests.py

"""
Dashboard Testing Runner Script

This script provides an easy way to run dashboard tests with different configurations.
Supports running individual test suites, performance tests, and full regression testing.
"""

import sys
import subprocess
import argparse
import asyncio
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

class DashboardTestRunner:
    """Enhanced test runner for dashboard interactive tests."""
    
    def __init__(self):
        self.project_root = project_root
        self.test_dir = self.project_root / "tests"
        
    def run_command(self, cmd: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
        """Run shell command with proper error handling."""
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                check=False
            )
            return result
        except Exception as e:
            print(f"Error running command: {e}")
            sys.exit(1)
    
    def install_dependencies(self):
        """Install required test dependencies."""
        print("Installing test dependencies...")
        
        dependencies = [
            "pytest",
            "pytest-asyncio", 
            "pytest-cov",
            "pytest-timeout",
            "playwright",
            "pytest-playwright"
        ]
        
        for dep in dependencies:
            cmd = [sys.executable, "-m", "pip", "install", dep]
            result = self.run_command(cmd, capture_output=True)
            if result.returncode != 0:
                print(f"Failed to install {dep}: {result.stderr}")
            else:
                print(f"✓ Installed {dep}")
        
        # Install playwright browsers
        print("Installing Playwright browsers...")
        playwright_cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
        result = self.run_command(playwright_cmd, capture_output=True)
        if result.returncode == 0:
            print("✓ Installed Playwright browsers")
        else:
            print(f"Failed to install Playwright browsers: {result.stderr}")
    
    def run_unit_tests(self, verbose: bool = True):
        """Run unit tests only."""
        print("\n" + "="*60)
        print("RUNNING UNIT TESTS")
        print("="*60)
        
        cmd = [sys.executable, "-m", "pytest", "tests/", "-m", "unit"]
        if verbose:
            cmd.extend(["-v", "-s"])
        
        result = self.run_command(cmd)
        return result.returncode == 0
    
    def run_interactive_tests(self, headless: bool = True, verbose: bool = True):
        """Run interactive dashboard tests."""
        print("\n" + "="*60)
        print("RUNNING INTERACTIVE DASHBOARD TESTS")
        print("="*60)
        
        cmd = [sys.executable, "-m", "pytest", "tests/test_interactive_dashboard.py"]
        if verbose:
            cmd.extend(["-v", "-s"])
        if headless:
            cmd.extend(["--headless"])
        
        result = self.run_command(cmd)
        return result.returncode == 0
    
    def run_performance_tests(self, verbose: bool = True):
        """Run performance-focused tests."""
        print("\n" + "="*60)
        print("RUNNING PERFORMANCE TESTS")
        print("="*60)
        
        cmd = [sys.executable, "-m", "pytest", "tests/", "-m", "performance"]
        if verbose:
            cmd.extend(["-v", "-s"])
        
        result = self.run_command(cmd)
        return result.returncode == 0
    
    def run_regression_tests(self, headless: bool = True, verbose: bool = True):
        """Run full regression test suite."""
        print("\n" + "="*60)
        print("RUNNING FULL REGRESSION TEST SUITE")
        print("="*60)
        
        cmd = [sys.executable, "-m", "pytest", "tests/"]
        if verbose:
            cmd.extend(["-v", "-s"])
        if headless:
            cmd.extend(["--headless"])
        
        # Add coverage reporting
        cmd.extend([
            "--cov=meluna",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing"
        ])
        
        result = self.run_command(cmd)
        
        if result.returncode == 0:
            print("\n✓ All tests passed!")
            print("Coverage report generated in htmlcov/index.html")
        else:
            print("\n✗ Some tests failed!")
        
        return result.returncode == 0
    
    def run_specific_test(self, test_path: str, verbose: bool = True):
        """Run a specific test file or test method."""
        print(f"\n" + "="*60)
        print(f"RUNNING SPECIFIC TEST: {test_path}")
        print("="*60)
        
        cmd = [sys.executable, "-m", "pytest", test_path]
        if verbose:
            cmd.extend(["-v", "-s"])
        
        result = self.run_command(cmd)
        return result.returncode == 0
    
    def run_smoke_tests(self):
        """Run quick smoke tests to verify basic functionality."""
        print("\n" + "="*60)
        print("RUNNING SMOKE TESTS")
        print("="*60)
        
        # Run a subset of critical tests
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/test_interactive_dashboard.py::TestChartInteractivity::test_equity_curve_zoom_functionality",
            "-v", "-s"
        ]
        
        result = self.run_command(cmd)
        return result.returncode == 0
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*60)
        print("GENERATING TEST REPORT")
        print("="*60)
        
        cmd = [
            sys.executable, "-m", "pytest", "tests/",
            "--html=reports/test_report.html",
            "--self-contained-html",
            "--junitxml=reports/junit.xml",
            "--cov=meluna",
            "--cov-report=html:reports/coverage"
        ]
        
        # Create reports directory
        reports_dir = self.project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        result = self.run_command(cmd)
        
        if result.returncode == 0:
            print("\n✓ Test report generated successfully!")
            print(f"HTML Report: {reports_dir}/test_report.html")
            print(f"Coverage Report: {reports_dir}/coverage/index.html")
            print(f"JUnit XML: {reports_dir}/junit.xml")
        
        return result.returncode == 0

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced Dashboard Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_dashboard_tests.py install        # Install dependencies
  python scripts/run_dashboard_tests.py smoke         # Quick smoke test
  python scripts/run_dashboard_tests.py interactive   # Interactive tests
  python scripts/run_dashboard_tests.py performance   # Performance tests
  python scripts/run_dashboard_tests.py regression    # Full test suite
  python scripts/run_dashboard_tests.py specific tests/test_dashboard.py::TestClass::test_method
        """
    )
    
    parser.add_argument(
        "command",
        choices=["install", "smoke", "unit", "interactive", "performance", "regression", "specific", "report"],
        help="Test command to run"
    )
    
    parser.add_argument(
        "test_path",
        nargs="?",
        help="Specific test path (for 'specific' command)"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser tests in headless mode (default: True)"
    )
    
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser tests with visible browser"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Handle headless flag
    headless = args.headless and not args.no_headless
    verbose = not args.quiet
    
    runner = DashboardTestRunner()
    
    success = True
    
    if args.command == "install":
        runner.install_dependencies()
    
    elif args.command == "smoke":
        success = runner.run_smoke_tests()
    
    elif args.command == "unit":
        success = runner.run_unit_tests(verbose=verbose)
    
    elif args.command == "interactive":
        success = runner.run_interactive_tests(headless=headless, verbose=verbose)
    
    elif args.command == "performance":
        success = runner.run_performance_tests(verbose=verbose)
    
    elif args.command == "regression":
        success = runner.run_regression_tests(headless=headless, verbose=verbose)
    
    elif args.command == "specific":
        if not args.test_path:
            print("Error: test_path required for 'specific' command")
            sys.exit(1)
        success = runner.run_specific_test(args.test_path, verbose=verbose)
    
    elif args.command == "report":
        success = runner.generate_test_report()
    
    if success:
        print("\n🎉 Test execution completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test execution failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()