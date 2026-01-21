#!/usr/bin/env python3

"""
Script to run UI tests for the enhanced dashboard.
Validates all the fixes and improvements made to the analytics control panel.
"""

import sys
import subprocess
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def install_playwright_browsers():
    """Install Playwright browsers if not already installed."""
    try:
        logger.info("Installing Playwright browsers...")
        subprocess.run(['playwright', 'install', 'chromium'], check=True)
        logger.info("Playwright browsers installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install Playwright browsers: {e}")
        return False
    except FileNotFoundError:
        logger.error("Playwright not found. Please install with: pip install playwright")
        return False


def run_ui_tests():
    """Run the UI tests using pytest."""
    test_file = project_root / 'tests' / 'test_dashboard_ui_fixes.py'
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return False
    
    try:
        logger.info("Running UI tests...")
        
        # Run pytest with specific configuration for UI tests
        cmd = [
            'python', '-m', 'pytest',
            str(test_file),
            '-v',  # Verbose output
            '--tb=short',  # Short traceback format
            '--durations=10',  # Show 10 slowest tests
            '--color=yes',  # Colored output
            '-x',  # Stop on first failure
        ]
        
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ All UI tests passed!")
            logger.info(result.stdout)
            return True
        else:
            logger.error("❌ Some UI tests failed!")
            logger.error(result.stdout)
            logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False


def run_specific_test_category(category):
    """Run a specific category of tests."""
    test_file = project_root / 'tests' / 'test_dashboard_ui_fixes.py'
    
    test_classes = {
        'dropdown': 'TestDropdownZIndex',
        'kpi': 'TestKPICards', 
        'equity': 'TestEquityCurveLoading',
        'scaling': 'TestChartScaling',
        'layout': 'TestDesktopOnlyLayout',
        'metrics': 'TestPerformanceMetricsFormatting',
        'interaction': 'TestInteractivity',
        'rendering': 'TestChartRendering'
    }
    
    if category not in test_classes:
        logger.error(f"Unknown test category: {category}")
        logger.info(f"Available categories: {list(test_classes.keys())}")
        return False
    
    try:
        logger.info(f"Running {category} tests...")
        
        cmd = [
            'python', '-m', 'pytest',
            f"{test_file}::{test_classes[category]}",
            '-v',
            '--tb=short',
            '--color=yes'
        ]
        
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Error running {category} tests: {e}")
        return False


def main():
    """Main function to run UI tests."""
    print("=" * 60)
    print("MELUNA DASHBOARD UI TESTS")
    print("=" * 60)
    print()
    
    # Check if specific test category requested
    if len(sys.argv) > 1:
        category = sys.argv[1].lower()
        if category == 'install':
            success = install_playwright_browsers()
            return 0 if success else 1
        else:
            success = run_specific_test_category(category)
            return 0 if success else 1
    
    # Install browsers first
    print("1. Installing Playwright browsers...")
    if not install_playwright_browsers():
        print("Failed to install browsers. Exiting.")
        return 1
    
    print("\n2. Running UI tests...")
    success = run_ui_tests()
    
    if success:
        print("\n✅ All dashboard fixes verified successfully!")
        print("\nFixed issues:")
        print("- ✅ Dropdown z-index layering")
        print("- ✅ Replaced speedometer cards with clean text displays")  
        print("- ✅ Fixed equity curve loading")
        print("- ✅ Removed mobile responsive complications")
        print("- ✅ Improved chart scaling")
        print("- ✅ Enhanced performance metrics formatting")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)