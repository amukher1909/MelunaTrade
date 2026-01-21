#!/usr/bin/env python3
"""
Pytest fixtures for UAT test suite.
"""

import pytest
import pandas as pd
from pathlib import Path
from tests.uat.excel_exporter import UATExcelExporter


@pytest.fixture(scope="session")
def uat_exporter():
    """
    Session-scoped Excel exporter for Margin & Position Management UAT (Issue #151).

    Creates a single Excel workbook for the entire test session,
    then saves it after all tests complete.

    Yields:
        UATExcelExporter instance
    """
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'results/uat/margin_position_uat_{timestamp}.xlsx'

    exporter = UATExcelExporter(output_path)

    yield exporter

    # After all tests complete, generate summary and save
    exporter.add_summary_dashboard()
    exporter.save()


@pytest.fixture(scope="session")
def liquidation_uat_exporter():
    """
    Session-scoped Excel exporter for Liquidation Mechanics UAT (Issue #152).

    Creates a single Excel workbook for the entire test session,
    then saves it after all tests complete.

    Yields:
        UATExcelExporter instance
    """
    timestamp = pd.Timestamp.now().strftime("%Y%m%d")
    output_path = f'results/uat/liquidation_mechanics_uat_{timestamp}.xlsx'

    exporter = UATExcelExporter(output_path, title="LIQUIDATION MECHANICS UAT")

    yield exporter

    # After all tests complete, generate summary and save
    exporter.add_summary_dashboard()
    exporter.save()
