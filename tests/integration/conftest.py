"""Shared fixtures for integration tests."""

import pytest
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from meluna.utils.binance_client import BinanceClient
from data.binance_handler import BinanceDataHandler


@pytest.fixture(scope='module')
def testnet_client():
    """Create anonymous mainnet client for integration tests."""
    return BinanceClient({'testnet': False})  # Use mainnet for reliable data


@pytest.fixture(scope='module')
def binance_handler(testnet_client, tmp_path_factory):
    """Create BinanceDataHandler with mainnet client and temporary cache."""
    data_path = tmp_path_factory.mktemp("data")
    return BinanceDataHandler(
        ['BTCUSDT'],
        testnet_client,
        interval='1d',
        data_path=str(data_path)
    )
