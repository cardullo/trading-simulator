"""Pytest fixtures and shared test configuration."""

import pytest


@pytest.fixture
def default_config():
    """Return a default SimulationConfig for testing."""
    from trading_simulator.config import SimulationConfig
    return SimulationConfig()


@pytest.fixture
def sample_account():
    """Return a sample Account with €1,000 balance."""
    from trading_simulator.models import Account
    return Account(balance=1000.0, initial_balance=1000.0)