"""Actual published aggregate results, no synthetic empirical substitution."""
from src.public_audit import audit


def test_published_estimates_recalculate_without_credentials(monkeypatch):
    monkeypatch.delenv("DART_API_KEY",raising=False)
    monkeypatch.delenv("OPENDART_API_KEY",raising=False)
    audit()
