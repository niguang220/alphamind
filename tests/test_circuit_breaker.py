"""
Circuit-breaker state machine. Pure synchronous logic: no LLM, no network, no clock sleeps.

This is the component the tool layer leans on to stop hammering a dependency that is already
failing, and until now it had no test at all — which is how the HALF_OPEN probe leak survived.
"""
import os
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from mcp.tool_manager import CircuitBreaker, CircuitState


def test_starts_closed_and_allows():
    b = CircuitBreaker(failure_threshold=3, recovery_s=60)
    assert b.state == CircuitState.CLOSED
    assert b.allow() is True


def test_opens_after_consecutive_failures_and_refuses():
    b = CircuitBreaker(failure_threshold=3, recovery_s=60)
    for _ in range(3):
        b.record_failure()
    assert b.state == CircuitState.OPEN
    assert b.allow() is False


def test_success_resets_the_failure_run():
    """Threshold counts *consecutive* failures; one success must clear the streak."""
    b = CircuitBreaker(failure_threshold=3, recovery_s=60)
    b.record_failure()
    b.record_failure()
    b.record_success()
    b.record_failure()
    b.record_failure()
    assert b.state == CircuitState.CLOSED
    assert b.allow() is True


def test_half_open_admits_exactly_one_probe():
    """
    The bug this locks down: HALF_OPEN used to return True unconditionally, so when the
    recovery window elapsed under concurrent load every waiting caller was admitted at once —
    the stampede the breaker had just opened to prevent.
    """
    b = CircuitBreaker(failure_threshold=1, recovery_s=0)  # recovery_s=0 -> window already elapsed
    b.record_failure()
    assert b.state == CircuitState.OPEN

    assert b.allow() is True            # first caller becomes the probe
    assert b.state == CircuitState.HALF_OPEN
    assert b.allow() is False           # everyone else waits
    assert b.allow() is False


def test_successful_probe_closes_the_breaker():
    b = CircuitBreaker(failure_threshold=1, recovery_s=0)
    b.record_failure()
    assert b.allow() is True
    b.record_success()
    assert b.state == CircuitState.CLOSED
    assert b.allow() is True


def test_failed_probe_reopens_without_waiting_for_the_threshold():
    """A probe that fails is decisive on its own — it must not need `threshold` more failures."""
    b = CircuitBreaker(failure_threshold=5, recovery_s=0)
    for _ in range(5):
        b.record_failure()
    assert b.state == CircuitState.OPEN

    assert b.allow() is True            # probe admitted
    assert b.state == CircuitState.HALF_OPEN
    b.record_failure()                  # probe fails
    assert b.state == CircuitState.OPEN
