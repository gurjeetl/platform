"""Feature-flag registry backed by the platform Settings."""
from __future__ import annotations

from genie.platform.config import Settings


class FeatureFlags:
    """Runtime feature flag store initialised from Settings.

    Flags can be toggled at runtime (e.g. for tests or gradual rollouts)
    without restarting the application.
    """

    def __init__(self, settings: Settings) -> None:
        self._flags: dict[str, bool] = {
            "rag": settings.enable_rag,
            "hitl": settings.enable_hitl,
            "tracking": settings.enable_tracking,
        }

    def is_enabled(self, flag: str) -> bool:
        """Return True if *flag* is enabled, False if disabled or unknown."""
        return self._flags.get(flag, False)

    def enable(self, flag: str) -> None:
        """Enable *flag* (creates it if it did not exist)."""
        self._flags[flag] = True

    def disable(self, flag: str) -> None:
        """Disable *flag* (creates it if it did not exist)."""
        self._flags[flag] = False

    def all_flags(self) -> dict[str, bool]:
        """Return a snapshot copy of all current flag values."""
        return dict(self._flags)
