"""Command-line entry points for the futures roll analysis package."""

from .daily import main as daily_main
from .hourly import main as hourly_main
from .organize import main as organize_main

__all__ = ["daily_main", "hourly_main", "organize_main"]
