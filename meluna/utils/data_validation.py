# meluna/utils/data_validation.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation errors."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationError:
    """
    Structured validation error.

    Attributes:
        rule: Validation rule that failed (e.g., 'high_low_violation')
        severity: ValidationSeverity enum
        bar_index: Index of problematic bar in DataFrame
        message: Human-readable description
        timestamp: Timestamp of problematic bar (if available)
        details: Additional context (dict)
    """
    rule: str
    severity: ValidationSeverity
    bar_index: int
    message: str
    timestamp: Optional[pd.Timestamp] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (
            f"ValidationError(rule='{self.rule}', severity={self.severity.value}, "
            f"bar_index={self.bar_index}, message='{self.message}')"
        )


class DataValidator:
    """
    OHLCV data quality validator.

    Validates OHLC relationships, value ranges, timestamp continuity, and volume sanity.

    Args:
        interval: Timeframe for gap detection (e.g., '1d', '1h')
        strictness: 'warn' (log warnings) or 'error' (raise exception)
        gap_tolerance_multiplier: Multiplier for expected interval (default 1.5x)

    Usage:
        validator = DataValidator(interval='1d', strictness='warn')
        errors = validator.validate(df)
        if errors:
            print(f"Found {len(errors)} data quality issues")
    """

    INTERVAL_DELTAS = {
        '1m': pd.Timedelta(minutes=1),
        '5m': pd.Timedelta(minutes=5),
        '15m': pd.Timedelta(minutes=15),
        '1h': pd.Timedelta(hours=1),
        '4h': pd.Timedelta(hours=4),
        '1d': pd.Timedelta(days=1),
    }

    def __init__(
        self,
        interval: str = '1d',
        strictness: str = 'warn',
        gap_tolerance_multiplier: float = 1.5
    ):
        self.interval = interval
        self.strictness = strictness
        self.gap_tolerance = gap_tolerance_multiplier
        self.expected_delta = self.INTERVAL_DELTAS.get(interval, pd.Timedelta(days=1))

    def validate(self, df: pd.DataFrame) -> List[ValidationError]:
        """
        Run all validation checks on DataFrame.

        Args:
            df: OHLCV DataFrame with columns: date, open, high, low, close, volume

        Returns:
            List of ValidationError objects (empty if all checks pass)
        """
        errors = []

        errors.extend(self._validate_ohlc_relationships(df))
        errors.extend(self._validate_value_ranges(df))
        errors.extend(self._validate_timestamps(df))
        errors.extend(self._validate_volume_sanity(df))

        if errors:
            self._log_errors(errors)
            if self.strictness == 'error' and any(e.severity == ValidationSeverity.ERROR for e in errors):
                critical_count = sum(1 for e in errors if e.severity == ValidationSeverity.ERROR)
                raise ValueError(f"Critical data quality errors: {critical_count} issues found")

        return errors

    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> List[ValidationError]:
        """Validate OHLC price relationships."""
        errors = []

        # Check: high >= low
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            for idx in df[invalid_hl].index:
                errors.append(ValidationError(
                    rule='high_low_violation',
                    severity=ValidationSeverity.ERROR,
                    bar_index=idx,
                    timestamp=df.loc[idx, 'date'],
                    message=f"High < Low at index {idx}: high={df.loc[idx, 'high']}, low={df.loc[idx, 'low']}",
                    details={'high': df.loc[idx, 'high'], 'low': df.loc[idx, 'low']}
                ))

        # Check: open within [low, high]
        invalid_open = (df['open'] > df['high']) | (df['open'] < df['low'])
        if invalid_open.any():
            for idx in df[invalid_open].index:
                errors.append(ValidationError(
                    rule='open_out_of_range',
                    severity=ValidationSeverity.ERROR,
                    bar_index=idx,
                    timestamp=df.loc[idx, 'date'],
                    message=f"Open outside HL range at index {idx}",
                    details={'open': df.loc[idx, 'open'], 'high': df.loc[idx, 'high'], 'low': df.loc[idx, 'low']}
                ))

        # Check: close within [low, high]
        invalid_close = (df['close'] > df['high']) | (df['close'] < df['low'])
        if invalid_close.any():
            for idx in df[invalid_close].index:
                errors.append(ValidationError(
                    rule='close_out_of_range',
                    severity=ValidationSeverity.ERROR,
                    bar_index=idx,
                    timestamp=df.loc[idx, 'date'],
                    message=f"Close outside HL range at index {idx}",
                    details={'close': df.loc[idx, 'close'], 'high': df.loc[idx, 'high'], 'low': df.loc[idx, 'low']}
                ))

        return errors

    def _validate_value_ranges(self, df: pd.DataFrame) -> List[ValidationError]:
        """Validate price and volume ranges."""
        errors = []

        # Check: no negative prices
        for col in ['open', 'high', 'low', 'close']:
            negative = df[col] < 0
            if negative.any():
                errors.append(ValidationError(
                    rule='negative_price',
                    severity=ValidationSeverity.ERROR,
                    bar_index=df[negative].index[0],
                    timestamp=df.loc[df[negative].index[0], 'date'],
                    message=f"Negative {col} prices: {negative.sum()} bars"
                ))

        # Check: no negative volumes
        negative_vol = df['volume'] < 0
        if negative_vol.any():
            errors.append(ValidationError(
                rule='negative_volume',
                severity=ValidationSeverity.ERROR,
                bar_index=df[negative_vol].index[0],
                timestamp=df.loc[df[negative_vol].index[0], 'date'],
                message=f"Negative volumes: {negative_vol.sum()} bars"
            ))

        # Check: zero closes (suspicious but not critical)
        zero_close = df['close'] == 0
        if zero_close.any():
            errors.append(ValidationError(
                rule='zero_close',
                severity=ValidationSeverity.WARNING,
                bar_index=df[zero_close].index[0],
                timestamp=df.loc[df[zero_close].index[0], 'date'],
                message=f"Zero close prices: {zero_close.sum()} bars (suspicious)"
            ))

        return errors

    def _validate_timestamps(self, df: pd.DataFrame) -> List[ValidationError]:
        """Validate timestamp continuity."""
        errors = []

        # Check: monotonically increasing
        if not df['date'].is_monotonic_increasing:
            errors.append(ValidationError(
                rule='non_monotonic_timestamps',
                severity=ValidationSeverity.ERROR,
                bar_index=0,
                message="Timestamps are not monotonically increasing"
            ))

        # Check: no duplicates
        duplicates = df['date'].duplicated()
        if duplicates.any():
            errors.append(ValidationError(
                rule='duplicate_timestamps',
                severity=ValidationSeverity.ERROR,
                bar_index=df[duplicates].index[0],
                timestamp=df.loc[df[duplicates].index[0], 'date'],
                message=f"Duplicate timestamps: {duplicates.sum()} duplicates"
            ))

        # Check: gaps
        time_diffs = df['date'].diff()
        gap_threshold = self.expected_delta * self.gap_tolerance
        gaps = time_diffs[time_diffs > gap_threshold]

        if not gaps.empty:
            for idx in gaps.index:
                gap_start = df.loc[idx - 1, 'date']
                gap_end = df.loc[idx, 'date']
                gap_duration = gap_end - gap_start

                errors.append(ValidationError(
                    rule='timestamp_gap',
                    severity=ValidationSeverity.WARNING,
                    bar_index=idx,
                    timestamp=gap_start,
                    message=f"Gap detected: {gap_start} → {gap_end} ({gap_duration})",
                    details={'gap_start': gap_start, 'gap_end': gap_end, 'duration': gap_duration}
                ))

        return errors

    def _validate_volume_sanity(self, df: pd.DataFrame) -> List[ValidationError]:
        """Validate volume sanity."""
        errors = []

        if len(df) < 10:
            return errors

        median_volume = df['volume'].median()

        # Check: volume spikes > 100x median
        volume_spikes = df['volume'] > (median_volume * 100)
        if volume_spikes.any():
            for idx in df[volume_spikes].index:
                errors.append(ValidationError(
                    rule='volume_spike',
                    severity=ValidationSeverity.WARNING,
                    bar_index=idx,
                    timestamp=df.loc[idx, 'date'],
                    message=f"Volume spike at index {idx}: {df.loc[idx, 'volume']:.2f} (median: {median_volume:.2f})",
                    details={'volume': df.loc[idx, 'volume'], 'median_volume': median_volume}
                ))

        # Check: long zero-volume streaks (>5 consecutive bars)
        zero_volume = df['volume'] == 0
        if zero_volume.any():
            streak_lengths = zero_volume.astype(int).groupby((zero_volume != zero_volume.shift()).cumsum()).sum()
            long_streaks = streak_lengths[streak_lengths > 5]

            if not long_streaks.empty:
                errors.append(ValidationError(
                    rule='zero_volume_streak',
                    severity=ValidationSeverity.WARNING,
                    bar_index=0,
                    message=f"Long zero-volume streaks: {len(long_streaks)} streaks > 5 bars"
                ))

        return errors

    def _log_errors(self, errors: List[ValidationError]) -> None:
        """Log validation errors."""
        logger.warning(f"Data validation found {len(errors)} issues:")
        for error in errors:
            if error.severity == ValidationSeverity.ERROR:
                logger.error(f"  {error}")
            else:
                logger.warning(f"  {error}")
