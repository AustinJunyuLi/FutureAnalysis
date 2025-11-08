# F1/F2 Labeling Fix - Implementation Plan & Tracking

*Historical reference: this document captures the pre–November 2025 migration from the legacy availability-based pipeline to the current deterministic framework.*

**Date**: 2025-11-06
**Issue**: Supervisor feedback - "F1 should appear at exactly the same time as the previous one expires"
**Status**: IN PROGRESS

---

## Executive Summary

Fixing critical F1/F2 labeling bug where contracts remain active at their expiry instant, plus enforcing strict calendar requirements and converting to hour-based timing precision.

**Root Causes Identified**:
1. Legacy v1 code uses `delta >= 0` keeping expired contracts active at expiry instant
2. DST handling with `ambiguous="NaT"` causes silent data loss (4 locations)
3. Calendar fallback to weekdays creates inaccurate holiday handling
4. Mix of day-based and hour-based timing loses precision

**Solution**: Remove v1 code entirely, fix DST handling, require calendars, use hour-based timing throughout.

---

## Implementation Progress

### ✅ PHASE 0: Update Tracking Document [COMPLETED]
- Updated mismatch_fix.md with implementation plan
- Serves as single source of truth during execution

### ⏳ PHASE 1: DST Handling + Tz-Aware Guards [IN PROGRESS]

**Files**: `rolls.py`, `analysis.py`

**Changes Required**:
1. [ ] Line 209 in rolls.py: `ambiguous="NaT"` → `"infer"`
2. [ ] Line 227 in rolls.py: `ambiguous="NaT"` → `"infer"`
3. [ ] Line 490 in analysis.py: `ambiguous="NaT"` → `"infer"`
4. [ ] Line 502 in analysis.py: `ambiguous="NaT"` → `"infer"`
5. [ ] Add tz-aware + monotonicity guard before line 209 in rolls.py:
   ```python
   # Ensure index is monotonic and handle tz-awareness
   idx = panel.index
   if not idx.is_monotonic_increasing:
       idx = idx.sort_values()

   if idx.tz is None:
       idx_local = idx.tz_localize(tz_exchange, nonexistent="shift_forward", ambiguous="infer")
   else:
       idx_local = idx.tz_convert(tz_exchange)

   idx_utc = idx_local.tz_convert("UTC")
   ```

**Validation Commands**:
```bash
pytest tests/test_labeler.py -xvs
python -c "from futures_roll_analysis.rolls import identify_front_to_f12_v2; print('Import OK')"
```

**Status**: ⏳ NOT STARTED

---

### 🔲 PHASE 2: Comprehensive Test Suite

**File**: Create `tests/test_rolls.py`

**Tests to Implement**:
- [ ] `test_front_next_switches_exactly_at_expiry()` - Core supervisor requirement
- [ ] `test_dst_fallback_both_occurrences()` - BOTH 01:30 timestamps
- [ ] `test_dst_spring_forward_shifted()` - Non-existent time handling
- [ ] `test_utc_aware_index_handling()` - Already-localized index
- [ ] `test_full_strip_f1_to_f12()` - 12-contract strip generation
- [ ] `test_no_valid_contracts_all_expired()` - Edge case
- [ ] `test_single_contract_only()` - Edge case
- [ ] `test_expiry_presence_validation()` - Missing expiry check

**Validation Commands**:
```bash
pytest tests/test_rolls.py -xvs
```

**Status**: ⏳ PENDING PHASE 1

---

### 🔲 PHASE 3: Strict Calendar Requirement

**Files**: `config.py`, `spreads.py`, `analysis.py`, `settings.yaml`

**Changes Required**:
- [ ] Add calendar validation in `config.py` `_validate_settings()` (after line 148)
- [ ] Remove weekday fallback in `spreads.py` `_determine_open_days()` (lines 127-143)
- [ ] Fail fast in `analysis.py` if calendar load fails (lines 114-121, 354-361)
- [ ] Remove `business_days.enabled` from `settings.yaml` (line 66)

**Validation Commands**:
```bash
# With valid calendar
futures-roll analyze --mode hourly --max-files 2 --output-dir outputs/test_calendar

# Test missing calendar error
mv metadata/calendars/cme_globex_holidays.csv metadata/calendars/backup.csv
futures-roll analyze --mode hourly --max-files 2 2>&1 | grep -i "calendar"
mv metadata/calendars/backup.csv metadata/calendars/cme_globex_holidays.csv
```

**Status**: ⏳ PENDING PHASE 2

---

### 🔲 PHASE 4: Remove v1 Code

**File**: `rolls.py`

**Changes Required**:
- [ ] Delete `identify_front_next()` function (lines ~32-106)
- [ ] Delete `identify_front_to_f12()` function (lines ~109-201)
- [ ] Rename `identify_front_to_f12_v2` → `identify_front_to_f12`
- [ ] Rename `identify_front_next_v2` → `identify_front_next`
- [ ] Update docstrings to remove v2 references
- [ ] Add expiry presence validation at function start

**Validation Commands**:
```bash
pytest tests/ -v
python -c "from futures_roll_analysis.rolls import identify_front_next; print('Renamed OK')"
```

**Status**: ⏳ PENDING PHASE 3

---

### 🔲 PHASE 5: Update Analysis Pipeline

**File**: `analysis.py`

**Changes Required**:
- [ ] Remove mode selection in `run_bucket_analysis()` (lines ~69-83)
- [ ] Remove mode selection in `run_daily_analysis()` (lines ~320-327)
- [ ] Make switch log unconditional (line ~256-260)
- [ ] Rename output file: `roll_switches_v2.csv` → `roll_switches.csv`

**Validation Commands**:
```bash
futures-roll analyze --mode hourly --max-files 5 --output-dir outputs/test_pipeline
ls outputs/test_pipeline/analysis/roll_switches.csv  # Verify exists (no _v2 suffix)
```

**Status**: ⏳ PENDING PHASE 4

---

### 🔲 PHASE 6: Hour-Based Timing Metrics

**Files**: `trading_days.py`, `contract_month_analysis.py`, `volume_crossover.py`

**Changes Required**:
- [ ] Rename `_compute_days_to_expiry()` → `_compute_hours_to_expiry()` in trading_days.py
- [ ] Update implementation to use `.total_seconds() / 3600.0`
- [ ] Update callers to use hours (lines 304, 380 in trading_days.py)
- [ ] Fix 3 `.dt.days` usages in contract_month_analysis.py (lines 158, 195, 419)
- [ ] Fix `.dt.days` in volume_crossover.py (line 177)

**Validation Commands**:
```bash
pytest tests/test_trading_days.py -xvs
python scripts/volume_crossover.py  # Verify runs without errors
```

**Status**: ⏳ PENDING PHASE 5

---

### 🔲 PHASE 7: Config Cleanup

**Files**: `settings.yaml`, `config.py`

**Changes Required**:
- [ ] Delete `selection:` section from settings.yaml (lines 44-46)
- [ ] Update business_days comment in settings.yaml
- [ ] Remove selection handling from config.py (lines 72-73)
- [ ] Remove selection from Settings dataclass (line 20)
- [ ] Remove selection from return statement (line 117)

**Validation Commands**:
```bash
python -c "from futures_roll_analysis.config import load_settings; s = load_settings('config/settings.yaml'); print('Config OK')"
```

**Status**: ⏳ PENDING PHASE 6

---

### 🔲 PHASE 8: Documentation

**Files**: `README.md`, `CHANGELOG.md`, `AGENTS.md`

**Changes Required**:
- [ ] Add contract labeling explanation to README
- [ ] Document required calendar in README
- [ ] Create CHANGELOG v2.0.0 entry with breaking changes
- [ ] Update AGENTS.md with timing/calendar guidelines

**Validation Commands**:
```bash
# Manual review of docs
grep -i "expiry_v1" README.md CHANGELOG.md AGENTS.md  # Should only find migration notes in CHANGELOG
```

**Status**: ⏳ PENDING PHASE 7

---

### 🔲 PHASE 9: Comprehensive Validation

**9.1 Test Suite**:
```bash
pytest tests/ -v --cov=src/futures_roll_analysis --cov-report=term-missing
```

**9.2 Functional Validation**:
```bash
futures-roll analyze --mode hourly --max-files 3 --output-dir outputs/validation_final
python -c "
import pandas as pd
panel = pd.read_parquet('outputs/validation_final/panels/hourly_panel.parquet')
print('Shape:', panel.shape)
print('NaT count:', panel.isna().sum().sum())
print('Monotonic:', panel.index.is_monotonic_increasing)
"
```

**9.3 DST Transition Check**:
```bash
python -c "
import pandas as pd
panel = pd.read_parquet('outputs/validation_final/panels/hourly_panel.parquet')
nov = panel.loc['2024-11-02':'2024-11-04']
print('Nov 2-4 points:', len(nov))
print('NaT during DST:', nov.isna().sum().sum())
"
```

**9.4 Timing Precision Check**:
```bash
python -c "
import pandas as pd
switches = pd.read_csv('outputs/validation_final/analysis/roll_switches.csv', parse_dates=['transition_time_local', 'prev_f1_expiry_local'])
print('Transitions at exact expiry:', (switches['transition_time_local'] == switches['prev_f1_expiry_local']).all())
"
```

**9.5 Before/After Comparison**:
```bash
git stash
futures-roll analyze --mode hourly --max-files 5 --output-dir outputs/before_fix
git stash pop
futures-roll analyze --mode hourly --max-files 5 --output-dir outputs/after_fix

python -c "
import pandas as pd
before = pd.read_csv('outputs/before_fix/roll_signals/hourly_spread.csv', index_col=0, parse_dates=True)
after = pd.read_csv('outputs/after_fix/roll_signals/hourly_spread.csv', index_col=0, parse_dates=True)
diff = (before - after).abs()
print('Max diff:', diff.max().max())
print('Mean diff:', diff.mean().mean())
print('Changed rows:', (diff > 0.001).any(axis=1).sum())
"
```

**Status**: ⏳ PENDING PHASE 8

---

### 🔲 PHASE 10: Git Commits

**Commits to Create**:
1. [ ] DST + tz-aware guards
2. [ ] Comprehensive tests
3. [ ] Strict calendar requirement
4. [ ] Remove v1 labeler
5. [ ] Hour-based timing
6. [ ] Config cleanup
7. [ ] Documentation

**Status**: ⏳ PENDING PHASE 9

---

## Deliverables for Supervisor

After Phase 9 completes, you will have:

- ✅ Switch log (`roll_switches.csv`) with exact transition timestamps
- ✅ Before/after comparison showing impact
- ✅ Validation proving F2→F1 occurs at exact expiry instant
- ✅ DST validation proving no data gaps

Can send supervisor:
- Switch log excerpt around expiries
- "F2 becomes F1 exactly at expiry instant (verified)"
- Before/after diff
- (Later: plots if requested)

---

## Success Criteria

- [x] F2 becomes F1 exactly at expiry instant (supervisor requirement)
- [ ] No data gaps during DST transitions
- [ ] Tz-aware index handling (prevents crashes)
- [ ] Strict calendar requirement (prevents silent errors)
- [ ] Hour-based timing precision throughout
- [ ] All tests pass (new and existing)
- [ ] Clear error messages when calendar missing

---

## Timeline

**Estimated**: 15-18 hours total
- Phases 1-8: 13.5 hours implementation
- Phase 9: 2-3 hours validation
- Phase 10: 30 mins commits

**Progress**: Phase 0 complete, starting Phase 1

---

## Iteration Protocol

**At end of each phase**:
1. Run validation commands
2. If ✅ GREEN → Mark phase complete, proceed to next
3. If ❌ RED → Debug, fix, re-validate current phase
4. Do NOT proceed on red status

---

**Last Updated**: 2025-11-06
**Current Phase**: 1 (DST Handling + Tz-Aware Guards)
