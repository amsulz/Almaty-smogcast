"""Shared settings for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

# Almaty city center
LATITUDE = 43.24
LONGITUDE = 76.92
TIMEZONE = "Asia/Almaty"
# Kazakhstan moved from UTC+6 to UTC+5 on 2024-03-01. Open-Meteo returns the whole history
# with one offset (+5), so all my "local time" is UTC+5, also for the old data.
UTC_OFFSET_HOURS = 5

# OpenAQ location: US Embassy Almaty (reference-grade PM2.5, data via AirNow, since 2020-09)
OPENAQ_LOCATION_ID = 8876
EMBASSY_LAT, EMBASSY_LON = 43.253, 76.931
# the embassy station went silent on 2025-11-14. After that I use AirGradient sensors
EMBASSY_END = "2025-11-15"

# AirGradient: cheap sensors, also on OpenAQ. Most of them appeared in Almaty in Dec 2025.
# I take the ones close to the embassy, so the data is from the same part of the city
AIRGRADIENT_RADIUS_KM = 1.6
# the only AirGradient sensor that worked at the same time as the embassy (Nov 2024 - Feb 2026).
# I use it to see how much AirGradient over-reads compared to the embassy
AIRGRADIENT_BRIDGE_LOCATION = 3243331

START_DATE = "2020-09-01"

# time split, no shuffling.
# the US Embassy station stopped on 2025-11-14, so the test is 2025 and not 2026
TRAIN_END = "2024-01-01"  # train: Sep 2020 - 2023
VAL_END = "2025-01-01"    # validation: 2024, test: 2025 (until EMBASSY_END)
# after EMBASSY_END there is one more check on AirGradient data ("airgradient" split)

# heating season in Almaty, roughly
HEATING_START = (10, 15)  # 15 Oct
HEATING_END = (4, 15)     # 15 Apr

# CHP-2 coal -> gas. Planned for end of 2026, put the real date here when it happens.
# Everything before this date counts as the "coal" period.
CHP2_SWITCH_DATE = "2026-11-01"
