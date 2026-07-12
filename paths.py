"""
This is the file of all the paths we use.
NOTICE: we use relative paths so in the future people could use this code.
"""

from pathlib import Path

# Absolute path to the project root folder.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

TECH_UNIVERSE_CSV = DATA_DIR / "tech_universe.csv"
TOP_40_TECH_COMPANIES_NAMES = DATA_DIR / "top_40_teach_companies_names.pkl"

FIVE_THOUSAND_DAYS_DATA = DATA_DIR / "five_thousand_days_data.pkl"
FIVE_THOUSAND_DAYS_DATA_TESTING = DATA_DIR / "five_thousand_days_data_testing.pkl"
FIVE_THOUSAND_DAYS_DATA_EXPERIMENT = DATA_DIR / "five_thousand_days_data_experiment.pkl"

FIVE_THOUSAND_HOURLY_DATA = DATA_DIR / "five_thousand_hourly_data.pkl"
FIVE_THOUSAND_HOURLY_TEMP_DATA = DATA_DIR / "five_thousand_hourly_temp_data.pkl"
FIVE_THOUSAND_HOURLY_FINAL_DATA = DATA_DIR / "five_thousand_hourly_final_data.pkl"

EXPERIMENT_TRAIN_AND_VALIDATION_DATA = DATA_DIR / "experiment_train_and_validation_data.pkl"
EXPERIMENT_TEST_DATA = DATA_DIR / "experiment_test_data.pkl"

EARLIEST_TIMESTAMPS_DAILY_TOP_40 = DATA_DIR / "Earliest_Timestamps_daily_top_40_teach_companies_data.pkl"
EARLIEST_TIMESTAMPS_HOURLY_TOP_40 = DATA_DIR / "Earliest_Timestamps_hourly_top_40_teach_companies_data.pkl"
EARLIEST_TIMESTAMPS_TOP_40 = DATA_DIR / "Earliest_Timestamps_top_40_teach_companies_data.pkl"
