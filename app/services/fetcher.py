import os
import requests
import logging
import traceback
import app.services.preprocess as preprocess

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Pipe-delimited log format: TIME|LEVEL|MODULE|MESSAGE
formatter = logging.Formatter(
    fmt='%(asctime)s|%(levelname)-8s|%(module)-15s|%(funcName)-20s|%(lineno)4d|%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Stream handler (console)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Avoid adding multiple handlers if reloaded
if not logger.hasHandlers():
    logger.addHandler(console_handler)

BALL_BY_BALL_DATA = "https://raw.githubusercontent.com/ritesh-ojha/IPL-DATASET/refs/heads/main/csv/Ball_By_Ball_Match_Data.csv"
MATCH_INFO_DATA = "https://raw.githubusercontent.com/ritesh-ojha/IPL-DATASET/refs/heads/main/csv/Match_Info.csv"
PLAYER_INFO_DATA = "https://raw.githubusercontent.com/ritesh-ojha/IPL-DATASET/refs/heads/main/csv/2024_players_details.csv"
DATA_DIR = "data/ipl"


def fetch_ball_by_ball_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info("Starting Ball By Ball Data CSV download from GitHub")
    filename = "ball_by_ball_data.csv"
    local_path = os.path.join(DATA_DIR, filename)

    try:
        response = requests.get(BALL_BY_BALL_DATA, timeout=10)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
            logger.debug(f"Downloaded {filename}")
    except Exception as e:
        logger.error(f"Failed to fetch {filename}|{e}")

    logger.info(f"Finished processing the fetch request")
    return {"status": "done"}


def fetch_match_info():
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info("Starting Match Info CSV download from GitHub")
    filename = "match_info.csv"
    local_path = os.path.join(DATA_DIR, filename)

    try:
        response = requests.get(MATCH_INFO_DATA, timeout=10)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
            logger.debug(f"Downloaded {filename}")
    except Exception as e:
        logger.error(f"Failed to fetch {filename}|{e}")

    logger.info(f"Finished processing the fetch request")
    return {"status": "done"}


def fetch_player_info():
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info("Starting Player Info CSV download from GitHub")
    filename = "player_info.csv"
    local_path = os.path.join(DATA_DIR, filename)

    try:
        response = requests.get(PLAYER_INFO_DATA, timeout=10)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
            logger.debug(f"Downloaded {filename}")
    except Exception as e:
        logger.error(f"Failed to fetch {filename}|{e}")

    logger.info(f"Finished processing the fetch request")
    return {"status": "done"}


def get_player_stats():
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info("Updating Player Stats")
    filename = "player_stats.csv"
    local_path = os.path.join(DATA_DIR, filename)

    try:
        preprocess.aggregate_stats()

    except Exception as e:
        logger.error(f"Failed to update player stats | {e}\n{traceback.format_exc()}")

    logger.info(f"Finished updating the player stats file")
    return {"status": "done"}
