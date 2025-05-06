import pandas as pd
from flask import Blueprint, jsonify, request, render_template

from .services import datastore
from .services.fetcher import fetch_ball_by_ball_data, fetch_match_info, fetch_player_info, get_player_stats
from .services.model.cat_boost import run_catboost_training, run_catboost_testing

main = Blueprint("main", __name__)


@main.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# @main.route("/suggest", methods=["POST"])
# def suggest():
#     input_data = request.json
#     result = suggest_bowler(input_data)
#     return jsonify(result)

@main.route("/load_data_frames", methods=["GET"])
def refresh_data(result=None):
    datastore.bbb_df = pd.read_csv("data/ipl/ball_by_ball_data.csv")
    datastore.mi_df = pd.read_csv("data/ipl/match_info.csv")
    datastore.pi_df = pd.read_csv("data/ipl/player_info.csv")
    result["ball_by_ball_rows"] = len(datastore.bbb_df)
    result["match_info_rows"] = len(datastore.mi_df)
    result["player_info_rows"] = len(datastore.pi_df)
    return jsonify(result)


@main.route("/fetch_ball_by_ball_data", methods=["GET"])
def fetch_b_by_b_data():
    result = fetch_ball_by_ball_data()
    datastore.bbb_df = pd.read_csv("data/ipl/ball_by_ball_data.csv")
    result["downloaded_rows"] = len(datastore.bbb_df)
    return jsonify(result)


@main.route("/fetch_match_info", methods=["GET"])
def fetch_mi_data():
    result = fetch_match_info()
    datastore.mi_df = pd.read_csv("data/ipl/match_info.csv")
    result["downloaded_rows"] = len(datastore.mi_df)
    return jsonify(result)


@main.route("/fetch_player_info", methods=["GET"])
def fetch_pi_data():
    result = fetch_player_info()
    datastore.pi_df = pd.read_csv("data/ipl/player_info.csv")
    result["downloaded_rows"] = len(datastore.pi_df)
    return jsonify(result)


@main.route("/update_player_stats", methods=["GET"])
def update_player_stats():
    result = get_player_stats()
    result["player_stats_rows"] = len(datastore.performance_df)
    return jsonify(result)


# @main.route("/clean_model_data", methods=["GET"])
# def model_data_cleanup():
#     result = clean_model_data()
#     return jsonify(result)


@main.route("/run_catboost_training", methods=["GET"])
def trigger_catboost_classifier_training():
    result = run_catboost_training()
    return jsonify(result)


@main.route("/run_catboost_testing", methods=["POST"])
def trigger_catboost_classifier_testing():
    input_data = request.json
    result = run_catboost_testing(input_data['test_file'])
    return jsonify(result)
