"""
Here we will create an API for the project,
so other tools or future UI components can consume.
The API calls return data in JSON format.
"""

# todo: every endpoint you add, make sure you add a test to it

from flask import Blueprint, jsonify
from .services.data_service import get_dataset_inventory, get_model_catalog

# status end point: checks if the backend is alive
# important for CI and more to check.
# A Blueprint in Flask is a way to organize routes into separate files
# all routes start with /api
api = Blueprint("api", __name__, url_prefix="/api")

# Get endpoint at /api/status
@api.get("/status")
# the function that runs when /api/status is called
def status():
    """
    Return a simple health response for the API.

    This endpoint helps developers, tests, CI, and future monitoring tools check
    that the Flask app is running and the API blueprint is registered.
    """
    return jsonify({
        "service": "stock-return-prediction",
        "status": "ok",
        "version": "1.0",
    })


# We want to enable the user to see in json what models are there
@api.get("/models")
def models():
    """

    :return: The models we defined
    looks something like this:
    data = {
    "models": [
        {"flag": 0, "name": "Dumb baseline"},
        {"flag": 1, "name": "XGBoost"},
    ]
}
    """
    list_of_models = get_model_catalog()
    return jsonify({"models": list_of_models})


# see the data about the files we have
@api.get("/datasets")
def datasets():
    """

    :return:
    """
    return get_dataset_inventory()

# @api.get("/status")
# # the function that runs when /api/status is called
# def status():
#     """
#     Return a simple health response for the API.
#
#     This endpoint helps developers, tests, CI, and future monitoring tools check
#     that the Flask app is running and the API blueprint is registered.
#     """
#     return jsonify({
#         "service": "stock-return-prediction",
#         "status": "ok",
#         "version": "1.0",
#     })


