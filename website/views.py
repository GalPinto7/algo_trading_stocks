"""
This handles the HTML pages.
We define routes like Home and everything not related to
authentication.
Blueprint helps us splitting the pages to different files.

* we defined status 200 to be -> good.
"""

# A Blueprint in Flask is a way to organize your application into smaller, modular pieces.
# Instead of putting all routes in one file, you can split them into multiple files (like views.py, auth.py, etc.).
# 'views' is the name of the blueprint.
# __name__ tells Flask where this blueprint is located.
from flask import Blueprint, render_template


# we will import and use this
views = Blueprint('views', __name__)

# todo: write real data in the websites
# default route -> home
@views.route('/')
def home():
    return render_template('home.html'),200

# This page is not mainly for prediction. It is for viewing the data.
@views.route('/dashboard')
def dashboard():
    return("<h1>This is the dashboard page</h1><p>will print stuff like"
           "list of the stocks, first and last date of the stocks prices and so on</p>")

# This is the main page
@views.route('/predict')
def predict():
    return("<h1>This is the predict page</h1><p>The user could select "
           "a model and a stock and get the predicted price of the stock</p>")

# This is the evaluation_and_simulation page
@views.route('/evaluation_and_simulation')
def evaluation_and_simulation():
    return("<h1>This is the evaluation and simulation page</h1>"
           "<p>We will run the simulation code here:"
           "expanding-window evaluation, portfolio simulation, money made/lost, and number of buys/sells.</p>")

# This is a page for me, to see what is
# the status of the server.
@views.route('/status')
def status():
    """
    Returns the status of the app.
    :return: A dictionary with the status of the app.
    """
    return {"version": "1.0", "status": "OK"}, 200