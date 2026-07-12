"""
set up the application. Again, the init.py file will run
automatically when we import anything from the website package.
"""

# todo: fix this 1
from flask import Flask
import os
# Imports the existing HTML page routes and API routes.
from .views import views
from .api_routes import api



def create_app():
    """
    This function creates a flask website.
    :return: This returns the configured Flask app object, so it can be used elsewhere
    """
    app = Flask(__name__)
    # Secret key to secure sessions and cookies.
    # It can be any random string, but it must be kept secret.
    # so if SECRET_KEY is def in our env file, use it here
    app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "dev-only-secret")

    from .views import views
    from .api_routes import api

    # Register the blueprints with the Flask app
    # You can specify a URL prefix for each blueprint if you want.
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(api)

    return app
