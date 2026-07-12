"""
The file we run in order to run the app.
What do we want the website to show:
1: landing page
"""

from website import create_app
import os

# creating the app
app = create_app()

# we moved the variables to the env file,
if __name__ == "__main__":
    # Debug mode: Every time we make changes to the python code,
    # the server will automatically restart.
    # todo: (do not use in production) -> fix later
    # we use the vars in the env file if they are def,
    # if not, we use the fall back values
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))

    app.run(debug=debug, host=host, port=port)