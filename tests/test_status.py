"""
This file is about testing the connections of the pages
of the website.
“we will act as clients testing our app, and we will assert that we get the responses we expect”.

We will use "assert" to check if the response is correct.

All the responses are defined in the view.py file.
"""

# todo: if we change the views.py, we need to change this as well.
# todo: add more tests.

import pytest
from website import create_app


"""
for CI, I can't use something like: http://localhost:5000/status
That only works if you manually start the Flask server before running tests.
That is bad for CI because GitHub Actions will not automatically have your server running.
So the test should use Flask’s built-in test client.
A test client is a fake browser/request tool built into Flask.
"""

@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()

def test_status_page(client):
    """
    Checks if the status page is working correctly.
    We check if: status code is 200 -> good.
                 version = 1.0.
                 status = ok
    :return:
    """
    # return {"version": "1.0", "status": "OK"}, 200
    response = client.get("/status")
    # check if the status_code - 200
    assert response.status_code == 200
    data = response.get_json()
    assert data["version"] == "1.0"
    assert data["status"] == "OK"








