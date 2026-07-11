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
import requests

SERVER_HOST = "localhost"
PORT = 5000


def base_url():
    """
    This returns the base of the url of the website
    We will use it to get the start of the website url and then write the rest.
    :return: Base of the url of the website
    """
    return f"http://{SERVER_HOST}:{PORT}"

def test_status_page():
    """
    Checks if the status page is working correctly.
    We check if: status code is 200 -> good.
                 version = 1.0.
                 status = ok
    :return:
    """
    # return {"version": "1.0", "status": "OK"}, 200
    response = requests.get(f"{base_url()}/status", timeout=10)
    # check if the status_code - 200
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "1.0"
    assert data["status"] == "OK"






