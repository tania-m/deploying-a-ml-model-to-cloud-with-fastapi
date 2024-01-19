# Tests for functions from main.py (API)

import pytest
from fastapi.testclient import TestClient
# TestClient needs httpx module installed to work
import json

from main import app


@pytest.fixture(scope="module")
def api_client():
    client = TestClient(app)
    return client


def test_get_welcome_root_route(api_client):
    """
    Tests root route page
    """
    
    route_under_test = "/"
    request_result = api_client.get(route_under_test)
    
    assert request_result.status_code == 200, "GET request to root route failed"
    assert request_result.json() is not None 
    # We're not testing exact test response here, to be flexible to change it 


def test_post_inference_larger_50K(api_client):
    """
    Tests POST can run inference
    """
    # Simulate curl -d @tests/test-request.json -H "Content-Type: application/json" -H "Accept: application/json" http://localhost:8000/predict

    test_data_source_file = "tests/test-request.json"
    with open(test_data_source_file) as json_file:
        data_from_json = json.load(json_file)
    
    route_under_test = "/predict"
    request_result = api_client.post(route_under_test, json=data_from_json)

    assert request_result.status_code == 200, "POST request for inference failed"
    assert request_result.json() is not None, "Inference POST endpoint failed to respond in JSON format" 
    assert request_result.json()["predictions"] == ">50K", "Inference POST endpoint failed to make correct prediction"


def test_post_inference_smaller_than_or_equal_50K(api_client):
    """
    Tests POST can run inference
    """
    # Simulate curl -d @tests/test-request-2.json -H "Content-Type: application/json" -H "Accept: application/json" http://localhost:8000/predict

    test_data_source_file = "tests/test-request-2.json"
    with open(test_data_source_file) as json_file:
        data_from_json = json.load(json_file)
    
    route_under_test = "/predict"
    request_result = api_client.post(route_under_test, json=data_from_json)

    assert request_result.status_code == 200, "POST request for inference failed"
    assert request_result.json() is not None, "Inference POST endpoint failed to respond in JSON format" 
    assert request_result.json()["predictions"] == "<=50K", "Inference POST endpoint failed to make correct prediction"