import requests
import json


def make_request(target_url, data_from_json):
    """ Make a POST request to a target URL with 
    JSON data.
    
    Inputs
        ------
        target_url: target HTTP(S) URL
        data_from_json: request data

        Returns
        -------
        status_code: request response status code
        request_results: request (predictions) results
    """
    
    request_results = requests.post(target_url, json=data_from_json)
    status_code = request_results.status_code
    prediction_result = request_results.json()["predictions"]
    print(f"Request status code: {status_code}")
    print(f"Prediction result: {prediction_result}")
    
    return status_code, request_results


if __name__ == "__main__":
    # For local testing
    target_url = "http://localhost:8000/predict"
    
    test_data_source_file = "tests/test-request-2.json"
    with open(test_data_source_file) as json_file:
        data_from_json = json.load(json_file)
        
    make_request(target_url, data_from_json)