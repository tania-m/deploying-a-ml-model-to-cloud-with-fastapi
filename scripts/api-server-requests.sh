# GET request
curl http://localhost:8080/

# POST request with test data that is from last line of census_clean.csv
curl -d @tests/test-request.json -H "Content-Type: application/json" -H "Accept: application/json" http://localhost:8000/predict