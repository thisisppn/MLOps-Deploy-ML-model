import requests as r

ENDPOINT = "https://ml-deployment-pipeline.herokuapp.com/"

payload = {
      "age": 39,
      "fnlgt": 77516,
      "education-num": 13,
      "capital-gain": 2174,
      "capital-loss": 0,
      "hours-per-week": 40,
      "workclass": "State-gov",
      "education": "Bachelors",
      "marital-status": "Never-married",
      "occupation": "Adm-clerical",
      "relationship": "Not-in-family",
      "race": "White",
      "sex": "Male",
      "native-country": "United-States"
    }

print("Sending POST requests to API: {}".format(ENDPOINT + "infer/"))

response = r.post(ENDPOINT + "infer/", json=payload)

print("Response status code: {}".format(response.status_code))
print("Response body: {}".format(response.text))
