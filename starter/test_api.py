import requests as r

ENDPOINT = "https://ml-deployment-pipeline.herokuapp.com/"


def test_get_req():
    response = r.get(ENDPOINT)
    assert response.status_code == 200


def test_less_than_50k():

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

    response = r.post(ENDPOINT + "infer/", json=payload)

    assert response.status_code == 200 and response.text == '"<=50K"'


def test_more_than_50k():


    payload = {
        "age": 52,
        "fnlgt": 209642,
        "education-num": 9,
        "capital-gain": 123387,
        "capital-loss": 0,
        "hours-per-week": 40,
        "workclass": "Self-emp-not-inc",
        "education": "Bachelors",
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native-country": "United-States"
    }

    response = r.post(ENDPOINT + "infer/", json=payload)

    assert response.status_code == 200 and response.text == '\">50K\"'
