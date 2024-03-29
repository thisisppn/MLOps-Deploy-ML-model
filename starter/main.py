# Put the code for your API here.
import pickle
import os

import uvicorn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference

app = FastAPI()


@app.get("/")
def read_root():
    return {"welcome-message": "Hello world! welcome to the API. "
                               "Please visit https://ml-deployment-pipeline.herokuapp.com/docs to get starter."}


class Item(BaseModel):
    age: int
    fnlgt: int
    education_num: int = Field(alias='education-num')
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')

    workclass: str
    education: str
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {
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
        }


@app.post("/infer/")
async def infer(item: Item):

    # separate the categorical and normal values
    item_dict = item.dict(by_alias=True)

    X = pd.DataFrame.from_records([item_dict])

    encoder = pickle.load(open("starter/model/encoder.pkl", 'rb'))
    model = pickle.load(open("starter/model/naive_model.pkl", 'rb'))

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_test, y_test, _, _ = process_data(
        X, categorical_features=cat_features, training=False, encoder=encoder,
    )

    pred = inference(model, X_test)

    labels = ["<=50K", ">50K"]

    return labels[pred[0]]


# if "DYNO" in os.environ and os.path.isdir(".dvc"):
if "DYNO" in os.environ:
    try:
        os.system("dvc config core.no_scm true")
        # os.system("dvc config core.hardlink_lock true")

        os.system("dvc pull")
        # os.system("rm -r .dvc .apt/usr/lib/dvc")
    except Exception as e:
        print("Unexpected exception {}".format(e))
    # if os.system("dvc pull") != 0:
    #     exit("dvc pull failed")
    # os.system("rm -r .dvc .apt/usr/lib/dvc")

#
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)
