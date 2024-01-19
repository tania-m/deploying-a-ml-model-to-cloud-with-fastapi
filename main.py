import os
import uvicorn
from pydantic import Field, BaseModel
from fastapi import FastAPI
import sklearn
import pickle
import pandas as pd
from data import process_data
from model import inference


app = FastAPI()


# Schema for data submitted by POST
class PredictionRequestData(BaseModel):
    # Examples taken from first few lines of data
    # TODO: Define aliases
    age: int = Field(example=39)
    workclass: str = Field(
        examples=[
            "State-gov",
            "Self-emp-not-inc",
            "Private"])
    fnlgt: int = Field(example=77516)
    education: str = Field(examples=["Bachelors", "Masters", "9th", "HS-grad"])
    education_num: int = Field(example=13)
    marital_status: str = Field(examples=["Never-married", "Divorced"])
    occupation: str = Field(
        examples=[
            "Adm-clerical",
            "Exec-managerial",
            "Sales"])
    relationship: str = Field(examples=["Not-in-family", "Unmarried"])
    race: str = Field(examples=["White", "Black"])
    sex: str = Field(examples=["Male", "Female"])
    capital_gain: int = Field(example=2174)
    capital_loss: int = Field(example=0)
    hours_per_week: int = Field(example=40)
    native_country: str = Field(examples=["United-States", "Cuba", "Jamaica"])


def parse_item(request_data):
    """ Parse request data items

        Inputs
        ------
        request_data: input request data of type PredictionRequestData

        Returns
        -------
        Parsed request data item
    """

    parsed_request_data = {
        "age": request_data.age,
        "workclass": request_data.workclass,
        "fnlgt": request_data.fnlgt,
        "education": request_data.education,
        "education-num": request_data.education_num,
        "marital-status": request_data.marital_status,
        "occupation": request_data.occupation,
        "relationship": request_data.relationship,
        "race": request_data.race,
        "sex": request_data.sex,
        "capital-gain": request_data.capital_gain,
        "capital-loss": request_data.capital_loss,
        "hours-per-week": request_data.hours_per_week,
        "native-country": request_data.native_country
    }

    return parsed_request_data


def prepare_inference_df(request_data):
    """ Prepare data to be used for inference

        Inputs
        ------
        request_data: input request data of type PredictionRequestData

        Returns
        -------
        Prepare data for inference
    """

    inference_data_item = parse_item(request_data)
    input_data = pd.DataFrame(inference_data_item, index=[0])
    processed_X, y, encoder_from_processing, lb_from_processing = process_data(
        input_data,
        categorical_features,  # set at server startup
        x_label,  # set at server startup
        is_training,  # set at server startup
        encoder,  # loaded at server startup
        lb  # loaded at server startup
    )

    return processed_X


# Load model and supporting files
file_mode = "rb"
model_folder_path = "model"

print("Starting API server: Loading files")

if os.environ.get("RENDER_SERVICE_ID") is not None:
    service_id = os.environ["RENDER_SERVICE_ID"]
    print(f"Service ID: {service_id}")
print('The scikit-learn version is {}.'.format(sklearn.__version__))

model_name = "model.pkl"
model_full_path = os.path.join(model_folder_path, model_name)
model = pickle.load(open(model_full_path, file_mode))
print(f"Starting API server: Model loaded from {model_full_path}")

encoder_name = "encoder.pkl"
encoder_full_path = os.path.join(model_folder_path, encoder_name)
encoder = pickle.load(open(encoder_full_path, file_mode))
print(f"Starting API server: Encoder loaded from {encoder_full_path}")

label_binarizer_name = "lb.pkl"
label_binarizer_full_path = os.path.join(
    model_folder_path, label_binarizer_name)
lb = pickle.load(open(label_binarizer_full_path, file_mode))
print(
    f"Starting API server: Label binarizer loaded from {label_binarizer_full_path}")


# Setup known configs
is_training = False
x_label = None

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
print(f"Supported (known) categorical features: {categorical_features}")


# Routes #################################################################
# Root route
@app.get("/")
async def root_route():
    message = "Welcome to a data science pipeline server"
    return {"message": message}


# Inference route
@app.post("/predict")
async def inference_route(request_data: PredictionRequestData):
    processed_X = prepare_inference_df(request_data)

    inference_result = inference(model, processed_X)

    # parse prediction to return a nicer result
    # use label binarizer inverse operation to do so, since we have it
    inference_result_values = lb.inverse_transform(inference_result)
    parsed_prediction = inference_result_values[0]

    result = {"predictions": str(parsed_prediction)}
    return result


if __name__ == "__main__":
    host_name = "0.0.0.0"
    port_number = 8000
    log_level = "info"
    is_debug = True
    uvicorn.run("main:app",
                host=host_name,
                port=port_number,
                log_level=log_level,
                reload=is_debug)
