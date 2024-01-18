import uvicorn
from pydantic import Field, BaseModel
from fastapi import FastAPI

app = FastAPI()

# Data submitted by POST
class PredictionRequestData(BaseModel):
    # Examples taken from first few lines of data
    age: int = Field(example=39)
    workclass: str = Field(examples=["State-gov", "Self-emp-not-inc", "Private"])
    fnlgt: int = Field(example=77516)
    education: str = Field(examples=["Bachelors", "Masters", "9th", "HS-grad"])
    education_num: int = Field(example=13)
    marital_status: str = Field(examples=["Never-married", "Divorced"])
    occupation: str = Field(examples=["Adm-clerical", "Exec-managerial", "Sales"])
    relationship: str = Field(examples=["Not-in-family", "Unmarried"])
    race: str = Field(examples=["White", "Black"])
    sex: str = Field(examples=["Male", "Female"])
    capital_gain: int = Field(example=2174)
    capital_loss: int = Field(example=0)
    hours_per_week: int = Field(example=40)
    native_country: str = Field(examples=["United-States", "Cuba", "Jamaica"])


# Root route 
@app.get("/")
async def root_route():
    message = "Welcome to a data science pipeline server"
    return {"message": message}


# Inference route
@app.post("/predict")
async def inference_route(request_data: PredictionRequestData):
    print("Hitting post route")
    return {"message": request_data}


if __name__ == "__main__":
    host_name = "0.0.0.0"
    port_number = 8080
    uvicorn.run("main:app",
                host=host_name,
                port=port_number,
                log_level="info",
                reload=True)