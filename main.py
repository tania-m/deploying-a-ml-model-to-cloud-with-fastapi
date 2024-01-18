import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    message = "Welcome to a data science pipeline server"
    return {"message": message}


if __name__ == "__main__":
    host_name = "0.0.0.0"
    port_number = 8080
    uvicorn.run("main:app",
                host=host_name,
                port=port_number,
                log_level="info",
                reload=True)