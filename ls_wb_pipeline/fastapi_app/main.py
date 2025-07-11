from fastapi import FastAPI
from routes import router

app = FastAPI(title="LS WebDAV Pipeline API")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)