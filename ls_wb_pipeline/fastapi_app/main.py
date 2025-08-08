from fastapi import FastAPI
from ls_wb_pipeline.fastapi_app.routes import router
import uvicorn

app = FastAPI(title="LS WebDAV Pipeline API")

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
