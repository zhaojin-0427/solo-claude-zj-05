"""本机启动入口：python3 run.py 或 uvicorn foucault.app:app"""
import os

import uvicorn

if __name__ == "__main__":
    os.environ.setdefault("FOUCAULT_DB", "foucault.db")
    uvicorn.run("foucault.app:app", host="127.0.0.1", port=8000, reload=False)
