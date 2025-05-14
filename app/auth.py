from jose import JWTError, jwt
from fastapi import HTTPException
from typing import Optional
from datetime import datetime, timedelta


PUBLIC_KEY_PATH = "public.pem"


def get_public_key():
    with open(PUBLIC_KEY_PATH, "r") as f:
        return f.read()


def verify_token(token: str):
    try:
        public_key = get_public_key()
        payload = jwt.decode(token, public_key, algorithms=["RS256"])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
