from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBasic
from fastapi.responses import JSONResponse
import h5py
import json
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import numpy as np
import logging

# Add logging configuration at the top after imports
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create limiter before FastAPI app
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="RTDC File API",
    description="API for serving RTDC file metadata"
)

# Store the path in app state instead of global variable
app.state.RTDC_FILE_PATH = Path("datasets/S8_Lung_Healthy.rtdc")

# Set up rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add middleware in correct order
# 1. CORS middleware first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
    expose_headers=["X-Content-Type-Options", "X-Frame-Options"]
)

# 2. Rate limiting middleware
app.add_middleware(SlowAPIMiddleware)

# 3. Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*"]  # Added "*" for testing
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    # Prevent XSS attacks
    response.headers["X-XSS-Protection"] = "1; mode=block"
    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    # Enable strict HTTPS (uncomment if using HTTPS)
    # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

@app.get("/metadata")
@limiter.limit("5/minute")
async def get_metadata(request: Request):
    """Return metadata from the RTDC file"""
    file_path = app.state.RTDC_FILE_PATH
    logger.debug(f"Checking file: {file_path} (exists: {file_path.exists()})")
    
    if not file_path.exists():
        logger.debug(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="RTDC file not found")
    
    try:
        with h5py.File(file_path, 'r') as f:
            logger.debug(f"File attributes: {dict(f.attrs)}")
            
            metadata = {
                "filename": file_path.name,
                "groups": list(f.keys()),
                "attributes": {}
            }
            
            # Convert HDF5 attributes to Python types
            for key, value in f.attrs.items():
                logger.debug(f"Processing attribute {key}: {type(value)} = {value}")
                if isinstance(value, (np.ndarray, np.generic)):
                    if value.dtype.kind in ['S', 'U']:
                        metadata["attributes"][key] = value.astype(str).item()
                    else:
                        metadata["attributes"][key] = value.item()
                else:
                    metadata["attributes"][key] = value
            
            logger.debug(f"Final metadata: {metadata}")
            return metadata
            
    except Exception as e:
        logger.exception("Error reading RTDC file")
        raise HTTPException(
            status_code=500, 
            detail=f"Error reading RTDC file: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint returning API info"""
    return {
        "message": "RTDC File API",
        "endpoints": [
            "/metadata - Get RTDC file metadata",
            "/ - This help message"
        ]
    } 