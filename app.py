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

app = FastAPI(
    title="RTDC File API",
    description="API for serving RTDC file metadata"
)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Middleware - should be first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
    expose_headers=["X-Content-Type-Options", "X-Frame-Options"]
)

# Other middleware follows
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1"]
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

# Configure this path to point to your .rtdc file
RTDC_FILE_PATH = Path("datasets/S8_Lung_Healthy.rtdc")

@app.get("/metadata")
@limiter.limit("5/minute")  # Limit to 5 requests per minute per IP
async def get_metadata(request: Request):
    """Return metadata from the RTDC file"""
    if not RTDC_FILE_PATH.exists():
        raise HTTPException(status_code=404, detail="RTDC file not found")
    
    try:
        with h5py.File(RTDC_FILE_PATH, 'r') as f:
            # Extract basic metadata
            metadata = {
                "filename": RTDC_FILE_PATH.name,
                "groups": list(f.keys()),
                "attributes": dict(f.attrs)
            }
            
            # Convert any numpy types to native Python types for JSON serialization
            metadata["attributes"] = {
                k: v.item() if hasattr(v, 'item') else v 
                for k, v in metadata["attributes"].items()
            }
            
            return metadata
            
    except Exception as e:
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