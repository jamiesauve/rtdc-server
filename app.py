from fastapi import FastAPI, HTTPException, Request, Query
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
import hdf5plugin  # This registers the plugins automatically

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
@limiter.limit("50/minute")
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

@app.get("/data")
@limiter.limit("50/minute")
async def get_data(
    request: Request,
    offset: int = Query(default=None, ge=0, description="Starting index"),
    limit: int = Query(default=1000, ge=1, le=5000, description="Number of events to return")
):
    """Return data points from the RTDC file with pagination"""
    # Parse range header if present (e.g. "events=0-499")
    range_header = request.headers.get("Range")
    if range_header and offset is None:  # Only use range header if offset not specified in query
        try:
            range_type, range_value = range_header.split("=")
            if range_type != "events":
                raise ValueError("Invalid range type")
            
            try:
                start, end = map(int, range_value.split("-"))
            except ValueError:
                raise ValueError("Invalid range values")
                
            if end < start:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid range: end must be greater than start"
                )
                
            offset = start
            limit = end - start + 1  # +1 because range is inclusive
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid range header format. Expected: events=start-end. Error: {str(e)}"
            )

    # Use default offset if neither query param nor range header specified
    if offset is None:
        offset = 0

    file_path = app.state.RTDC_FILE_PATH
    logger.debug(f"Reading data from: {file_path} (offset={offset}, limit={limit})")
    
    if not file_path.exists():
        logger.debug(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="RTDC file not found")
    
    try:
        with h5py.File(file_path, 'r') as f:
            if 'events' not in f:
                logger.debug("No events group found")
                raise HTTPException(
                    status_code=500,
                    detail="Invalid RTDC file: no events group found"
                )
            
            events = f['events']
            safe_features = []
            
            # Test read each feature to ensure we can access it
            for feature in events.keys():
                if isinstance(events[feature], h5py.Dataset):
                    try:
                        # Try reading first value to verify accessibility
                        _ = events[feature][0]
                        if events[feature].dtype.kind in ['i', 'u', 'f', 'S', 'U']:
                            if feature not in ['image', 'mask', 'trace']:
                                safe_features.append(feature)
                    except OSError:
                        logger.warning(f"Skipping feature {feature} due to plugin requirement")
                        continue
            
            logger.debug(f"Safe features to read: {safe_features}")
            
            if not safe_features:
                raise HTTPException(
                    status_code=500,
                    detail="No readable features found in RTDC file"
                )
            
            # Get the total number of events before any other operations
            total_events = len(events[safe_features[0]])
            if offset >= total_events:
                logger.debug(f"Offset {offset} exceeds total events {total_events}")
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"Offset {offset} exceeds total events {total_events}"}
                )
            
            # Calculate actual limit (don't read past end of file)
            actual_limit = min(limit, total_events - offset)
            
            # Collect events data for the requested range
            data_points = []
            # Read each feature's data in one slice operation
            feature_data = {
                feature: events[feature][offset:offset + actual_limit] 
                for feature in safe_features
            }
            
            # Convert to list of points
            for i in range(actual_limit):
                point = {}
                for feature in safe_features:
                    try:
                        value = feature_data[feature][i]
                        if isinstance(value, np.generic):
                            value = value.item()
                        point[feature] = value
                    except OSError:
                        continue
                data_points.append(point)
            
            return {
                "total": total_events,
                "offset": offset,
                "limit": actual_limit,
                "features": safe_features,
                "data": data_points
            }
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("Error reading RTDC file data")
        raise HTTPException(
            status_code=500,
            detail=f"Error reading RTDC file data: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint returning API info"""
    return {
        "message": "RTDC File API",
        "endpoints": [
            "/metadata - Get RTDC file metadata",
            "/data - Get all data points from the RTDC file",
            "/ - This help message"
        ]
    } 