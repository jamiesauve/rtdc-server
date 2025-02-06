from fastapi import FastAPI, HTTPException
import h5py
import json
from pathlib import Path

app = FastAPI(
    title="RTDC File API",
    description="API for serving RTDC file metadata"
)

# Configure this path to point to your .rtdc file
RTDC_FILE_PATH = Path("path/to/your/file.rtdc")

@app.get("/metadata")
async def get_metadata():
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