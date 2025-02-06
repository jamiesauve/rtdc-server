import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import h5py
import numpy as np
from app import app, limiter
from fastapi import FastAPI

@pytest.fixture(autouse=True)
def reset_limiter():
    """Reset rate limiter before each test"""
    limiter.reset()

@pytest.fixture
def mock_rtdc_file(tmp_path):
    """Create a mock RTDC file for testing"""
    file_path = tmp_path / "test.rtdc"
    with h5py.File(file_path, 'w') as f:
        # Create attributes with experiment: prefix to match real RTDC files
        f.attrs.create('experiment:date', '2024-01-01', dtype='S10')
        f.attrs.create('experiment:sample', 'test_sample', dtype='S10')
        f.attrs.create('experiment:time', '12:00:00', dtype='S8')
        
        # Create required groups
        group = f.create_group('events')
        group.create_dataset('area', data=np.array([100, 200, 300]))
        
        # Verify attributes were set
        print(f"Test file attributes: {dict(f.attrs)}")
    
    return file_path

@pytest.fixture
def client(mock_rtdc_file):
    """Create a test client with mock file"""
    app.state.RTDC_FILE_PATH = mock_rtdc_file
    return TestClient(app)

def test_root_endpoint(client):
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "RTDC File API"
    assert isinstance(data["endpoints"], list)

def test_metadata_endpoint(client):
    """Test metadata endpoint"""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert all(k in data for k in ["filename", "groups", "attributes"])
    assert "events" in data["groups"]
    assert data["attributes"]["experiment:date"] == "2024-01-01"

def test_metadata_file_not_found():
    """Test file not found error"""
    nonexistent_path = Path("nonexistent.rtdc")
    
    # Create a new app instance for this test
    test_app = FastAPI()
    test_app.state.RTDC_FILE_PATH = nonexistent_path
    
    test_client = TestClient(test_app)
    response = test_client.get("/metadata")
    assert response.status_code == 404

def test_cors_headers(client):
    """Test CORS headers"""
    response = client.get(
        "/metadata",
        headers={"Origin": "http://localhost:5173"}
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"

def test_security_headers(client):
    """Test security headers"""
    response = client.get("/")
    assert response.headers["x-xss-protection"] == "1; mode=block"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"

def test_rate_limiting(client):
    """Test rate limiting"""
    # Make 6 requests
    responses = [client.get("/metadata") for _ in range(6)]
    
    # First 5 should succeed
    for i, r in enumerate(responses[:5]):
        assert r.status_code == 200, f"Request {i+1} failed with {r.status_code}"
    
    # 6th should be rate limited
    assert responses[5].status_code == 429 