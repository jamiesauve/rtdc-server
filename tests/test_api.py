import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import h5py
import numpy as np
from app import app, limiter, _rate_limit_exceeded_handler
from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request

@pytest.fixture(autouse=True)
def reset_limiter():
    """Reset rate limiter before each test"""
    limiter.reset()

@pytest.fixture
def mock_rtdc_file(tmp_path):
    """Create a mock RTDC file for testing"""
    file_path = tmp_path / "test.rtdc"
    with h5py.File(file_path, 'w') as f:
        # Create attributes
        f.attrs.create('experiment:date', '2024-01-01', dtype='S10')
        f.attrs.create('experiment:sample', 'test_sample', dtype='S10')
        f.attrs.create('experiment:time', '12:00:00', dtype='S8')
        
        # Create events group with more test data
        group = f.create_group('events')
        # Create 1000 test points
        test_data = np.arange(1000) * 100
        group.create_dataset('area', data=test_data)
        
        print(f"Test file attributes: {dict(f.attrs)}")
    
    return file_path

@pytest.fixture
def client(mock_rtdc_file):
    """Create a test client with mock file"""
    app.state.RTDC_FILE_PATH = mock_rtdc_file
    return TestClient(app)

@pytest.fixture
def invalid_rtdc_file(tmp_path):
    """Create an invalid RTDC file (no events group) for testing"""
    invalid_file = tmp_path / "invalid.rtdc"
    with h5py.File(invalid_file, 'w') as f:
        f.attrs.create('test', 'value')
        f.create_group('not_events')
    return invalid_file

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
    # Make 51 requests (one over the limit)
    responses = [client.get("/metadata") for _ in range(51)]
    
    # First 50 should succeed
    for i, r in enumerate(responses[:50]):
        assert r.status_code == 200, f"Request {i+1} failed with {r.status_code}"
    
    # 51st should be rate limited
    assert responses[50].status_code == 429

def test_data_endpoint(client):
    """Test the data endpoint returns correct event data"""
    response = client.get("/data")
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert all(k in data for k in ["total", "features", "data"])
    assert isinstance(data["total"], int)
    assert isinstance(data["features"], list)
    assert isinstance(data["data"], list)
    
    # Check data content
    assert "area" in data["features"]
    assert len(data["data"]) == data["total"]
    assert len(data["data"]) == 1000  # We created 1000 test points
    
    # Check first data point structure
    first_point = data["data"][0]
    assert "area" in first_point
    assert first_point["area"] == 0

def test_data_endpoint_file_not_found():
    """Test data endpoint with missing file"""
    test_app = FastAPI()
    test_app.state.RTDC_FILE_PATH = Path("nonexistent.rtdc")
    
    test_client = TestClient(test_app)
    response = test_client.get("/data")
    assert response.status_code == 404

def test_data_endpoint_invalid_file(client, invalid_rtdc_file):
    """Test data endpoint with invalid RTDC file"""
    # Temporarily change the file path
    original_path = app.state.RTDC_FILE_PATH
    app.state.RTDC_FILE_PATH = invalid_rtdc_file
    
    try:
        response = client.get("/data")
        assert response.status_code == 500
        assert "no events group found" in response.json()["detail"]
    finally:
        # Restore original path even if test fails
        app.state.RTDC_FILE_PATH = original_path

def test_data_endpoint_pagination(client):
    """Test data endpoint pagination"""
    # Test default pagination (should return default limit)
    response = client.get("/data")
    data = response.json()
    assert response.status_code == 200
    assert len(data["data"]) == 1000  # Default limit
    assert data["offset"] == 0
    
    # Test with custom offset and limit
    response = client.get("/data?offset=1&limit=2")
    data = response.json()
    assert response.status_code == 200
    assert len(data["data"]) == 2
    assert data["offset"] == 1
    assert data["limit"] == 2
    
    # Test offset beyond available data
    try:
        response = client.get("/data?offset=1001")  # Beyond our 1000 test points
        assert response.status_code == 400
        assert "exceeds total events" in response.json()["detail"]
    except Exception as e:
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.json()}")
        raise

def test_data_endpoint_range_header(client):
    """Test data endpoint with range header"""
    response = client.get(
        "/data",
        headers={"Range": "events=0-499"}
    )
    data = response.json()
    assert response.status_code == 200
    assert len(data["data"]) == 500
    assert data["offset"] == 0
    assert data["limit"] == 500

    # Test invalid range format
    response = client.get(
        "/data",
        headers={"Range": "invalid"}
    )
    assert response.status_code == 400 

def test_data_endpoint_range_header_comprehensive(client):
    """Test all range header scenarios"""
    # Test basic range
    response = client.get(
        "/data",
        headers={"Range": "events=0-499"}
    )
    data = response.json()
    assert response.status_code == 200
    assert len(data["data"]) == 500
    assert data["offset"] == 0
    assert data["limit"] == 500

    # Test range with different start
    response = client.get(
        "/data",
        headers={"Range": "events=100-599"}
    )
    data = response.json()
    assert response.status_code == 200
    assert len(data["data"]) == 500
    assert data["offset"] == 100
    assert data["limit"] == 500

    # Test invalid range type
    response = client.get(
        "/data",
        headers={"Range": "invalid=0-499"}
    )
    assert response.status_code == 400
    assert "Invalid range header format" in response.json()["detail"]

    # Test malformed range values
    response = client.get(
        "/data",
        headers={"Range": "events=abc-def"}
    )
    assert response.status_code == 400
    assert "Invalid range header format" in response.json()["detail"]

    # Test query params override range header
    response = client.get(
        "/data",
        headers={"Range": "events=0-499"},
        params={"offset": 1, "limit": 2}
    )
    data = response.json()
    assert response.status_code == 200
    assert len(data["data"]) == 2
    assert data["offset"] == 1
    assert data["limit"] == 2

    # Test end greater than start
    response = client.get(
        "/data",
        headers={"Range": "events=500-0"}
    )
    assert response.status_code == 400
    assert "Invalid range" in response.json()["detail"]

    # Test missing range parts
    response = client.get(
        "/data",
        headers={"Range": "events=500"}
    )
    assert response.status_code == 400
    assert "Invalid range header format" in response.json()["detail"] 