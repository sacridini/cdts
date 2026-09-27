import ee
from typing import Optional

def initialize_gee(project: Optional[str] = None) -> None:
    """
    Initializes the Google Earth Engine Python API.
    If authentication is required, it prompts the user to authenticate.
    
    Args:
        project (str, optional): The Google Cloud Project ID to use for authentication.
    """
    try:
        if project:
            ee.Initialize(project=project, opt_url='https://earthengine-highvolume.googleapis.com')
        else:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    except Exception:
        print("GEE authentication required. Please authenticate...")
        ee.Authenticate()
        if project:
            ee.Initialize(project=project, opt_url='https://earthengine-highvolume.googleapis.com')
        else:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
