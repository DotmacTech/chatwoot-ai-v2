from fastapi.templating import Jinja2Templates
from pathlib import Path

# Initialize templates
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

def get_templates() -> Jinja2Templates:
    """Get the Jinja2Templates instance"""
    return templates
