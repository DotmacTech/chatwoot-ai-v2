import logging
from typing import Dict, Any, Optional

class AppLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create console handler if no handlers exist
        if not self.logger.handlers:
            # Create console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        self.logger.debug(message, extra=extra or {})
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        self.logger.info(message, extra=extra or {})
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        self.logger.warning(message, extra=extra or {})
    
    def error(self, message: str, exc_info: bool = False, extra: Optional[Dict[str, Any]] = None):
        self.logger.error(message, exc_info=exc_info, extra=extra or {})
