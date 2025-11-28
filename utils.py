#!/usr/bin/env python3
"""
Utility functions for the Dropout Risk Assessment System
- Logging setup
- Input validation
- Error handling helpers
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import re

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log file
    """
    # Create logger
    logger = logging.getLogger('dropout_risk_assessment')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# ============================================================================
# INPUT VALIDATION
# ============================================================================

def validate_uid(uid: str) -> tuple[bool, str]:
    """
    Validate case UID format.
    
    Args:
        uid: Case UID to validate
        
    Returns:
        (is_valid, error_message)
    """
    if not uid:
        return False, "UID cannot be empty"
    
    if len(uid) > 255:
        return False, "UID must be 255 characters or less"
    
    # Allow alphanumeric, underscores, hyphens
    if not re.match(r'^[a-zA-Z0-9_-]+$', uid):
        return False, "UID can only contain letters, numbers, underscores, and hyphens"
    
    return True, ""

def validate_case_description(text: str, min_length: int = 50) -> tuple[bool, str]:
    """
    Validate case description text.
    
    Args:
        text: Case description text
        min_length: Minimum required length
        
    Returns:
        (is_valid, error_message)
    """
    if not text:
        return False, "Case description cannot be empty"
    
    if len(text.strip()) < min_length:
        return False, f"Case description must be at least {min_length} characters"
    
    if len(text) > 50000:  # Reasonable upper limit
        return False, "Case description is too long (max 50,000 characters)"
    
    # Check for potential PII patterns
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{10,}\b'
    
    if re.search(email_pattern, text):
        return False, "Case description appears to contain an email address. Please remove PII."
    
    if re.search(phone_pattern, text) and len(re.findall(phone_pattern, text)) > 2:
        return False, "Case description appears to contain phone numbers. Please remove PII."
    
    return True, ""

def sanitize_text(text: str) -> str:
    """
    Basic text sanitization (removes potentially dangerous characters).
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)
    
    return text.strip()

# ============================================================================
# ERROR HANDLING HELPERS
# ============================================================================

def log_error(error: Exception, context: str = "", user_id: Optional[str] = None):
    """
    Log errors with context for better debugging.
    
    Args:
        error: Exception object
        context: Additional context about where error occurred
        user_id: Optional user ID for audit trail
    """
    error_msg = f"Error in {context}: {str(error)}"
    if user_id:
        error_msg += f" (User: {user_id})"
    logger.error(error_msg, exc_info=True)

def format_error_message(error: Exception, context: str = "") -> str:
    """
    Format user-friendly error messages.
    
    Args:
        error: Exception object
        context: Context about the operation
        
    Returns:
        User-friendly error message
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    # User-friendly messages for common errors
    if "connection" in error_msg.lower() or "psycopg2" in error_type:
        return f"❌ Database connection error. Please check your database configuration."
    elif "permission" in error_msg.lower() or "access" in error_msg.lower():
        return f"🚫 Access denied. You don't have permission to perform this action."
    elif "not found" in error_msg.lower():
        return f"❌ {context or 'Resource'} not found."
    elif "duplicate" in error_msg.lower() or "unique" in error_msg.lower():
        return f"⚠️ This record already exists. Please use a different identifier."
    else:
        return f"❌ {context or 'An error occurred'}: {error_msg}"

# ============================================================================
# ENVIRONMENT VARIABLES HELPERS
# ============================================================================

def get_db_config() -> Dict[str, Any]:
    """
    Get database configuration from environment variables or config file.
    
    Returns:
        Dictionary with database configuration
    """
    import json
    from pathlib import Path
    
    # Try environment variables first
    config = {
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
        'database': os.getenv('DB_NAME'),
        'username': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }
    
    # If all env vars are set, use them
    if all(config.values()):
        logger.info("Using database configuration from environment variables")
        return config
    
    # Fallback to config file
    config_file = Path("db_config.json")
    if config_file.exists():
        logger.info("Loading database configuration from db_config.json")
        with open(config_file, 'r') as f:
            file_config = json.load(f)
        
        # Merge with env vars (env vars take precedence)
        for key in config:
            if config[key]:
                file_config[key] = config[key]
        
        return file_config
    
    # If neither exists, raise error
    raise ValueError(
        "Database configuration not found. "
        "Set environment variables (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD) "
        "or create db_config.json file."
    )



