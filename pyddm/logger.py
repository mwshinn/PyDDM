import logging

DEFAULT_LOG_LEVEL = logging.INFO

logger = logging.getLogger(__package__)

# Rewrite the textual representation of logging levels to make them a bit less scary
logging.addLevelName(logging.DEBUG, 'Debug')
logging.addLevelName(logging.INFO, 'Info')
logging.addLevelName(logging.WARNING, 'Warning')
logging.addLevelName(logging.ERROR, 'Error')
logging.addLevelName(logging.CRITICAL, 'Critical')
# Define custom formatting: produces messages that look like "Loglevel: <message here...>"
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console_handler)

def set_log_level(level):
    """Sets the log level throughout pyddm.
    
    `level` must be an int or str as allowed by the Python logging
    module. See https://docs.python.org/3/library/logging.html for more.
    """
    logger.setLevel(level)

# Set default log-level
set_log_level(DEFAULT_LOG_LEVEL)
