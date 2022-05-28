import logging

DEFAULT_LOG_LEVEL = logging.INFO

_logger_initialized = False
logger = logging.getLogger(__package__)

def _init_logger():
    """Initialize the pyddm logger object.
    
    The logger is already ready to call and use, but this ensures that
    messages are formatted as desired and output to console.
    """
    global _logger_initialized
    if not _logger_initialized:
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
        # Set default log-level
        _logger_initialized = True
        set_log_level(DEFAULT_LOG_LEVEL)

def set_log_level(level):
    """Sets the log level throughout pyddm.
    
    `level` must be an int or str as allowed by the Python logging
    module. See https://docs.python.org/3/library/logging.html for more.
    """
    if not _logger_initialized:
        _init_logger()
    logger.setLevel(level)