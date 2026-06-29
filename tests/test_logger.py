import os
import logging
import shutil
from src.utils.logger import setup_logger

def test_setup_logger():
    """
    Test if the logger is created correctly and log file is generated.
    """
    log_dir = "tests/test_logs"
    logger = setup_logger(name="TestLogger", log_dir=log_dir)
    
    # Assertions to prove logger is correctly configured
    assert isinstance(logger, logging.Logger), "Logger is not an instance of logging.Logger"
    assert logger.name == "TestLogger", "Logger name mismatch"
    
    # Log a test message
    logger.info("Test message")
    
    # Check if a log file was created in the directory
    assert os.path.exists(log_dir), "Log directory was not created"
    log_files = os.listdir(log_dir)
    assert len(log_files) > 0, "No log files were found in directory"
    assert any(f.startswith("training_") for f in log_files), "Log filename format is incorrect"
    
    # Cleanup to ensure tests don't pollute the workspace
    shutil.rmtree(log_dir)
