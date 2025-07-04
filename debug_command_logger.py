#!/usr/bin/env python3
"""
Debug command logger - logs each command before execution to help identify crash points
"""

import os
import sys
import traceback
from datetime import datetime


class CommandLogger:
    def __init__(self, log_file="debug_commands.log"):
        self.log_file = log_file
        self.start_time = datetime.now()
        self.log_entry(f"=== DEBUG SESSION STARTED: {self.start_time} ===")

    def log_entry(self, message):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
            f.flush()

    def log_command(self, command_name, details=""):
        """Log a command before execution"""
        self.log_entry(f"EXECUTING: {command_name} {details}")

    def log_success(self, command_name, result=""):
        """Log successful command completion"""
        self.log_entry(f"SUCCESS: {command_name} - {result}")

    def log_error(self, command_name, error):
        """Log command error"""
        self.log_entry(f"ERROR: {command_name} - {str(error)}")
        self.log_entry(f"TRACEBACK: {traceback.format_exc()}")

    def log_crash(self, location=""):
        """Log crash point"""
        self.log_entry(f"CRASH DETECTED AT: {location}")
        self.log_entry(f"TRACEBACK: {traceback.format_exc()}")


# Global logger instance
logger = CommandLogger()


def safe_execute(func, command_name, *args, **kwargs):
    """Safely execute a function with logging"""
    try:
        logger.log_command(command_name, f"args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logger.log_success(command_name, f"result type: {type(result)}")
        return result
    except Exception as e:
        logger.log_error(command_name, e)
        raise


if __name__ == "__main__":
    print("Debug command logger initialized")
    print(f"Log file: {logger.log_file}")
