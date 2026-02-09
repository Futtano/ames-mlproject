import sys
from typing import Any


def get_error_details(error: Exception, error_details: Any | None = None) -> str:
    """Format detailed error message including file name and line number.

    Args:
        error (Exception): The exception object that occurred.
        error_details (Any, optional): The sys module or object containing exception info.
                                     Defaults to sys.exc_info() if None.

    Returns:
        str: Formatted error message with file path, line number, and error text.
    """
    if error_details is None:
        _, _, exc_tb = sys.exc_info()
    else:
        _, _, exc_tb = error_details.exc_info()

    file_name: str = "unknown"
    line_number: str = "unknown"

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = str(exc_tb.tb_lineno)

    error_class = error.__class__.__name__
    error_message = f"[{error_class}] in [{file_name}] at line {line_number}: {str(error)}"

    return error_message


class CustomException(Exception):
    """Custom exception class for project-specific error handling.

    Attributes:
        error_message (str): The detailed, formatted error message.
    """

    def __init__(self, error: Exception, error_details: Any | None = None):
        """Initialize the CustomException with detailed error information.

        Args:
            error (Exception): The original exception.
            error_details (Any | None, optional): Exception context. Defaults to sys.exc_info().
        """
        super().__init__(str(error))
        self.error_message = get_error_details(error=error, error_details=error_details)

    def __str__(self) -> str:
        """Return the detailed error message as the string representation."""
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        # Now you can just do this:
        raise CustomException(e) from e
