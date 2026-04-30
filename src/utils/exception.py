import sys


def _format(error) -> str:
    _, _, exc_tb = sys.exc_info()
    if exc_tb is None:
        return f"Error: {error}"
    file_name = exc_tb.tb_frame.f_code.co_filename
    return (
        f"Error in [{file_name}] line [{exc_tb.tb_lineno}] message [{error}]"
    )


class CustomException(Exception):
    def __init__(self, error_message, error_detail=None):
        super().__init__(error_message)
        self.error_message = _format(error_message)

    def __str__(self) -> str:
        return self.error_message
