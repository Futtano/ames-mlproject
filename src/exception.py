import sys

def get_error_details(error, error_details: sys) -> str:  
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename  
    error_message = "Error occurred in file [{0}], line {1}, error messsage: {2}".format(file_name, exc_tb.tb_lineno, str(error))

    return error_message

class CustomException(Exception):
    def __init__(self, error, error_details):
        super().__init__(error)
        self.error_message = get_error_details(error=error, error_details=error_details)

    def __str__(self):
        return self.error_message
    
if __name__ == '__main__':
    try:
        a = 1/0
    except Exception as e:
        raise(CustomException(e, sys))