# sys -> manipulate the exceptions in the python libraries.
import sys 
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    # exec_tb consists the specific details about the occurences of the execption.. i.e. what line it is, which file is it, etc. 
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, 
        exc_tb.tb_lineno, 
        str(error)
        )
    logging.info(f"Error: {error_message}")
    return error_message
    
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
    