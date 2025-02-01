import sys
from src.logger import logging

def error_message_detail(error,error_detail:sys): 
    """
    This function is used to get the error message with the file name and line number where the error occurred.

    Args: 
        error: error message
        error_detail: sys.exc_info() object
        
    Returns:
        error_message: error message with file name and line number
        
    """
    _,_,exc_tb=error_detail.exc_info() 
    file_name=exc_tb.tb_frame.f_code.co_filename # file name where error occurred 
    error_message = "Error occuered in file: "+file_name+" at line number: "+str(exc_tb.tb_lineno)+" with error message: "+str(error)
    return error_message


class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail)

        def __str__(self):
            return self.error_message