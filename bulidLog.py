import sys
import os
import time
from typing import Any

# saving log recorder
class Logger(object):

    def __init__(self, output_dir: str, log_name: str='',stream: Any=sys.stdout) -> None:
        #output_dir = os.path.dirname(os.path.realpath(__file__)  # folder 
        output_dir = os.path.realpath(output_dir)
        print(f'Save log in {output_dir}')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        #log_name = '{}.txt'.format(time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time())))
        log_name_time = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
        log_name = f"{log_name}_{log_name_time}.log"
            
        if stream == sys.stderr:
            log_name = 'Error_' + log_name

        filename = os.path.join(output_dir, log_name)
        
        self.terminal = stream
        self.log = open(filename, 'a+')
    # get print information both in terminal and log files
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass