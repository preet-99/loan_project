import logging 
import os
from datetime import datetime

# name of the file
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# print(LOG_FILE)

# define path of logs folder in current working directory
logs_path = os.path.join(os.getcwd(), "logs",LOG_FILE) 

# make logs folder
os.makedirs(logs_path, exist_ok=True)