import os
from pathlib import Path
import logging


# Setting up logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

# Embedded content of exception.py and logger.py
EXCEPTION_CONTENT = 'import sys\n\n\n# ________________ DEF THE ERROR MESSAGE ________________ #\ndef error_message_detail(error, error_detail: sys):\n    """\n    This function is to return a message regarding error details occuring in the execution of the code\n\n    """\n\n    # no interest in the 1st and 2nd items in the return of the exc_info()\n    _, _, exc_tb = error_detail.exc_info()\n\n    file_name = exc_tb.tb_frame.f_code.co_filename\n    error_message = "Error occured in the script, name: [{0}], line number: [{1}] error message: [{2}]".format(\n        file_name, exc_tb.tb_lineno, str(error)\n    )\n\n    return error_message\n\n\n# ________________ MAKE ERROR CAPTURE HANDLER ________________ #\nclass CustomException(Exception):\n    def __init__(self, error_message, error_detail: sys):\n        super().__init__(error_message)\n        self.error_message = error_message_detail(\n            error_message, error_detail=error_detail\n        )\n\n    def __str__(self):\n        return self.error_message\n'

LOGGER_CONTENT = 'import logging, os\nfrom datetime import datetime\n\nLOG_FILE = f"{datetime.now().strftime(\'%m _%d_%Y_%H_%M_%S\')}.log"\nlogs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)\nos.makedirs(logs_path, exist_ok=True)  # keep on appending the file\n\nLOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)\n\nlogging.basicConfig(\n    filename=LOG_FILE_PATH,\n    # recommended format\n    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",\n    level=logging.INFO,\n)\n'

COMMON_CONTENT = """# AWS
scp -r -i <PEM key> <AWS path> <Local path>
scp -r -i computervisionec2.pem ubuntu@ec2-15-156-80-182.ca-central-1.compute.amazonaws.com:/home/ubuntu/Projects/Bird_classification/Research/output /home/hoangng/Projects/Bird_Classification/Research/from_ec2/

scp -r -i <PEM key> <Local path> <AWS path>
scp -i computervisionec2.pem /home/hoangng/Projects/Bird_Classification/Research/models/model_base_test1.py ubuntu@ec2-15-156-80-182.ca-central-1.compute.amazonaws.com:/home/ubuntu/Projects/Bird_classification/Research/models/model_base_test1.py

aws s3 sync <local path> <s3:// path>
aws s3 sync /home/ubuntu/Projects/Bird_classification/Research/output/data s3://computervision0/output/data

aws s3 cp <s3:// path> <local path>
aws s3 cp s3://computervision0/models/mobilenetv2_finetune1.py /home/ubuntu/Projects/Bird_classification/Research/models/mobilenetv2_finetune1.py

ps aux | grep <process-name>
ps aux | grep mobilenet_finetune1.py

screen -D -r <screen ID> # Re-attach to a screen on EC2

screen -S <name-of-the-screen-session> # Create a screen session on EC2

# Heroku

heroku --version

heroku login 

heroku container:login

heroku create <app-name>

heroku container:push web --app <app-name>

heroku ps:scale web=<number> --app <app-name> #web = 1: $7/month or $5/month

heroku container:release web --app <app-name>

# Note: need to be inside the app folder for Heroku commands"""


# List of files and directories
list_of_files = [
    "input",
    "input/data",
    "input/data/train",
    "input/data/val",
    "input/data/test",
    "input/viz",
    "input/data/real_imgs",
    "output",
    "output/data",
    "output/viz",
    "notebook",
    "notebook/EDA.ipynb",
    "notebook/draft.ipynb",
    "models",
    # "models/save_best_models",
    "from_ec2",
    "resource/data_org",
    "resource/pretrained_models",
    "resource/common.txt",
    "Temp",
    "logger.py",
    "exception.py",
    "utils_model.py",
    "utils_data.py",
    "requirements.txt",
]

for file_path in list_of_files:
    file_path = Path(file_path)

    # If it's a directory or doesn't have a dot (assuming it's a directory)
    if file_path.is_dir() or "." not in file_path.name:
        if not file_path.exists():
            os.makedirs(file_path, exist_ok=True)
            logging.info(f"Created directory: requirements.txt")
        else:
            logging.info(
                f"Directory requirements.txt already exists => re-creating ignored."
            )
    else:
        file_dir, file_name = os.path.split(file_path)

        if file_dir != "" and not Path(file_dir).exists():
            os.makedirs(file_dir, exist_ok=True)
            logging.info(f"Created directory:  for the file requirements.txt")

        # If the file does not exist or its size is 0
        if not file_path.exists() or file_path.stat().st_size == 0:
            with open(file_path, "w") as file:
                # Write the content of exception.py if the file is exception.py
                if file_name == "exception.py":
                    file.write(EXCEPTION_CONTENT)
                # Write the content of logger.py if the file is logger.py
                elif file_name == "logger.py":
                    file.write(LOGGER_CONTENT)
                # Else, just create an empty file
                elif file_name == "common.txt":
                    file.write(COMMON_CONTENT)
                else:
                    pass
            logging.info(f"Created file: requirements.txt")
        else:
            logging.info(
                f"File requirements.txt already exists and is not empty => re-creating ignored."
            )

if __name__ == "__main__":
    print("Project structure generated successfully!")




