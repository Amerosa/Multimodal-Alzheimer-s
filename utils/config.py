import json
import os

import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

from pprint import pprint

from utils.basic_utils import create_dirs
from easydict import EasyDict

def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)

def get_config_from_json(json_file):
    with open(json_file, 'r') as f:
        try:
            config = json.load(f)
            return EasyDict(config)
        except ValueError:
            print('Invalid json file format. Please make sure your config file is formatted properly!')
            exit(-1)

def process_config(json_file):

    config = get_config_from_json(json_file)
    print("Processing the config file for the experiment..")
    pprint(config)

    # making sure that you have provided the exp_name.
    try:
        print('*' * 50)
        print(f"Configuration for the experiment {config.exp_name}")
        print('*' * 50)
    except AttributeError:
        print("ERROR!! Please provide the exp_name in json file!")
        exit(-1)

    config.exp_root = os.path.join("experiments", config.exp_name)
    # create some important directories to be used for that experiment.
    config.summary_dir = os.path.join("experiments", config.exp_name, "summaries/")
    config.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoints/")
    config.out_dir = os.path.join("experiments", config.exp_name, "out/")
    config.log_dir = os.path.join("experiments", config.exp_name, "logs/")
    create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir])

    # setup logging in the project
    setup_logging(config.log_dir)

    logging.getLogger().info("Hi, This is root.")
    logging.getLogger().info("After the configurations are successfully processed and dirs are created.")
    logging.getLogger().info("The pipeline of the project will begin now.")

    return config
