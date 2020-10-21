from argparse import ArgumentParser
from utils.config import *
#from utils.general import *

from agents.mgaf import *
from datasets.entropy import *


def main():
    
    # parse the path of the json config file
    arg_parser = ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()

    # parse the config json file
    config = process_config(args.config)
    # Create the Agent and pass all the configuration to it then run it..
    
    #agent_class = globals()[config.agent]
    #agent = agent_class(config)
    agent = eval(config.agent)(config)
    agent.run()
    #agent.finalize()
    


if __name__ == '__main__':
    main()