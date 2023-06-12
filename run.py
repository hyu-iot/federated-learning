import argparse
import os
import config
import Server
import logging
from visualization import Visualization

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./configs/config.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')
parser.add_argument('-v', '--vis', type=str, default='null',
                    help='Option to skip training and only do visualization. Pass the path of the result.csv file.')
args = parser.parse_args()

# Set logging
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')


def main():
    """Run a federated learning simulation"""
    fl_config = config.Config(args)
    server = Server.Server(fl_config)
    # Run when 'only_visualization' option is false.
    if fl_config.visualization['only_visualization'] == 'null':
        server.run()
        vis = Visualization(fl_config, server.mydir)
    else: 
        logging.info('Only visualization option is on. Skipping training and starting visualization.')
        vis = Visualization(fl_config, fl_config.visualization['only_visualization'])
    vis.run()

if __name__ == "__main__":
    main()