import argparse
import os
import config
import Server

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./configs/config.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')
args = parser.parse_args()



def main():
    """Run a federated learning simulation"""
    fl_config = config.Config(args.config)
    
    server = Server.Server(fl_config)
    server.run()
    # name = 1
    pass


if __name__ == "__main__":
    main()