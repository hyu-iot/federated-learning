import argparse
import os

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')
args = parser.parse_args()



def main():
    """Run a federated learning simulation"""
    pass


if __name__ == "__main__":
    main()