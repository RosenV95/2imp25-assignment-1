import csv
import sys
from data import Data


if __name__ == "__main__":
    '''
    Entry point for the script
    '''
    if len(sys.argv) < 2:
        print("Please provide an argument to indicate which matcher should be used")
        exit(1)

    match_type = 0

    try:
        match_type = int(sys.argv[1])
    except ValueError as e:
        print("Match type provided is not a valid number")
        exit(1)

    type = int(sys.argv[1])
    if type == 0:
        treshold = 0
    if type == 1:
        treshold = 0.25
    if type == 2:
        treshold = 0.67

    data = Data("/input/low.csv", "/input/high.csv", treshold)
