import csv
import sys
from data import Data


def write_output_file():
    '''
    Writes a dummy output file using the python csv writer, update this
    to accept as parameter the found trace links.
    '''
    with open('/output/links.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)


        fieldnames = ["id", "links"]

        writer.writerow(fieldnames)

        writer.writerow(["UC1", "L1, L34, L5"])
        writer.writerow(["UC2", "L5, L4"])

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

    data = Data("/input/low.csv", "/input/high.csv")

    write_output_file()