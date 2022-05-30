import sys

# fname and search_str are the first and second arguments passed to
# the program when it's called
fname = sys.argv[1]
search_str = sys.argv[2]

# Open fname (for reading) and iterate through each line in it
with open(fname) as infile:
    for line in infile:
        # Split line based on white space
        line = line.split()
        # The frequency is at index 0, the word is at index 1
        word = line[1]
        # Check if search_str is a substring of word; apply case folding
        # before cheking
        if search_str.lower() in word.lower():
            print(word)
