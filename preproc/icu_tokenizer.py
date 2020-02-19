#-------------------------------------------------------------------------------------------------------------------
# Packages & Settings
#-------------------------------------------------------------------------------------------------------------------

# General Packages
import sys

# Math and Data Structures
import numpy as np

# ICU Tokenizer
from icu import UnicodeString, BreakIterator, Locale
#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------
# Tokenizing one Line
#-------------------------------------------------------------------------------------------------------------------

def tokenize_line(line,  locale):
	token_list = list()

	boundary = BreakIterator.createWordInstance(locale)
	boundary.setText(line)
	
	start = boundary.first()
	for end in boundary:
		token = line[start:end]
		token_list.append(token)
		start = end

	# Join to line
	tokenized_line = ' '.join(token_list)

	# Remove double spaces 
	tokenized_line = ' '.join(tokenized_line.split())

	return tokenized_line

#-------------------------------------------------------------------------------------------------------------------
# Iterating over File
#-------------------------------------------------------------------------------------------------------------------

input_filename = sys.argv[1]
output_filename = sys.argv[2]
language = sys.argv[3]

if (language) == 'hi':
	locale = Locale('hi_IN')

output_file = open(output_filename, 'w')

with open(input_filename, 'r') as input_file:
	untokenized_line = input_file.readline()
	while untokenized_line:
		tokenized_line = tokenize_line(untokenized_line, locale) + '\n'
		output_file.write(tokenized_line)
		untokenized_line = input_file.readline()