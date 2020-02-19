#-------------------------------------------------------------------------------------------------------------------
# Packages & Settings
#-------------------------------------------------------------------------------------------------------------------

import datetime
import os
#-------------------------------------------------------------------------------------------------------------------
# WRITE LOG LINE
#
# Takes a filestream, a key and a value and writes a line in the specified log format to the file.
#-------------------------------------------------------------------------------------------------------------------	

def write_log_line(log, desc, val, size):
	log.write('{desc:{size}s}{val}\n'.format(desc = desc, val = val, size = size))

#-------------------------------------------------------------------------------------------------------------------
# PRINT LOG LINE
#
# Takes a key and a value and prints a line in the specified log format to the Console.
#-------------------------------------------------------------------------------------------------------------------

def print_log_line(desc, val, size):
	print('{desc:{size}s}{val}'.format(desc = desc, val = val, size = size))

#-------------------------------------------------------------------------------------------------------------------
# FILETYPE CHECK
#
# Ensures filename extensions are correct & changes them otherwise.
#-------------------------------------------------------------------------------------------------------------------

def get_filename(folder, name, ending):
	if (folder[-1] != '/'):
		folder += '/'

	if not os.path.exists(folder):
		os.makedirs(folder)

	file_name = folder + name

	if(file_name.find('.') == -1):
		return(file_name + '.' + ending)
	else:
		return(file_name[:file_name.rfind('.')+1] + ending)

#-------------------------------------------------------------------------------------------------------------------
# TBD
#-------------------------------------------------------------------------------------------------------------------

def get_word_counts_from_file(folder, name, ending):
	TBD