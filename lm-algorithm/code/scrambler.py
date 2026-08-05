###########################################
#   Copyright (C) 2024
###########################################
#   Authors:
#	- Sherenaz Al-Haj Baddar (s.baddar@ju.edu.jo)
#	- Alessandro Languasco (alessandro.languasco@unipd.it)
#	- Mauro Migliardi (mauro.migliardi@unipd.it)
#   This program read a file that contains a count dataset and produces a random sample from it in another file
########################################################
#     This software is licensed under the GNU General Public License version 3 (GPL-3), as published by the Free Software Foundation.
#     You can find a copy of the license at https://www.gnu.org/licenses/gpl-3.0.en.html
########################################################

import random
import sys
ratio = float(sys.argv[1])
#ratio = sys.argv[1]
def read_file(file_in, file_out):
    lines_1 = []
    with open(file_in) as f:
        lines_1 = f.readlines()
    f.close()
    sample_size = (int)(ratio*len(lines_1))
    random_sample = random.sample(range(0, len(lines_1)-1), sample_size)
    with open(file_out, 'w') as my_file:
        for i in range(len(random_sample)):
            loc = random_sample[i]
            my_file.write(lines_1[loc])
    my_file.close()
    print("Scrambling is Done");

file_in = "all_fix_counts.csv"
file_out = "test_fix_counts.csv"
read_file(file_in, file_out)
