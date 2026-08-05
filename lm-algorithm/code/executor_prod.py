###########################################
#   Copyright (C) 2024
#   Modified on April 9th 2025 by S. Al-Haj Baddar
###########################################
#   Authors:
#	- Sherenaz Al-Haj Baddar (s.baddar@ju.edu.jo)
#	- Alessandro Languasco (alessandro.languasco@unipd.it)
#	- Mauro Migliardi (mauro.migliardi@unipd.it)
#   This program is an entry point, it runs other the Python programs in this project and prompts the user for necessary input
###########################################
#   This software is licensed under the GNU General Public License version 3 (GPL-3), as published by the Free Software Foundation.
#   You can find a copy of the license at https://www.gnu.org/licenses/gpl-3.0.en.html
########################################################

import subprocess
import shutil
import os
import sys

def copier(source_dir, destination_dir):
    """
        Copies files from a source directory to a destination directory.
    """
    for root, dirs, files in os.walk(source_dir, topdown=False):
        for file in files:
            source_file = os.path.join(root, file)
            if('__pycache__' in source_file):
                continue
            relative_path = ''#os.path.relpath(source_file, source_dir)
            destination_file = os.path.join(destination_dir, relative_path)
            
            shutil.copy(source_file, destination_file)
            
def copy_files(mode):
   
    """
        decides how to call copier based on whether the user used pip from pypi or download files
    """
    # Get the current working directory
    current_dir = os.getcwd()
    destination_dir = current_dir

    if('p' in mode):
        abs_path = input('provide absolute path to alm2 installation, using pip show alm2:')
        source_dir = abs_path+'/alm2'
        copier(source_dir,destination_dir)
    else:
        abs_path = input('provide absolute path to alm2 installation:')
        source_dir_data = abs_path+'/the_data'
        source_dir_lib = abs_path+'/the_lib'
        copier(source_dir_data,destination_dir)
        copier(source_dir_lib,destination_dir)

    

            
def run_script():
  """
        Runs other Python scripts to implement and compare 3 techniques for estimating the log-likelihood function.
  """
  '''  
  # print current files
  files = [file for file in os.listdir(".") if file.endswith("csv") and "fix_counts" not in file]
  #files = os.listdir()
  for file in files:
      print(file)
  '''
  #read dataset file name from user
  #change the source_file according to the name of the dataset file you want to have, these are the options
  '''
  TIP.csv
  TCP.csv
  auto.csv
  cane.csv
  cavallo.csv
  space.csv
  mideast.csv
  f_vs_m.csv
  IoT.csv
  '''
  source_file = "IoT.csv"
  dataset_file = "all_fix_counts.csv"
  sample_file =   "test_fix_counts.csv"
  scrambler_script_path = "scrambler.py"
  ULang_script_path = "LogL-global-v5.py"

  #call file loader to copy it to all_fix_counts.csv
  try:
    shutil.copy(source_file, dataset_file)
    print(f"File copied successfully")
  except FileNotFoundError:
    print(f"Error: File not found at {source_file}")
    print("script aborted")
    sys.exit()
    
  #call scrambler to read the perentage of input and generate test_fix_counts.csv
  # change ratio here if you want, default ratio is 0.2
  # possible values are [0.05, 0.1, 0.15, 0.2, 0.25]
  ratio = str(0.05)
  try:
    subprocess.run(["python3.11", scrambler_script_path, ratio])
  except FileNotFoundError:
    print(f"Error: Script not found at {script_path}")
    print("script aborted")
    sys.exit()

  
  #call Uncle Lang script
  accuracy = str(6)
  precision = str(6)
  try:
    subprocess.run(["python3.11", ULang_script_path, accuracy, precision, source_file])
  except FileNotFoundError:
    print(f"Error: Script not found at {script_path}")
    print("script aborted")
    sys.exit()
  
  
#main program
'''
selector = input("enter p if you used pip, or d if you used download:")
if('p' in selector):
    copy_files('pip')
elif('d' in selector):
    copy_files('download')
'''
run_script()
