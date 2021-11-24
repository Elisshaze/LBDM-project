# LBDM-project
Repository for the laboratory of biological data mining project, Oct. 2021
## Installation
Simply clone the repository
## Search significant
The `search_significant.py` script allows the user to merge (using pandas) the csv of the expansion lists of a gene of interest, retrieved from the gene@home portal. It merges based on the gene name to reduce complexity, so the user must take into account all isoforms listed in the final dataset when proceeding with the analysis.
Run the script with python. You must provide:
* (-i, --inputf) folder containing the csv files (not zipped) 
* (-l, --list) .txt file containing the list of genes that you want to check if they appear in the merged dataframe.
