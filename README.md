# weather-eodp
Uses supervised machine learning to predict energy consumption based on weather conditions. 

# Members: 
1007904, adras, Samer Adra
1217370, rholstege, Remco Holstege
1226974, adasgupta, Amritesh Dasgupta

# Introduction of intended use for code
# File structure and use of each file
Main.py, controls the data processing pipeline.
Merge.py, merges initial data files together.
Preprocessing.py, does preprocessing on merged data, and some graphing.
Analysis.py, does regression, classification, and more graphing on the processed data.
utils.py, has universal functions, imports, column groups and variables.
/weather holds the initial data, that was merged.
/merged holds the merged data.
/processed holds the preprocessed merged data.
/graphs holds many graphs, organised by state. 

# Instructions on how to run your code.
Make sure that "do_merge", "do_preprocess" and "do_analysis" are set to True in Main.py.
Then do command: "python main.py"

# Any additional requirements needed to run code.
None
