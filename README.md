# ec_bandit

This is the source code for the NIPS2018 paper: Bandit Learning with Implicit Feedback.

Version and Package requirement:
python3 supported with numpy/scipy/matplotlib/pandas/coloredlogs. All package should be easily installed with conda.

Data:
Please go to the following link and download the data named Xiaomu Questioning:
  http://moocdata.cn/data
The description of the data is contained in the downlowded package.

Please unpack the package and move the data file(rawData.pkl) to the subdirectory ./data/ (configured in config.py)
Then the following command should run the training phase:
  > python train.py
 
To see the figure of cumulative reward, try:
  > python plot.py

The figure should be plotted and saved.

