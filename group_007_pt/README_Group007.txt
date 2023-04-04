COMP0197: Applied Deep Learning
Assessed Component 2 (Group Work – 25%) 2022-23

Group 007
Student IDs: 22212027, 22198951, 22197823, 19090665, 22146100, 22143200, 22178092, 22209135

******************
ENVIRONMENT SETUP
******************
1. Development environment
The module coursework uses Python, NumPy and PyTorch. The Development environment document contains details of the supported development environment, though it is not mandatory.

2. Quick start
To run the coursework, follow the instruction below.

First, set up the environment:

conda create --name comp0197-cw2-pt pytorch torchvision
conda activate comp0197-cw2-pt

No additional libraries are installed for this project.


******************
RUNNING SCRIPTS
******************
1. change directory cd to unzipped submited folder to run training and evaluate scripts.
2. To Train a model with with L = 25% labelled / (1-L) = 75% unlabelled data (referred to as 'M25' or 'M-25' in the reports). This also trains two benchmark supervised models (lower and upper bounds), with L% labelled data (referred to as 'M25L' or 'M-25L' in the report, and 100% data respectively ('MU' or M-100'). To do so, run the following and change the float after the .py for a different L (labelled percentage):
2a. eg, to train models M25, M25L, M100 run:
 - python main_pt.py 0.25 

3. To evaluate all of our models with our evaluation metrics run the following:
 - python main_pt.py evaluate    


