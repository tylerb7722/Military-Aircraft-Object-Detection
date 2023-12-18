# Military Aircraft Object Detection Using YOLOv8

-------------------- Created by Tyler Boudreau --------------------

Check Requirements file for the lists of Packages used and required to run, however the list may not be inclusive of everything, check console error logs if needed to install dependencies. You must also first download the dataset from Kaggle and extract it as well:
https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset

-------------------- Important Notice --------------------

Check dataset to see if it was updated with any additional classes not included in the .yaml file. If there is additional classes in the dataset, the .yaml must be updated or an out of bounds error will occur.


This project focuses on training YOLOv8 models in python to detect 43 different classes of military aircraft, as well as an application that overlays the military aircraft classification on any desired video file. 

-------------------- This Repository Contains Two Main Files. --------------------

The first being the Jupyter NoteBook file 'Military_Aircraft_Detection_YOLOV8_NoteBook_T2.ipynb' which shows the code and steps taken to create the YOLOv8 model, train it on the Militry Aircraft Dataset, and Create an overlay for any desired video using the model weights as well.

The second is the Python source code file 'Aircraft_Detection_Application_Source_Code1.py' which includes the full appplication source code with user input prompts to use the desired YOLOv8 model weights, and choose a video to overlay with the military aircraft classification results.

There is also Two example picture PNG files showing the application running the overlay on an example video, as well as an example user input and console output.

Link of Original Dataset: https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset
