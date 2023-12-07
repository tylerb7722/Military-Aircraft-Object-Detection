# Created by Tyler Boudeau
# Using Custom Trained YOLOv8 Model for Model Weights
# Trained on Miitary Aircraft Detection Dataset: https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset

# Import Required Packages:
import cv2
import os
from sahi import AutoDetectionModel
from sahi.predict import get_prediction
from pathlib import Path
from ultralytics.utils.files import increment_path
from sahi.utils.yolov8 import download_yolov8l_model

print("\nCreated By Tyler Boudreau\n")

print("Welcome To the Military Aircraft Detection Video Overlay Program\n")

Weight_Exception=True
while(Weight_Exception==True):
    weights=input(r'Enter the file location of the YOLOv8 Model Weights: For Example Enter: C:\\Users\\Tyler\\Desktop\\Orig_Data_Aircraft_Detection\\data\\runs\\detect\\train8\\weights\\best.pt')

    weights_exist = os.path.exists(weights)

    if (weights_exist==False):
        print("\nError: File path for Weights not found, Please try again.")
    elif (weights_exist==True):
        Weight_Exception=False

Source_Exception = True
print("\n")
while(Source_Exception==True):
    source=input(r'Enter the file Location of the Source Video to analyze: For Example Enter: C:\\Users\\Tyler\\Downloads\\Aircraft_Video_Test.mp4')

    weights_exist = os.path.exists(source)

    if (weights_exist==False):
        print("\nError: File path for Video not found, Please try again.")
    elif (weights_exist==True):
        Source_Exception=False

confidence_threshhold_input_Exception = True

while(confidence_threshhold_input_Exception==True):
    try:
        confidence_threshhold_input = float(input("\nPlease Enter the Desired Confidence Threshold between 0.01 & 1.00, For example Enter: 0.75\n"))
        if confidence_threshhold_input<0.01 or confidence_threshhold_input>1.00:
            raise ValueError
        else:
            confidence_threshhold_input_Exception=False
    except ValueError:
        print("\nError: Please enter a valid float number between 0.01 and 1.00")

Device_Exception = True

while(Device_Exception==True):
    try:
        device = input("Please enter the Device to Render with (gpu Highly Reccommended): 'cpu' or 'gpu'\n").lower()
        if (device=="cpu"):
            Device_Exception=False
            print("Test")
        elif (device=="gpu"):
            device="cuda:0"
            Device_Exception=False
        else:
            raise ValueError
    except ValueError:
        print("Error: Please enter only 'cpu' or 'gpu' for Device Type")

print("\nPress 'q' at any point to exit the Video Overlay\n")

def run(weights, source, confidence_threshhold_input, device, view_img=False, save_img=False, exist_ok=False):
    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Error: Source path '{source}' does not exist.")

    # Download YOLOv8 model
    yolov8_model_path = weights
    # Please Note, if you want to use different yolov8 model such as:
    # yolov8m or yolov8x, etc. you must change code below:
    download_yolov8l_model(yolov8_model_path)
    detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8',
                                                         model_path=yolov8_model_path,
                                                         confidence_threshold=confidence_threshhold_input,
                                                         device=device)
    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'mp4v')

    # Output setup
    save_dir = increment_path(Path('ultralytics_results_with_sahi') / 'exp', exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f'{Path(source).stem}.mp4'), fourcc, fps, (frame_width, frame_height))

    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            print("Unexpected Error:")
            break
        results = get_prediction(frame,detection_model,)
        object_prediction_list = results.object_prediction_list
        boxes_list = []
        clss_list = []
        for ind, _ in enumerate(object_prediction_list):
            boxes = object_prediction_list[ind].bbox.minx, object_prediction_list[ind].bbox.miny, \
                object_prediction_list[ind].bbox.maxx, object_prediction_list[ind].bbox.maxy
            clss = object_prediction_list[ind].category.name
            boxes_list.append(boxes)
            clss_list.append(clss)

        for box, cls in zip(boxes_list, clss_list):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2)
            label = str(cls)
            t_size = cv2.getTextSize(label, 0, fontScale=0.82, thickness=1)[0]
            cv2.rectangle(frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), (56, 56, 255),
                          -1)
            cv2.putText(frame,
                        label, (int(x1), int(y1) - 2),
                        0,
                        0.6, [255, 255, 255],
                        thickness=1,
                        lineType=cv2.LINE_AA)
        if view_img:
            cv2.imshow(Path(source).stem, frame)
        if save_img:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting Program Early:\n")
            break
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()
    print("Military Aircraft Detection Program Successfully Exited:\n")

# Example usage for Jupyter Notebook
# weights = "C:\\Users\\Tyler\\Desktop\\Orig_Data_Aircraft_Detection\\data\\runs\\detect\\train8\\weights\\best.pt"
# source = "C:\\Users\\Tyler\\Downloads\\Aircraft_Video_Test.mp4"

run(weights, source, confidence_threshhold_input, device, view_img=True, save_img=False, exist_ok=False)
