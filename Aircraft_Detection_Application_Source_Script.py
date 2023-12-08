# %%
%matplotlib inline
# Tyler Boudreau
# Trained on Miitary Aircraft Detection Dataset: https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset
from pathlib import Path
import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
data_dir = Path("C:\\Users\\Tyler\\Desktop\\Orig_Data_Aircraft_Detection")

# read image dir
image_paths = []
annotation_paths = []
# collect image paths and annotations
for file_name in sorted(os.listdir(data_dir / 'dataset')):
    file_name = Path(file_name)
    if file_name.suffix == '.jpg':
        image_paths.append(data_dir / 'dataset' / file_name)
    if file_name.suffix == '.csv':
        annotation_paths.append(data_dir / 'dataset' / file_name)


# %%
exclude_classes = ['YF23', 'XB70', 'Vulcan']

class_names = [class_name for class_name in sorted(os.listdir(data_dir / 'crop')) if class_name not in exclude_classes]

class2idx = {class_name: i for i, class_name in enumerate(class_names)}

def filter_images_by_class(images_dir, exclude_classes):
    image_files = os.listdir(images_dir)
    filtered_images = [image for image in image_files if any(exclude_class not in image for exclude_class in exclude_classes)]
    return filtered_images

# %%
class2idx

# %%
def convert_bboxes_to_yolo_format(df: pd.DataFrame, class2idx: dict):
    df = df[df['class'].isin(class2idx.keys())]
    df['class'] = df['class'].apply(lambda x: class2idx[x]).values

    df['xmin'] = (df['xmin'] / df['width']).values
    df['ymin'] = (df['ymin'] / df['height']).values
    df['xmax'] = (df['xmax'] / df['width']).values
    df['ymax'] = (df['ymax'] / df['height']).values
    df['xc']   = (df['xmin'] + df['xmax']) / 2
    df['yc']   = (df['ymin'] + df['ymax']) / 2
    df['w']    = (df['xmax'] - df['xmin'])
    df['h']    = (df['ymax'] - df['ymin'])
    df.drop(
        ['filename', 'width', 'height', 'xmin', 'xmax', 'ymin', 'ymax'], 
        axis=1, 
        inplace=True
    )
    return df 

# %%
image_dir = data_dir / 'images'
label_dir = data_dir / 'labels'
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# %%
# create .txt annotations
for annotation_path in tqdm(annotation_paths):
    # get image_id
    image_id = annotation_path.parts[-1].split('.')[0]
    annotation_df = pd.read_csv(annotation_path)
    # transform to yolo format
    annotation_df = convert_bboxes_to_yolo_format(annotation_df, class2idx)
    # save to .txt resulting df
    with open(Path(label_dir) / f'{image_id}.txt', 'w') as f:
        f.write(annotation_df.to_string(header=False, index=False))

# %%
for image_path in image_paths:
    shutil.move(str(image_path), image_dir)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=1.3)

# %%
image_paths = [Path(image_dir) / image_path for image_path in sorted(os.listdir(image_dir))]
label_paths = [Path(label_dir) / label_path for label_path in sorted(os.listdir(label_dir))]

# %%
from sklearn.model_selection import train_test_split

# %%
image_paths = [f'images/{image_path}' for image_path in sorted(os.listdir(image_dir))]

# %%
train_size = 0.95

train_image_paths, val_image_paths = train_test_split(
    image_paths, train_size=train_size, random_state=3573, shuffle=True)

# %%
# make train split
with open(data_dir / 'train_split.txt', 'w') as f:
    f.writelines(f'./{image_path}\n' for image_path in train_image_paths)

# make val split
with open(data_dir / 'val_split.txt', 'w') as f:
    f.writelines(f'./{image_path}\n' for image_path in val_image_paths)

# %%
data_yaml = "C:\\Users\\Tyler\\Desktop\\Orig_Data_Aircraft_Detection\\data\\MilitaryAircraft.yaml"

# %%
from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("C:\\Users\\Tyler\\Downloads\\yolov8l.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="C:\\Users\\Tyler\\Desktop\\Orig_Data_Aircraft_Detection\\data\\MilitaryAircraft.yaml", epochs=350,lr0=0.00001,lrf=0.00001,optimizer='SGD')  # train the model

#path = model.export(format="onnx")  # export the model to ONNX format

# %%
from ultralytics import YOLO
model = YOLO("C:\\Users\\Tyler\\AppData\\Local\\Programs\\Microsoft VS Code\\runs\\detect\\train8\\weights\\best.pt")


# %%
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_prediction
from pathlib import Path
from ultralytics.utils.files import increment_path
from sahi.utils.yolov8 import download_yolov8l_model

def run(weights, source, view_img=False, save_img=False, exist_ok=False):
    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Download YOLOv8 model
    yolov8_model_path = weights
    download_yolov8l_model(yolov8_model_path)
    detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8',
                                                         model_path=yolov8_model_path,
                                                         confidence_threshold=0.80,
                                                         device='cuda:0')
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
            break

        results = get_prediction(frame,
                                        detection_model,)
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
            t_size = cv2.getTextSize(label, 0, fontScale=0.8, thickness=1)[0]
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
            break
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


# Example usage in Jupyter Notebook
weights = "C:\\Users\\Tyler\\Desktop\\Orig_Data_Aircraft_Detection\\data\\runs\\detect\\train8\\weights\\best.pt"
source = "C:\\Users\\Tyler\\Downloads\\Aircraft_Video_Test.mp4"
run(weights, source, view_img=True, save_img=False, exist_ok=False)



