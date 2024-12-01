import cv2
import torch
from ultralytics import YOLO
from baseline.model.DeepMAR import DeepMAR_ResNet50
from torchvision import transforms
import pickle
from PIL import Image
import time
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
from ZeroDCE.ZeroDCE_code.model import enhance_net_nopool

# Firebase Initialization
if not firebase_admin._apps:
    # Firebase Initialization
    cred = credentials.Certificate(r"xyz.json")  # Update the path
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://xyz.com/'  # Replace with your database URL
    })

ref = db.reference('violence_reports', url=database_url)  # Reference to the database

def send_to_firebase(victim_gender, suspect_gender, victim_attributes, suspect_attributes, confidence_score, timestamp):
    try:
        ref = db.reference('alerts')

        new_entry = ref.push({
            'victim': {
                'gender': victim_gender,
                'attributes': victim_attributes,
                'timestamp': timestamp
            },
            'suspect': {
                'gender': suspect_gender,
                'attributes': suspect_attributes,
                'confidence_score': confidence_score,
                'timestamp': timestamp
            }
        })
        print(f"Data sent to Firebase at: {timestamp} with key: {new_entry.key}")

    except Exception as e:
        print(f"Error sending data to Firebase: {e}")

# Configuration for DeepMAR models
class Config:
    def __init__(self):
        self.resize = (224, 224)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.deepmar_model_weight_file = 'ckpt_epoch25.pth'  # Update path if needed

        # Load attribute lists (Update these paths if needed)
        with open('X:/PETA/peta_dataset.pkl', 'rb') as f:
            peta_dataset = pickle.load(f)
        self.peta_attributes = [peta_dataset['att_name'][i] for i in peta_dataset['selected_attribute']]

        with open("X:/PA100K/Bpa100k_dataset.pkl", 'rb') as f:
            pa100k_dataset = pickle.load(f)
        self.pa100k_attributes = [pa100k_dataset['att_name'][i] for i in pa100k_dataset['selected_attribute']]

        self.max_num_att = max(len(self.peta_attributes), len(self.pa100k_attributes))
        self.model_kwargs = {'num_att': self.max_num_att, 'last_conv_stride': 2, 'drop_pool5': True,
                             'drop_pool5_rate': 0.5}

cfg = Config()

# Load YOLOv10 for Person Detection
yolo_person_model = YOLO("yolov9m.pt")

# Load YOLOv8 for Violence Detection
yolo_violence_model = YOLO('best.pt')  

# Load DeepMAR model
deepmar_model = DeepMAR_ResNet50(**cfg.model_kwargs).cuda()
model_ckpt = torch.load(cfg.deepmar_model_weight_file, map_location='cuda')
deepmar_model.load_state_dict(model_ckpt['state_dicts'][0])
deepmar_model.eval()

# Load Zero-DCE Model for Low-Light Enhancement
enhancement_model = enhance_net_nopool().cuda()
enhancement_model.load_state_dict(torch.load('ZeroDCE/ZeroDCE_code/snapshots/Epoch99.pth', map_location='cuda'))  # Update if needed
enhancement_model.eval()

# Preprocessing transforms for DeepMAR
normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
deepmar_transform = transforms.Compose([transforms.Resize(cfg.resize), transforms.ToTensor(), normalize])

# Video Capture
input_video_path = 'gym_test_final.mp4'  # Replace with your video path
output_video_path = 'inferencellie.mp4'
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Video Properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Proximity & Violence Detection
PROXIMITY_THRESHOLD = 150  # Distance in pixels
VIOLENCE_THRESHOLD_DISTANCE = 300  # 3 meters in pixels (adjust based on your camera calibration)
potential_victim_id = None
potential_suspect_id = None
prev_time = time.time()

BRIGHTNESS_THRESHOLD = 50  # Adjust this threshold as needed

def extract_attributes_from_prediction(attributes, dataset_attributes):
    """
    Extract the attributes from the prediction made by the DeepMAR model and store them in a list format.

    :param attributes: The raw attribute predictions from the model.
    :param dataset_attributes: The list of attribute names corresponding to the model's output.
    :return: A list of attribute names that are present.
    """
    present_attributes = []
    for i, attr_value in enumerate(attributes):
        if i < len(dataset_attributes):
            attribute_name = dataset_attributes[i]
            if attr_value > 0.5:  # Only add attributes that are present based on the threshold
                present_attributes.append(attribute_name)
    return present_attributes

# Main loop for video processing
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video or error reading the frame.")
        break

    # --- Low-Light Enhancement ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)

    cv2.putText(frame, f"ZERODCE BRIGHTNESS MONITORING : {brightness:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if brightness < BRIGHTNESS_THRESHOLD:
        data_lowlight = (np.asarray(frame_rgb) / 255.0)
        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_lowlight = data_lowlight.permute(2, 0, 1).cuda().unsqueeze(0)
        with torch.no_grad():
            _, enhanced_image, _ = enhancement_model(data_lowlight)
        enhanced_frame = enhanced_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        enhanced_frame = (enhanced_frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, "LLIE Applied", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        cv2.putText(frame, "BRIGHTNESS SATISFIED, LLIE NOT NEED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    # 1. Person Detection using YOLOv10
    results = yolo_person_model.predict(frame, conf=0.5)
    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            if cls == 0:  # Person detected
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size == 0:
                    continue

                # DeepMAR inference for victim (using PA100k attributes)
                person_roi_pil = Image.fromarray(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
                person_tensor = deepmar_transform(person_roi_pil).unsqueeze(0).cuda()
                with torch.no_grad():
                    attributes = deepmar_model(person_tensor)
                    attributes = torch.sigmoid(attributes).cpu().numpy()[0]

                # Extract attributes for the victim using PA100k attributes
                victim_attributes = extract_attributes_from_prediction(attributes, cfg.pa100k_attributes)
                print(f"Victim Attributes (PA100k): {victim_attributes}")  # This will print a list of present victim attributes

                # DeepMAR inference for suspect (using PETA attributes)
                person_roi_pil = Image.fromarray(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
                person_tensor = deepmar_transform(person_roi_pil).unsqueeze(0).cuda()
                with torch.no_grad():
                    attributes = deepmar_model(person_tensor)
                    attributes = torch.sigmoid(attributes).cpu().numpy()[0]

                # Extract attributes for the suspect using PETA attributes
                suspect_attributes = extract_attributes_from_prediction(attributes, cfg.peta_attributes)
                print(f"Suspect Attributes (PETA): {suspect_attributes}")  # This will print a list of present suspect attributes



                # Gender prediction using the PETA and PA100K dataset attributes
                peta_gender_index = 17  # Example index for female in PETA dataset
                pa100k_gender_index = 1  # Example index for female in PA100K dataset

                peta_gender = "Female" if attributes[peta_gender_index] > 0.5 else "Male" if peta_gender_index < len(attributes) else "Unknown"
                pa100k_gender = "Female" if attributes[pa100k_gender_index] > 0.5 else "Male" if pa100k_gender_index < len(attributes) else "Unknown"

                # Final gender decision
                gender = "Female" if peta_gender == "Female" or pa100k_gender == "Female" else "Male"

                # Store the detection with attributes
                detections.append(((x1, y1, x2, y2), conf, cls, gender, victim_attributes, suspect_attributes))

                # Display detected gender on the frame
                display_text = gender
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 2. Proximity Check and Violence Detection
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            bbox1, _, _, gender1, victim_attributes, _ = detections[i]
            bbox2, _, _, gender2, _, suspect_attributes = detections[j]

            # Calculate centroids
            centroid1 = (int((bbox1[0] + bbox1[2]) // 2), int((bbox1[1] + bbox1[3]) // 2))
            centroid2 = (int((bbox2[0] + bbox2[2]) // 2), int((bbox2[1] + bbox2[3]) // 2))
            distance = np.linalg.norm(np.array(centroid1) - np.array(centroid2))
            
            
            # Draw the line connecting the centroids
            cv2.line(frame, centroid1, centroid2, (0, 255, 255), 2)
            cv2.putText(frame, f"{distance:.1f}", ((centroid1[0] + centroid2[0]) // 2,
                                                    (centroid1[1] + centroid2[1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if (gender1 == "Male" and gender2 == "Female") or (gender1 == "Female" and gender2 == "Male"):
                if distance < PROXIMITY_THRESHOLD:
                    potential_victim_id = i if gender1 == "Female" else j
                    potential_suspect_id = j if gender1 == "Female" else i


                    # Violence Detection and Firebase update
                    roi = frame[min(bbox1[1], bbox2[1]):max(bbox1[3], bbox2[3]), 
                                        min(bbox1[0], bbox2[0]):max(bbox1[2], bbox2[2])]
                    if roi.size > 0:
                        violence_results = yolo_violence_model.predict(roi, conf=0.5)
                        for violence_result in violence_results:
                            if violence_result.boxes:
                                for box in violence_result.boxes:
                                    box_xyxy = box.xyxy[0].int().tolist()
                                    cv2.rectangle(roi, (box_xyxy[0], box_xyxy[1]), (box_xyxy[2], box_xyxy[3]), (255, 0, 0), 2)
                                    cv2.putText(roi, "Violence Detected!", (box_xyxy[0], box_xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                try:
                                    send_to_firebase(
                                        victim_gender=gender1,
                                        suspect_gender=gender2,
                                        victim_attributes=victim_attributes,
                                        suspect_attributes=suspect_attributes,
                                        confidence_score=conf,  # Adjust based on violence detection results
                                        timestamp=datetime.now().isoformat()
                                    )
                                except Exception as e:
                                    print(f"Error sending data to Firebase: {e}")

            # Draw bounding boxes for potential threats
            if potential_victim_id is not None and potential_suspect_id is not None:
                cv2.rectangle(frame, detections[potential_victim_id][0][:2], detections[potential_victim_id][0][2:], (0, 0, 255), 2)
                cv2.putText(frame, "Potential Victim", (detections[potential_victim_id][0][0], detections[potential_victim_id][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, detections[potential_suspect_id][0][:2], detections[potential_suspect_id][0][2:], (0, 0, 255), 2)
                cv2.putText(frame, "Potential Suspect", (detections[potential_suspect_id][0][0], detections[potential_suspect_id][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Write frame to output video
    out.write(frame)

    # Show frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()