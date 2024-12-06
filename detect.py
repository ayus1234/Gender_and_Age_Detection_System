#A Gender and Age Detection program 

import cv2
import argparse
import os
import time

def check_file_exists(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def process_with_rate_limit(net, blob, operation_name=""):
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            net.setInput(blob)
            predictions = net.forward()
            return predictions
        except Exception as e:
            if "resource_exhausted" in str(e).lower():
                if attempt < max_retries - 1:
                    print(f"Rate limit hit for {operation_name}. Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Max retries reached for {operation_name}. Please try again later.")
                    raise
            else:
                raise

parser=argparse.ArgumentParser()
parser.add_argument('--image', help='Path to image file (optional)')

args=parser.parse_args()

# Check if image file exists when provided
if args.image and not os.path.isfile(args.image):
    print(f"Error: Image file '{args.image}' not found")
    exit(1)

# Model files
faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

# Check if all model files exist
required_files = [faceProto, faceModel, ageProto, ageModel, genderProto, genderModel]
try:
    for file in required_files:
        check_file_exists(file)
except FileNotFoundError as e:
    print(f"Error: {str(e)}")
    print("Please ensure all model files are in the current directory")
    exit(1)

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
# Original age ranges
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# Middle values for each age range
ageMiddleValues=[1, 5, 10, 17, 28, 40, 50, 70]
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

try:
    video=cv2.VideoCapture(args.image if args.image else 0)
    if not video.isOpened():
        print("Error: Could not open video source")
        exit(1)

    padding=20
    frame_delay = 0.03  # 30ms delay for smooth video
    frame_count = 0
    # Variables to store last detection
    last_detection = None
    last_face_box = None
    
    if args.image:
        while True:
            hasFrame,frame=video.read()
            if not hasFrame:
                break
            
            resultImg,faceBoxes=highlightFace(faceNet,frame)
            if not faceBoxes:
                print("No face detected")
                continue

            for faceBox in faceBoxes:
                face=frame[max(0,faceBox[1]-padding):
                        min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                        :min(faceBox[2]+padding, frame.shape[1]-1)]

                if face.size == 0:
                    continue

                blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                
                try:
                    # Gender detection with rate limiting
                    genderPreds = process_with_rate_limit(genderNet, blob, "gender detection")
                    gender=genderList[genderPreds[0].argmax()]
                    print(f'Gender: {gender}')

                    # Age detection with rate limiting
                    agePreds = process_with_rate_limit(ageNet, blob, "age detection")
                    ageIndex = agePreds[0].argmax()
                    exactAge = ageMiddleValues[ageIndex]
                    ageRange = ageList[ageIndex]
                    print(f'Estimated Age: {exactAge} years (Range: {ageRange})')

                    cv2.putText(resultImg, f'{gender}, {exactAge} years', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                except Exception as e:
                    if "resource_exhausted" in str(e).lower():
                        print("Resource limit reached. Please try again in about an hour.")
                        cv2.putText(resultImg, "Rate limit reached", (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                    else:
                        print(f"Error during detection: {str(e)}")
                        cv2.putText(resultImg, "Error in detection", (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

                cv2.imshow("Detecting age and gender", resultImg)
            
            # Break only if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # For image mode, wait for a key press before closing
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
    else:
        # Webcam mode
        print("Starting webcam mode...")
        print("Press 'q' to quit")
        while True:
            hasFrame,frame=video.read()
            if not hasFrame:
                break
            
            frame_count += 1
            # Process every 10th frame for detection
            if frame_count % 10 == 0:
                resultImg,faceBoxes=highlightFace(faceNet,frame)
                if faceBoxes:
                    for faceBox in faceBoxes:
                        face=frame[max(0,faceBox[1]-padding):
                                min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                                :min(faceBox[2]+padding, frame.shape[1]-1)]

                        if face.size == 0:
                            continue

                        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                        
                        try:
                            # Gender detection with rate limiting
                            genderPreds = process_with_rate_limit(genderNet, blob, "gender detection")
                            gender=genderList[genderPreds[0].argmax()]

                            # Age detection with rate limiting
                            agePreds = process_with_rate_limit(ageNet, blob, "age detection")
                            ageIndex = agePreds[0].argmax()
                            exactAge = ageMiddleValues[ageIndex]
                            ageRange = ageList[ageIndex]
                            
                            # Store the results and face box for continuous display
                            last_detection = f'{gender}, {exactAge} years'
                            last_face_box = faceBox
                            print(f'Detected: {last_detection} (Range: {ageRange})')
                            
                        except Exception as e:
                            if "resource_exhausted" in str(e).lower():
                                print("Resource limit reached. Please try again in about an hour.")
                                last_detection = "Rate limit reached"
                            else:
                                print(f"Error during detection: {str(e)}")
                                last_detection = "Error in detection"
                            last_face_box = faceBox
            
            # Always display the last valid detection
            if last_detection and last_face_box:
                cv2.putText(frame, last_detection, 
                          (last_face_box[0], last_face_box[1]-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, 
                            (last_face_box[0], last_face_box[1]),
                            (last_face_box[2], last_face_box[3]),
                            (0,255,0), 2)
            
            # Show the frame
            cv2.imshow("Detecting age and gender", frame)
            
            # Small delay to prevent high CPU usage
            time.sleep(frame_delay)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

finally:
    video.release()
    cv2.destroyAllWindows()
