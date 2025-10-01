#Libs
import cv2
import numpy as np
from ultralytics import YOLO


#Detection

def detect_plate(image,plate_detction_model):
    image_arr = np.array(image)
    model=YOLO(plate_detction_model)
    results=model(image_arr)[0]
    cropped_images=[]
    isdetect= len(results.boxes.data.tolist())
    if isdetect is not 0:
        threshold=0.5
        for result in results.boxes.data.tolist():
            if result[4]>threshold:
                x1=int(result[0])
                y1=int(result[1])
                x2=int(result[2])
                y2=int(result[3])
                score=result[4]
                class_id=int(result[5])
                cropped_img=image_arr[y1:y2,x1:x2]
                cropped_images.append(cropped_img)
                #cv2.rectangle(image_arr,(x1,y1),(x2,y2),(0,255,0),2)
                class_name=model.names[class_id]
                score*=100
                text=f" %{score:.2f}"
                #cv2.putText(image_arr, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),2)
    else:
        text="No Plate Detected"
        cv2.putText(image_arr, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), cv2.LINE_AA)
    return image_arr,cropped_images

def plate_read(plate_imgs, plate_read_model):
    plate_texts = []
    model = YOLO(plate_read_model)
    
    all_results = model(plate_imgs)
    
    for results in all_results: 
        plate_text_location = [] 
        for result in results.boxes.data.tolist():
            x1 = int(result[0])
            class_id = int(result[5])
            char = model.names[class_id]
            plate_text_location.append((char, x1))

        if not plate_text_location:
            plaka_metni = "Karakter bulunamadı"
        else:
            plate_text_location.sort(key=lambda item: item[1])
            
            plaka_metni = "".join([item[0] for item in plate_text_location])
            
        plate_texts.append(plaka_metni)

    return plate_texts