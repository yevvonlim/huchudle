import os
import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np
import base64
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import imutils
import requests
from io import BytesIO
from PIL import Image

# Set basic settings
# image_path = "/workspace/image_landmark/sample_img/sample.jpeg"
filter_path = "/workspace/project-root/demo-backend/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(filter_path)



def get_initial_landmark(image_base64,point_landmark=False):
    # response = requests.get(image_base64)
    # image_base64 = base64.b64encode(response.content).decode('utf-8')
    # print("image_path",image_base64)
    
    
    # image processing
    try:
        decoded_image = base64.b64decode(image_base64)
        # print("decoded_image",decoded_image)
        image_stream = BytesIO(decoded_image)
        image = Image.open(image_stream)
    except:
        print("Failed to read the image")
        return None, None


    # image_data = base64.b64decode(image_base64)
    # image = Image.open(BytesIO(image_data))
    
    # Convert the image to OpenCV format
    if point_landmark:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image = np.array(image)
    image = imutils.resize(image, width=500)
    # image = cv2.imread(image_path)
    # image = imutils.resize(image, width=500)

    # Get image file name
    image_name = "base64_image.jpg"
    image_name_without_extension = os.path.splitext(image_name)[0]


    ## Detect faces in one image
    # rect contains the top-left corner
    rects = detector(image, 1)
    landmark_list=[]

    # For each faces, we will crop the face and add landmarks
    for (i, rect) in enumerate(rects):
        # Save cropped face
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        W = max(w,h) # to crop in square

        # Add padding
        p = int(W*0.4) 
        resized_x = x - p//2
        resized_y = y - p//2
        resized_W = W + p
        cropped_face = image[y:y + W, x:x + W]
        cropped_face_w_padding = image[resized_y:resized_y + resized_W, resized_x:resized_x + resized_W]

        ## Resize image into fixed size (256,256) and save the cropped face
        # resized_image = cv2.resize(cropped_face, (256, 256))
        # 이미지가 비어있지 않은지 확인
        image_shape = cropped_face_w_padding.shape
        # 이미지의 크기 출력
        print("Image size:", cropped_face_w_padding.shape)

        if len(image_shape) == 3 and 0 not in image_shape:
            # 이미지를 원하는 크기로 resize
            resized_image = cv2.resize(cropped_face_w_padding, (256, 256))
            # resized_image = cropped_face
        else:
            print(f"Failed to read the image {image_name}. Cannot do resize")
            break


        cv2.imwrite(f'/workspace/project-root/demo-backend/temp/{image_name_without_extension}_cropped_face_{i}.jpg', resized_image)
        encoded_image = base64.b64encode(cv2.imencode('.jpg', resized_image)[1]).decode()


        ## Add landmarks
        fixed_rect = dlib.rectangle(0, 0, W, W)
        shape = predictor(cropped_face, fixed_rect)
        shape = face_utils.shape_to_np(shape)

        # Resize the shape into (256,256)
        # resized_shape=shape*(256/W)
        landmark_list=[]

        for (a, b) in shape:
            # resized_a = (a+x-resized_x)*(256/W)
            # resized_b = (b+y-resized_y)*(256/W)
            resized_a = (a + p //2)*(256/resized_W)
            resized_b = (b + p //2)*(256/resized_W)
            landmark_list.append(([int(resized_a),int(resized_b)]))
            # cv2.circle(resized_image, (int(resized_a),int(resized_b)), 1, (0, 255, 0), 3)

        # resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'/workspace/project-root/demo-backend/temp/{image_name_without_extension}_cropped_face_with_landmark_{i}.jpg', resized_image)
        image_name = f"{image_name_without_extension}_cropped_face_{i}.jpg"
        print(landmark_list)
        # print(encoded_image)
    return encoded_image,landmark_list,resized_image


