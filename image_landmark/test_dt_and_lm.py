import os
import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# Set basic settings
# image_path = "/workspace/image_landmark/sample_img/sample.jpeg"
filter_path = "/workspace/image_landmark/dlib-landmarks-predictor/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(filter_path)


def dt_and_lm(image):
    # image processing
    # image_path = "/workspace/spandjp/img/og_img/IMG_0087.jpeg"
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=500)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get image file name
    image_name = os.path.basename(image_path)
    image_name_without_extension = os.path.splitext(image_name)[0]


    ## Detect faces in one image
    # rect contains the top-left corner
    rects = detector(image, 1)
    total_landmark_list = []
    total_img_names = []
    print("rects",rects)
    # For each faces, we will crop the face and add landmarks
    for (i, rect) in enumerate(rects):
        print("i",i)
        print("rect",rect)
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


        cv2.imwrite(f'/workspace/spandjp/img/cropped_img/{image_name_without_extension}_cropped_face_{i}.jpg', resized_image)


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
            cv2.circle(resized_image, (int(resized_a),int(resized_b)), 1, (0, 255, 0), 3)


        cv2.imwrite(f'/workspace/spandjp/img/landmark_img/{image_name_without_extension}_cropped_face_with_landmark_{i}.jpg', resized_image)
        image_name = f"{image_name_without_extension}_cropped_face_{i}.jpg"
        total_landmark_list.append(landmark_list)
        total_img_names.append(image_name)
        # print(landmark_list)
        print(image_name)

    return total_img_names, total_landmark_list
    



# img directory
image_dir = '/workspace/spandjp/img/og_img'
landmark_json ={
            "coordinate":[

            ]
        }


total_list = []
# tqdm을 사용하여 진행 상황 출력
for filename in tqdm(os.listdir(image_dir)):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        try:
            image_name_list,landmark_list = dt_and_lm(image_path)
            # image_name, landmark_list = dt_and_lm(image_path) 
        except TypeError:  # None을 반환하는 경우에 대한 예외 처리
            continue
        
        for img_name, landmark_list in zip(image_name_list, landmark_list):
            block = {
                "image_id": str(img_name),
                "landmark": landmark_list
            }
            total_list.append(block)

# # JSON 파일에 저장
with open('/workspace/spandjp/img/landmark.json', 'w') as json_file:
    json.dump(total_list, json_file, indent=4)
