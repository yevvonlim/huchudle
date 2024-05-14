import json

# 외부 JSON 파일 경로
json_file_path = "/workspace/image_landmark/landmark_output/landmark_coordinate.json"
# "image_id" 개수를 저장할 변수
image_id_count = 0

# JSON 파일 열기
with open(json_file_path, 'r') as json_file:
    # JSON 데이터 로드
    data = json.load(json_file)
    # "annotations" 키 아래에 있는 리스트 가져오기
    annotations = data["coordinate"]
    # 각 요소에서 "image_id" 개수 세기
    for item in annotations:
        if "image_id" in item:
            # "image_id"가 있는 경우 카운트 증가
            image_id_count += 1

# "image_id" 개수 출력
print("Total number of image_ids:", image_id_count)
