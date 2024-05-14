
# /workspace/image_landmark/caption_json/caption_annotations.json
# /workspace/image_landmark/caption_json/huchudle-train-wo-text-caption.json
# /workspace/image_landmark/landmark_output/huchudle-train-ann-wo-text.json
import json

# 외부 JSON 파일 경로
json_file_path = "/workspace/image_landmark/caption_json/huchudle-train-wo-text-caption.json"

# JSON 파일 열기
with open(json_file_path, 'r') as json_file:
    # JSON 데이터 로드
    json_data = json.load(json_file)
    # JSON 데이터의 리스트의 길이 구하기
    list_length = len(json_data)
    # 리스트의 길이 출력
    print("리스트의 길이:", list_length)
