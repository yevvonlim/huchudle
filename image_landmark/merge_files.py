import json

# JSON 파일 읽기 및 수정
def process_data(data, source_prefix):
    if source_prefix==1:
        for item in data["coordinate"]:
            item['image_id'] = f"1_celeba_hq_cropped_img/{str(item['image_id'])}"
    elif source_prefix==0:
        for item in data["coordinate"]:
            item['image_id'] = f"0_output_cropped_img/{str(item['image_id'])}"
    elif source_prefix==2:
        for item in data["coordinate"]:
            item['image_id'] = f"2_output_cropped_img/{str(item['image_id'])}"
    elif source_prefix==3:
        for item in data["coordinate"]:
            item['image_id'] = f"3_output_cropped_img/{str(item['image_id'])}"

    return data



with open('/workspace/image_landmark/landmark_output/0_output_landmark.json') as f1, open('/workspace/image_landmark/landmark_output/1_celeba_hq_landmark.json') as f2, open('/workspace/image_landmark/landmark_output/2_output_landmark.json') as f3, open("/workspace/image_landmark/landmark_output/3_output_landmark.json") as f4:
    data1 = json.load(f1)
    data2 = json.load(f2)
    data3 = json.load(f3)
    data4 = json.load(f4)

data1_processed = process_data(data1, 0)
data2_processed = process_data(data2, 1)
data3_processed = process_data(data3, 2)
data4_processed = process_data(data4, 3)

merged_data = {
    "coordinate": []
}

# 합친 데이터를 리스트로 만들어 새로운 JSON 파일로 저장
merged_data["coordinate"] = data1_processed["coordinate"] + data2_processed["coordinate"] + data3_processed["coordinate"] +data4_processed["coordinate"]

with open('/workspace/image_landmark/landmark_output/landmark_coordinate.json', 'w') as outfile:
    json.dump(merged_data, outfile)



##########
[{}, {}, ...]
yj_ann = []
dy_ann = []
new_ann = []

yj_ann_sorted, dy_ann_sorted = map(lambda x: sorted(x, key=lambda x: x['image_id']), [yj_ann, dy_ann])
for yj_ann, dy_ann in zip(yj_ann_sorted, dy_ann_sorted):
    if yj_ann['image_id'] == dy_ann['image_id']:
        new_ann.append({
            "image_id": yj_ann['image_id'],
            "landmark": dy_ann['landmark'],
            "caption": yj_ann['caption']
        })
    else:
        print('error')