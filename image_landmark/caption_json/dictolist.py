import json

json_file_path = "/workspace/image_landmark/caption_json/caption_annotations.json"
new_data = []
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)
    annotations = data["annotations"]
    new_data = annotations

new_json_file_path = "/workspace/image_landmark/caption_json/huchudle-train-wo-text-caption.json"

with open(new_json_file_path, 'w') as new_json_file:
    json.dump(new_data, new_json_file, indent=4)
