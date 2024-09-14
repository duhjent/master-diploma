import json

with open('./data/dfg/train.json', 'r') as annotations_file:
    annotations = json.load(annotations_file)

print(len(annotations['images']))

with open('dump.json', 'r') as dump_file:
    dump = json.load(dump_file)

for obj in dump:
    annotations['images'] = [img for img in annotations['images'] if img['id'] != obj['id']]

print(len(annotations['images']))

with open('./data/dfg/train-new.json', 'w') as annotations_file_new:
    json.dump(annotations, annotations_file_new)
