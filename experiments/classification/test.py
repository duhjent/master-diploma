import json
import numpy as np

def main():
    with open('./data/dfg/train.bak.json', 'r') as annot_file:
        annotations = json.load(annot_file)

    cnts = [0] * 200
    total = 0

    for annot in annotations['annotations']:
        if annot['area'] == 1:
            continue
        cnts[int(annot['category_id'])] += 1
        total += 1

    print(np.array(total / (np.array(cnts) * 200)))


if __name__ == "__main__":
    main()
