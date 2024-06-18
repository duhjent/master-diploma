import hashlib
# import s3
# import yaml
from torch.utils.data import Dataset
from zipfile import ZipFile
from os import path
import json
from PIL import Image
from torchvision.transforms import v2 as transforms
from torchvision import tv_tensors
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt

transform = transforms.Compose([
  transforms.ToImage(),
  transforms.Resize((400, 400)),
])

checksums = {
 'annotations.zip': '99394f7890112823880d14525c54467a',
 'test.zip': 'f111a735751470c098aca9bf4d721ccf',
 'train-0.zip': '982ea17dcb412f7fe57fa15a8cf91175',
 'train-1.zip': '008028e616f4bdd26cfcf802715f29eb',
 'train-2.zip': '48fd11f9bc1048b9ffa54a95605976b5',
 'val.zip': 'f1be4cb09ffcbd7c2850f7ac2ed2760f'
}

splits_files = {
  'train': ['train-0.zip', 'train-1.zip', 'train-2.zip'],
  'test': ['test.zip'],
  'val': ['val.zip']
}

class MTSDDataset(Dataset):
  def __init__(self, base_path='./data', split='train', transform=None, skip_validation=False):
    assert split in ['train', 'test', 'val']
    self.base_path = base_path
    self.split = split
    self.transform = transform if transform is not None else transforms.ToTensor()
    if not skip_validation:
      self.__check_files()
    self.__load_ids()
    self.__extract_files()

  def __check_files(self):
    files_to_check = ['annotations.zip'] + splits_files[self.split]
    for filename in files_to_check:
      checksum = checksums[filename]
      with open(f'{self.base_path}/{filename}', 'rb') as binary_file:
        data = binary_file.read()
        actual_checksum = hashlib.md5(data).hexdigest()
        if actual_checksum != checksum:
          raise Exception(f'File {filename} is incorrect!')
        
  def __load_ids(self):
    if not path.exists(f'{self.base_path}/annotations'):
      with ZipFile(f'{self.base_path}/annotations.zip', 'r') as annotations_zip:
        annotations_zip.extractall(f'{self.base_path}/annotations')
    with open(f'{self.base_path}/annotations/mtsd_v2_fully_annotated/splits/{self.split}.txt', 'r') as split_ids:
      self.ids = [line[:-1] for line in split_ids.readlines()]

  def __extract_files(self):
    self.folders = []
    for filename in splits_files[self.split]:
      folder_name = filename.split(".")[0]
      self.folders.append(folder_name)
      if not path.exists(f'{self.base_path}/{folder_name}'):
        with ZipFile(f'{self.base_path}/{filename}', 'r') as split_zip:
          split_zip.extractall(f'{self.base_path}/{folder_name}')

  def __getitem__(self, index) -> any:
    id = self.ids[index]
    annotation_path = f'{self.base_path}/annotations/mtsd_v2_fully_annotated/annotations/{id}.json'
    annotations = json.load(open(annotation_path, 'r'))
    for folder in self.folders:
      image_path = f'{self.base_path}/{folder}/images/{id}.jpg'
      if path.exists(image_path):
        break

    image = Image.open(image_path)

    bboxes = [[obj['bbox']['xmin'], obj['bbox']['ymin'], obj['bbox']['xmax'], obj['bbox']['ymax']] for obj in annotations['objects'] if obj['label'] != 'other-sign']
    bboxes = tv_tensors.BoundingBoxes(bboxes, format='XYXY', canvas_size=(image.height, image.width))
    label = {
      'boxes': bboxes,
      'labels': [obj['label'] for obj in annotations['objects']],
      'id': id,
    }

    return self.transform(image, label)

  def __len__(self):
    return len(self.ids)

def visualize(img, label):
  img_with_bboxes = draw_bounding_boxes(img, label['boxes'], width=1)

  plt.imshow(transforms.functional.to_pil_image(img_with_bboxes))
  plt.axis('off')
  plt.show()


ds = MTSDDataset(split='val', transform=transform, skip_validation=True)
visualize(*ds[1])
# print(len(ds))
