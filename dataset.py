import s3
import yaml
from torch.utils.data import Dataset

schecksums = {
 'annotations': '99394f7890112823880d14525c54467a',
 'test': 'f111a735751470c098aca9bf4d721ccf',
 'train-0': '982ea17dcb412f7fe57fa15a8cf91175',
 'train-1': '008028e616f4bdd26cfcf802715f29eb',
 'train-2': '48fd11f9bc1048b9ffa54a95605976b5',
 'val': 'f1be4cb09ffcbd7c2850f7ac2ed2760f'
}


with open('s3.yaml', 'r') as f:
  s3_config = yaml.load(f)

class MTSDDataset(Dataset):
  def __init__(self, base_path='./data'):
    self.base_path = base_path

  def __getitem__(self, index) -> any:
    return super().__getitem__(index)


