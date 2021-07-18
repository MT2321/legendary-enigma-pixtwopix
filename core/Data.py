import numpy as np
from PIL import Image
import os
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from typing import TypeVar
T_co = TypeVar('T_co', covariant=True)


class CityScapesDataset(Dataset):
    def __init__(self, root_dir):
        # super().__init__()
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.transform = transforms.Compose(
                        [transforms.ToTensor(), 
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index) -> T_co:
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)

        image = np.array(Image.open(img_path))
        image_width = image.shape[1]//2

        input_image = image[:, :image_width, :]
        target_image = image[:, image_width:, :]

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        return input_image, target_image

    def __len__(self):
        return len(self.list_files)


def test():
    print("Hola")
    dataset = CityScapesDataset('./data/City/train')
    loader = DataLoader(dataset, batch_size=1)
    import matplotlib.pyplot as plt
    for x, y in loader:
        save_image(x*0.5+0.5, 'prueba.png')
        print(x.size())
        print(type(x))
        x = x.permute(0, 2, 3, 1)*0.5 + 0.5
        plt.imshow(x[0])
        plt.show()
        y = y.permute(0, 2, 3, 1)*0.5 + 0.5
        plt.imshow(y[0])
        plt.show()
        break


if __name__ == '__main__':
    test()
