import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image

# class Load_data(Dataset):
#     def __init__(self, root):
#         imgs = []
#         for i in os.listdir(root):  #遍历整个文件夹
#             path = os.path.join(root, i)
#             if os.path.splitext(path)[1]==".png":
#                 imgs.append(i)
#         for i in filelist:
#             print(i)

class TrackDataset(Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        self.test = test
        imgs = []
        for i in os.listdir(root):  #遍历整个文件夹
            path = os.path.join(root, i)
            if os.path.splitext(path)[1]==".png":
                imgs.append(i)

        # imgs = [os.path.join(root, img) for img in os.listdir(root)]
        # if self.test:
        #     imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('\\')[-1]))
        #     # 对测试集的数据进行排序
        # else:
        #     imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        #     # 对非测试集的数据进行排序
        #     # 排序的目的是便于后续的分割

        imgs_num = len(imgs)  # 获取数据的长度便于切分数据集
        print(f"num of imgs:{imgs_num}")

        if self.test:
            self.imgs = imgs  # 将测试集的数据直接导入
        elif train:
            self.imgs = imgs[:int(0.1 * imgs_num)]  # 将train中数据的70%给train
            print(self.imgs)
        else:
            self.imgs = imgs[int(0.1 * imgs_num):]  # 剩下的30%做验证集

        if transforms is None:  # 对数据进行增强处理
            normalize = T.Normalize(mean=[0.488, 0.455, 0.417],
                                    std=[0.261, 0.255, 0.257])

            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(28),
                    T.CenterCrop(28),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(28),
                    T.CenterCrop(28),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __len__(self):  # 返回整个数据集的大小
        return len(self.imgs)

    def __getitem__(self, index):  # 根据索引index返回图像及标签
        img_path = self.imgs[index]
        # if self.test:
        #     label = int(self.imgs[index].split('.')[-2].split('\\')[-1])
        #     # 获取测试集文件名的部分作为标签
        # else:
        #     label = 1 if 'dog' in img_path.split('\\')[-1] else 0
        #     # 获取train中文件名中的标签并进行数字化，dog为1，cat为0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data


root_path = "/mnt/data/luoyan/road/track/08"
load_data = TrackDataset(root_path)
train_data = TrackDataset(root_path, train=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
print(train_loader)


# train_data = MyDataset(/data/train, train=True)
# val_data = MyDataset(/data/train, train=False)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=True)

