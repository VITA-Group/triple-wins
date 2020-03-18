import torch
import torch.nn as nn
from collections import OrderedDict


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)



class SmallCNN(nn.Module):
    def __init__(self, drop=0.5):
        super(SmallCNN, self).__init__()

        self.num_channels = 1
        self.num_labels = 10

         
        self.conv1 = nn.Conv2d(self.num_channels, 32, 3)
        self.relu = nn.ReLU(True)
        


        self.conv2 = nn.Conv2d(32, 32, 3)
        self.maxpool =  nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(64 * 4 * 4, 200)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, self.num_labels)

        self.branch_layer1 = nn.Sequential(nn.Conv2d(32, 16, 3, stride=2),
                                           nn.MaxPool2d(2, 2),
                                           View(-1, 16 * 6 * 6),
                                           nn.Linear(16 * 6 * 6, 200),
                                           nn.Dropout(drop),
                                           nn.Linear(200, 200),
                                           nn.Linear(200, self.num_labels))

        self.branch_layer2 = nn.Sequential(nn.MaxPool2d(2, 2),
                                           View(-1, 64 * 5 * 5),
                                           nn.Linear(64 * 5 * 5, 200),
                                           nn.Dropout(drop),
                                           nn.Linear(200, 200),
                                           nn.Linear(200, self.num_labels))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.fc3.weight, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, input):
        output_branch = []
        out = self.conv1(input)
        out = self.relu(out)
        output_branch.append(self.branch_layer1(out))
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = self.relu(out)
        output_branch.append(self.branch_layer2(out))
        out = self.conv4(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(-1, 64 * 4 * 4)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        output_branch.append(out)

        return output_branch




def mnist_smallcnn(pretrained=False):
    model = SmallCNN()
    return model   


def main():
    x = torch.rand(2, 1, 28, 28)
    model = mnist_smallcnn() 
    a = model(x)


if __name__ == "__main__":
    main()
