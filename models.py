import math

import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F


def get_model(name="vgg16", use_cuda=True, num_dataclasses=10):
    global model
    if name == "ResNet18ClientNetwork":
        model = ResNet18ClientNetwork()
    elif name=="ResNet18ServerNetwork":
        #model=ResNet18ServerNetwork(Baseblock,[2,2,2],num_dataclasses)
        model = ResNet18ServerNetwork(pretrained=True)
    elif name == "ResNet34ClientNetwork":
        model = ResNet34ClientNetwork()
    elif name == "ResNet34ServerNetwork":
        model = ResNet34ServerNetwork(Baseblock, [3, 4, 6, 3], num_dataclasses)
    elif name == "LeNetClientNetwork":
        model = LeNetClientNetwork()
    elif name == "LeNetServerNetwork":
        model = LeNetServerNetwork()
    elif name == "GeneratorNetwork":
        model = GeneratorNetwork()
    elif name == "LeNetComplete":
        model = LeNetComplete()
    elif name == "AlexNetClientNetwork":
        model = AlexNetClientNetwork()
    elif name == "AlexNetServerNetwork":
        model = AlexNetServerNetwork()
    elif name == "ResNet18_PFSL_front":
        model = ResNet18_PFSL_front(pretrained=True)
    elif name == "ResNet18_PFSL_center":
        model = ResNet18_PFSL_center(pretrained=True)
    elif name == "ResNet18_PFSL_back":
        model = ResNet18_PFSL_back(pretrained=True)
    elif name == "LeNet_PFSL_front":
        model = LeNet_PFSL_front()
    elif name == "LeNet_PFSL_center":
        model = LeNet_PFSL_center()
    elif name == "LeNet_PFSL_back":
        model = LeNet_PFSL_back()

    if torch.cuda.is_available() and use_cuda:
        return model.to('cuda')
    else:
        return model


class LeNetComplete(nn.Module):
    """CNN based on the classical LeNet architecture, but with ReLU instead of
    tanh activation functions and max pooling instead of subsampling."""
    def __init__(self):
        super(LeNetComplete, self).__init__()
        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Second block - convolutional
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Third block - fully connected
        self.block3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10))

    def forward(self, x):
        """Define forward pass of CNN
        Args:
            x: Input Tensor
        Returns:
          x: Output Tensor
        """
        # Apply first convolutional block to input tensor
        x = x.cuda()
        x = self.block1(x)
        # Apply second convolutional block to input tensor
        x = self.block2(x)
        # Flatten outputss
        x = x.view(-1, 4*4*16)
        # Apply first fully-connected block to input tensor
        x = self.block3(x)
        return F.log_softmax(x, dim=1)

class GeneratorNetwork(nn.Module):
    """only use for MNIST."""
    def __init__(self):
        super(GeneratorNetwork, self).__init__()

        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Linear(784, 256),
            # 该函数相比于ReLU，保留了一些负轴的值，缓解了激活值过小而导致神经元参数无法更新的问题，其中α\alphaα默认0.01。
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            # 将值映射到0~1
            nn.Sigmoid()
            )

    def forward(self, x):
        """Defines forward pass of CNN until the split layer, which is the first
        convolutional layer

        Args:
            x: Input Tensor

        Returns:
          x: Output Tensor
        """
        # Apply first convolutional block to input tensor
        x = x.cuda()
        x = self.block1(x)

        return x


class LeNetClientNetwork(nn.Module):
    """CNN following the architecture of:
    https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a- \
            fashion-clothes-dataset-e589682df0c5

    The ClientNetwork is used for Split Learning and implements the CNN
    until the first convolutional layer."""
    def __init__(self):
        super(LeNetClientNetwork, self).__init__()

        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        """Defines forward pass of CNN until the split layer, which is the first
        convolutional layer

        Args:
            x: Input Tensor

        Returns:
          x: Output Tensor
        """
        # Apply first convolutional block to input tensor
        x = x.cuda()
        x = self.block1(x)

        return x


class LeNetServerNetwork(nn.Module):
    """CNN following the architecture of:
    https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a- \
            fashion-clothes-dataset-e589682df0c5

    The ServerNetwork is used for Split Learning and implements the CNN
    from the split layer until the last."""
    def __init__(self):
        super(LeNetServerNetwork, self).__init__()

        # Second block - convolutional
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Third block - fully connected
        self.block3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10))

    def forward(self, x):
        """Defines forward pass of CNN from the split layer until the last

        Args:
            x: Input Tensor

        Returns:
          x: Output Tensor
        """
        # Apply second convolutional block to input tensor
        x = self.block2(x)

        # Flatten output
        #x = x.view(-1, 4*4*16)
        x = x.view(x.size(0), -1)

        # Apply fully-connected block to input tensor
        x = self.block3(x)

        return x


class ResNet18ClientNetwork_test(nn.Module):
    def __init__(self):
        super(ResNet18ClientNetwork_test, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = x.cuda()
        resudial1 = F.relu(self.layer1(x))
        out1 = self.layer2(resudial1)
        out1 = out1 + resudial1  # adding the resudial inputs -- downsampling not required in this layer
        resudial2 = F.relu(out1)
        return resudial2


class ResNet18ClientNetwork(nn.Module):
    def __init__(self, input_channels=3, pretrained=True):
        super(ResNet18ClientNetwork, self).__init__()
        model = models.resnet18(pretrained=pretrained)
        model_children = list(model.children())
        self.input_channels = input_channels
        if self.input_channels == 1:
            self.conv_channel_change = nn.Conv2d(1, 3, 3, 1, 2)  # to keep the image size same as input image size to this conv layer
        self.front_model = nn.Sequential(*model_children[:4])

        if pretrained:
            layer_iterator = iter(self.front_model)
            for i in range(4 - 0):
                layer = layer_iterator.__next__()
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = x.cuda()
        if self.input_channels == 1:
            x = self.conv_channel_change(x)
        x = self.front_model(x)
        return x

class ResNet18_PFSL_front(nn.Module):
    def __init__(self, input_channels=3, pretrained=False):
        super(ResNet18_PFSL_front, self).__init__()
        model = models.resnet18(pretrained=pretrained)
        model_children = list(model.children())
        self.input_channels = input_channels
        if self.input_channels == 1:
            self.conv_channel_change = nn.Conv2d(1, 3, 3, 1, 2)  # to keep the image size same as input image size to this conv layer
        self.front_model = nn.Sequential(*model_children[:4])

        if pretrained:
            layer_iterator = iter(self.front_model)
            for i in range(4 - 0):
                layer = layer_iterator.__next__()
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = x.cuda()
        if self.input_channels == 1:
            x = self.conv_channel_change(x)
        x = self.front_model(x)
        return x


class ResNet18_PFSL_center(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet18_PFSL_center, self).__init__()
        model = models.resnet18(pretrained=pretrained)
        model_children = list(model.children())
        global center_model_length
        center_model_length = len(model_children) - 4 - 3
        self.center_model = nn.Sequential(*model_children[4:center_model_length + 4])
        if pretrained:
            layer_iterator = iter(self.center_model)
            for i in range(center_model_length - 1):
                layer = layer_iterator.__next__()
                for param in layer.parameters():
                    param.requires_grad = False

    def freeze(self, epoch, pretrained=False):
        num_unfrozen_center_layers = 0
        if pretrained:
            layer_iterator = iter(self.center_model)
            for i in range(center_model_length - num_unfrozen_center_layers):
                layer = layer_iterator.__next__()
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.center_model(x)
        return x

class ResNet18_PFSL_back(nn.Module):
    def __init__(self, pretrained=False, output_dim=10):
        super(ResNet18_PFSL_back, self).__init__()
        model = models.resnet18(pretrained=pretrained)
        model_children = list(model.children())
        model_length = len(model_children)

        fc_layer = nn.Linear(512, output_dim)
        model_children = model_children[:-1] + [nn.Flatten()] + [fc_layer]
        self.back_model = nn.Sequential(*model_children[model_length-3:])

        if pretrained:
            layer_iterator = iter(self.back_model)
            for i in range(3-4):
                layer = layer_iterator.__next__()
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.back_model(x)
        return x

class AlexNetClientNetwork(nn.Module):
    def __init__(self):
        super(AlexNetClientNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = x.cuda()
        x = self.features(x)
        return x
class AlexNetServerNetwork(nn.Module):
    def __init__(self, class_num=10):
        super(AlexNetServerNetwork, self).__init__()

        # self.f2= nn.Sequential(

        # )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_num),
        )

    def forward(self, x):
        # x = self.f2(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


# Model at server side
# TODO:what's the function of the block
class Baseblock(nn.Module):
    expansion = 1

    def __init__(self, input_planes, planes, stride=1, dim_change=None):
        super(Baseblock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, stride=stride, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride=1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change

    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))

        if self.dim_change is not None:
            res = self.dim_change(res)

        output += res
        output = F.relu(output)

        return output


class ResNet18ServerNetwork(nn.Module):
    def __init__(self, pretrained=True, output_dim=10):
        super(ResNet18ServerNetwork, self).__init__()
        # 加载基础模型
        base_model = models.resnet18(pretrained=pretrained)
        model_children = list(base_model.children())

        # 中心部分配置
        center_model_length = len(model_children) - 4 - 3  # 保持原计算方式
        self.center = nn.Sequential(*model_children[4:4 + center_model_length])

        # 尾部部分配置
        tail_layers = model_children[len(model_children) - 3:-1]  # 获取原始尾部层
        tail_layers += [nn.Flatten(),
                        nn.Linear(512, output_dim)]  # 替换全连接层
        self.back = nn.Sequential(*tail_layers)

        # 参数冻结逻辑
        if pretrained:
            # 冻结中心部分
            for layer in list(self.center.children())[:-1]:  # 保持原冻结策略
                for param in layer.parameters():
                    param.requires_grad = False

            # 冻结尾部部分（原back的冻结逻辑）
            for layer in list(self.back.children())[:-2]:  # 保持原冻结策略
                for param in layer.parameters():
                    param.requires_grad = False

    def freeze(self, epoch, pretrained=False):
        # 保持原冻结方法逻辑
        if pretrained:
            num_unfrozen = 0
            for layer in list(self.center.children())[:-num_unfrozen - 1]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.center(x)
        x = self.back(x)
        return x

class ResNet18ServerNetwork_test(nn.Module):
    def __init__(self, block, num_layers, num_dataclasses):
        super(ResNet18ServerNetwork_test, self).__init__()
        self.input_planes = 64
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

        self.layer4 = self._layer(block, 128, num_layers[0], stride=2)
        self.layer5 = self._layer(block, 256, num_layers[1], stride=2)
        self.layer6 = self._layer(block, 512, num_layers[2], stride=2)
        self.averagePool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_dataclasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _layer(self, block, planes, num_layers, stride=2):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(
                nn.Conv2d(self.input_planes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride=stride, dim_change=dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion

        return nn.Sequential(*netLayers)

    def forward(self, x):
        out2 = self.layer3(x)
        out2 = out2 + x  # adding the resudial inputs -- downsampling not required in this layer
        x3 = F.relu(out2)

        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)

        x7 = F.avg_pool2d(x6, 1)
        x8 = x7.view(x7.size(0), -1)
        y_hat = self.fc(x8)

        return y_hat


class ResNet34ClientNetwork(nn.Module):
    def __init__(self):
        super(ResNet34ClientNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = x.cuda()
        resudial1 = F.relu(self.layer1(x))
        out1 = self.layer2(resudial1)
        out1 = out1 + resudial1  # adding the residual inputs -- downsampling not required in this layer
        resudial2 = F.relu(out1)
        return resudial2
class ResNet34ServerNetwork(nn.Module):
    def __init__(self, block, num_layers, num_dataclasses):
        super(ResNet34ServerNetwork, self).__init__()
        self.input_planes = 64
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

        # ResNet34 has more layers, so we need to adjust the number of blocks in each layer
        self.layer4 = self._layer(block, 128, num_layers[0], stride=2)
        self.layer5 = self._layer(block, 256, num_layers[1], stride=2)
        self.layer6 = self._layer(block, 512, num_layers[2], stride=2)
        self.layer7 = self._layer(block, 512, num_layers[3], stride=2)  # additional layer for ResNet34
        self.averagePool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_dataclasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _layer(self, block, planes, num_layers, stride=2):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(
                nn.Conv2d(self.input_planes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride=stride, dim_change=dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion

        return nn.Sequential(*netLayers)

    def forward(self, x):
        out2 = self.layer3(x)
        out2 = out2 + x  # adding the residual inputs -- downsampling not required in this layer
        x3 = F.relu(out2)

        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)  # additional layer for ResNet34

        x8 = F.avg_pool2d(x7, 1)
        x9 = x8.view(x8.size(0), -1)
        y_hat = self.fc(x9)

        return y_hat
    

class LeNet_PFSL_front(nn.Module):
    def __init__(self):
        super(LeNet_PFSL_front, self).__init__()
        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = x.cuda()
        x = self.block1(x)
        return x


class LeNet_PFSL_center(nn.Module):
    def __init__(self):
        super(LeNet_PFSL_center, self).__init__()
        # Second block - convolutional
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.block2(x)
        return x


class LeNet_PFSL_back(nn.Module):
    def __init__(self):
        super(LeNet_PFSL_back, self).__init__()
        # Third block - fully connected
        self.block3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        # Flatten output
        x = x.view(x.size(0), -1)
        x = self.block3(x)
        return x