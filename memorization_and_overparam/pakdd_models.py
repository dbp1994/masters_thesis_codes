import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn.init as init


class CIFAR_13_CNN(nn.Module):
    """
    Reference:
    https://github.com/CuriousAI/mean-teacher/blob/master/tensorflow/mean_teacher/model.py
    """
    def __init__(self):
        super(CIFAR_13_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=1)

        self.linear = nn.Linear(128, 10)

        self.pool1 = nn.MaxPool2d((2,2))
        self.pool2 = nn.MaxPool2d((2,2))
        self.pool3 = nn.AvgPool2d((6,6))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.pool1(F.leaky_relu(self.conv3(x), negative_slope=0.1))
        x = F.dropout2d(x, p=0.5)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv5(x), negative_slope=0.1)
        x = self.pool2(F.leaky_relu(self.conv6(x), negative_slope=0.1))
        x = F.dropout2d(x, p=0.5)
        x = F.leaky_relu(self.conv7(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv8(x), negative_slope=0.1)
        x = self.pool3(F.leaky_relu(self.conv9(x), negative_slope=0.1))
        x = self.linear(x)
        return x

class CIFAR_MobileNet(nn.Module):
    
    def __init__(self):
        super(CIFAR_MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn1   = nn.BatchNorm2d(num_features = 128)

        self.comb2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=128, bias=True, padding_mode='zeros')
        self.bn2   = nn.BatchNorm2d(num_features = 128)
        self.comb3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, groups=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=128, bias=True, padding_mode='zeros')
        self.bn3   = nn.BatchNorm2d(num_features = 128)
        self.comb4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, groups=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=128, bias=True, padding_mode='zeros')
        self.bn4   = nn.BatchNorm2d(num_features = 128)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.comb5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, groups=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=256, bias=True, padding_mode='zeros')
        self.bn5   = nn.BatchNorm2d(num_features = 256)
        self.comb6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, groups=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=256, bias=True, padding_mode='zeros')
        self.bn6   = nn.BatchNorm2d(num_features = 256)
        self.comb7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, groups=1, bias=False)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=256, bias=True, padding_mode='zeros')
        self.bn7   = nn.BatchNorm2d(num_features = 256)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=10, kernel_size=1, stride=1, groups=1, bias=True)
        self.gap   = nn.AvgPool2d(kernel_size=16)
    
    def forward(self, input):
        # 3*3 conv layer + BatchNorm + ReLU
        out_ = F.relu(self.bn1(self.conv1(input)))

        # 1*1 conv layer to form combinations of features maps
        # separate 3*3 convolutions for each feature map
        # BatchNorm + ReLU + max pooling
        out_ = self.comb2(out_)
        out_ = self.conv2(out_)
        out_ = self.bn2(out_)
        out_ = F.relu(out_)
        
        # 1*1 conv layer to form combinations of feature maps
        # separate 3*3 convolutions for each feature map
        # ReLU
        out_ = self.comb3(out_)
        out_ = self.conv3(out_)
        out_ = self.bn3(out_)
        out_ = F.relu(out_)

        # 1*1 conv layer to form combinations of feature maps
        # separate 3*3 convolutions for each feature map
        # ReLU + MaxPool
        out_ = self.comb4(out_)
        out_ = self.conv4(out_)
        out_ = self.bn4(out_)
        out_ = F.relu(out_)
        out_ = self.pool4(out_)
        
        # 1*1 conv layer to form combinations of feature maps
        # separate 3*3 convolutions for each feature map
        # ReLU
        out_ = self.comb5(out_)
        out_ = self.conv5(out_)
        out_ = self.bn5(out_)
        out_ = F.relu(out_)
        
        # 1*1 conv layer to form combinations of feature maps
        # separate 3*3 convolutions for each feature map
        # ReLU
        out_ = self.comb6(out_)
        out_ = self.conv6(out_)
        out_ = self.bn6(out_)
        out_ = F.relu(out_)
        
        # 1*1 conv layer to form combinations of feature maps
        # separate 3*3 convolutions for each feature map
        # ReLU
        out_ = self.comb7(out_)
        out_ = self.conv7(out_)
        out_ = self.bn7(out_)
        out_ = F.relu(out_)

        # 1*1 conv layer to get 10 feature maps
        # GAP
        out_ = self.conv8(out_)
        out_ = self.gap(out_)
        return out_.view(-1, 10)


class cnn_conv_block(nn.Module):
    def __init__(self, in_planes, n3x3, n_out, pad_3x3=0, batch_norm=True):
        super(cnn_conv_block, self).__init__()

        if batch_norm:
            self.b1  = nn.Sequential(
                nn.Conv2d(in_planes, n3x3, kernel_size=3, padding=pad_3x3, stride=1),
                nn.BatchNorm2d(n3x3), 
                nn.ReLU(), 
                nn.Conv2d(n3x3, n_out, kernel_size=3, padding=pad_3x3, stride=1),
                nn.BatchNorm2d(n_out), 
                nn.ReLU(), 
                nn.MaxPool2d(2, stride=2)
            )
        else:
            self.b1  = nn.Sequential(
                nn.Conv2d(in_planes, n3x3, kernel_size=3, padding=pad_3x3, stride=1),
                nn.ReLU(), 
                nn.Conv2d(n3x3, n_out, kernel_size=3, padding=pad_3x3, stride=1),
                nn.ReLU(), 
                nn.MaxPool2d(2, stride=2)
            )
    
    def forward(self, x):
        y1 = self.b1(x)
        return y1


class CIFAR_CNN(nn.Module):
    def __init__(self, num_classes, batch_norm=True):
        super(CIFAR_CNN, self).__init__()

        self.num_classes = num_classes
        self.batch_norm = batch_norm

        self.d1 = cnn_conv_block(3, 64, 64, pad_3x3=1, batch_norm=self.batch_norm)
        self.d2 = cnn_conv_block(64, 128, 128, pad_3x3=1, batch_norm=self.batch_norm)
        self.d3 = cnn_conv_block(128, 196, 16, pad_3x3=1, batch_norm=self.batch_norm)
        self.d4 = nn.Linear(256, self.num_classes)

    def forward(self, x):
      x = self.d1(x)
      x = self.d2(x)
      x = self.d3(x)
      x = x.view(x.shape[0], -1)
      x = self.d4(x)
      return x


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class Inception_small(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3, pad_1x1=0, pad_3x3=0, batch_norm=True):
        super(Inception_small, self).__init__()

        if not batch_norm:
            # in_planes -> 1x1 conv branch
            self.b1 = nn.Sequential(
                nn.Conv2d(in_planes, n1x1, kernel_size=1, padding=pad_1x1, stride=1),
                nn.ReLU(),
            )

            # 1x1 conv -> 3x3 conv branch
            self.b2 = nn.Sequential(
                nn.Conv2d(in_planes, n3x3, kernel_size=3, padding=pad_3x3, stride=1),
                nn.ReLU(),
            )
        else:
            # in_planes -> 1x1 conv branch
            self.b1 = nn.Sequential(
                nn.Conv2d(in_planes, n1x1, kernel_size=1, padding=pad_1x1, stride=1),
                nn.BatchNorm2d(n1x1),
                nn.ReLU(),
            )

            # 1x1 conv -> 3x3 conv branch
            self.b2 = nn.Sequential(
                nn.Conv2d(in_planes, n3x3, kernel_size=3, padding=pad_3x3, stride=1),
                nn.BatchNorm2d(n3x3),
                nn.ReLU(),
            )
            

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        return torch.cat([y1,y2], 1)

class Downsample(nn.Module):
    def __init__(self, in_planes, n3x3, pad_3x3=0, pad_pool=0, batch_norm=True):
        super(Downsample, self).__init__()
        
        # in_planes -> 3x3 conv branch
        if not batch_norm:
            self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3, kernel_size=3, padding=pad_3x3, stride=2),
            nn.ReLU(),
            )
        else:
            self.b1 = nn.Sequential(
                nn.Conv2d(in_planes, n3x3, kernel_size=3, padding=pad_3x3, stride=2),
                nn.BatchNorm2d(n3x3),
                nn.ReLU(),
            )

        # in_planes -> Max Pool branch
        self.b2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=pad_pool, stride=2),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        return torch.cat([y1,y2], 1)

class CIFAR_Inception_Small(nn.Module):
    def __init__(self, num_classes, batch_norm):
        super(CIFAR_Inception_Small, self).__init__()

        self.batch_norm = batch_norm

        if not self.batch_norm:
            self.pre_layers = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=3, stride=1),
                nn.ReLU(),
            )
        else:
            self.pre_layers = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=3, stride=1),
                nn.BatchNorm2d(96),
                nn.ReLU(),
            )

        self.num_classes = num_classes

        self.a1 = Inception_small(96, 32, 32, 0, 1, self.batch_norm)
        self.a2 = Inception_small(64, 32, 48, 0, 1, self.batch_norm)

        self.d1 = Downsample(80, 80, 0, 0, self.batch_norm)

        self.a3 = Inception_small(160, 112, 48, 0, 1, self.batch_norm)
        self.a4 = Inception_small(160, 96, 64, 0, 1, self.batch_norm)
        self.a5 = Inception_small(160, 80, 80, 0, 1, self.batch_norm)
        self.a6 = Inception_small(160, 48, 96, 0, 1, self.batch_norm)

        self.d2 = Downsample(144, 96, 0, 0, self.batch_norm)

        self.a7 = Inception_small(240, 176, 160, 0, 1, self.batch_norm)
        self.a8 = Inception_small(336, 176, 160, 0, 1, self.batch_norm)

        self.avgpool = nn.AvgPool2d(7, padding=1, stride=1)
        self.linear = nn.Linear(1344, self.num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a1(out)
        out = self.a2(out)
        out = self.d1(out)
        out = self.a3(out)
        out = self.a4(out)
        out = self.a5(out)
        out = self.a6(out)
        out = self.d2(out)
        out = self.a7(out)
        out = self.a8(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class alex_conv_block(nn.Module):
    def __init__(self, in_planes, n5x5, pad_5x5, pad_pool, batch_norm=True):
        super(alex_conv_block, self).__init__()

        # in_planes -> 5x5 conv branch
        if not batch_norm:
            self.b1 = nn.Sequential(
                nn.Conv2d(in_planes, n5x5, kernel_size=5, padding=pad_5x5, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, padding=pad_pool, stride=3),
            )
        else:
            self.b1 = nn.Sequential(
                nn.Conv2d(in_planes, n5x5, kernel_size=5, padding=pad_5x5, stride=1),
                nn.BatchNorm2d(n5x5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, padding=pad_pool, stride=3),
            )

    def forward(self, x):
        y1 = self.b1(x)
        return y1

class CIFAR_AlexCNN_Small(nn.Module):
    def __init__(self, num_classes, batch_norm=True):
        super(CIFAR_AlexCNN_Small, self).__init__()
        
        self.num_classes = num_classes
        self.batch_norm = batch_norm

        self.a1 = alex_conv_block(3, 200, pad_5x5=0, pad_pool=0, batch_norm=self.batch_norm)
        self.a2 = alex_conv_block(200, 200, pad_5x5=0, pad_pool=0, batch_norm=self.batch_norm)
        self.a3 = nn.Linear(200, 384)
        self.a4 = nn.BatchNorm1d(384)
        self.a5 = nn.Linear(384, 192)
        self.a6 = nn.BatchNorm1d(192)
        self.a7 = nn.Linear(192, self.num_classes)

    def forward(self, x):
        x = self.a1(x)
        x = self.a2(x)
        x = x.view(x.size(0), -1)
        x = self.a3(x)
        x = F.relu(self.a4(x))
        x = self.a5(x)
        x = F.relu(self.a6(x))
        x = self.a7(x)
        return x

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_RN32(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_RN32, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class CIFAR_ResNet32(nn.Module):
    def __init__(self, num_classes, block=BasicBlock_RN32, num_blocks=[5, 5, 5]):
        super(CIFAR_ResNet32, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, kernel_size=(out.shape[3], out.shape[3]))
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out


class CIFAR_ResNet34(ResNet):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(CIFAR_ResNet34, self).__init__(BasicBlock, [3,4,6,3], num_classes=self.num_classes)


class CIFAR_MLP_1x(nn.Module):
    def __init__(self, inp, num_classes):
        super(CIFAR_MLP_1x, self).__init__()

        self.num_classes = num_classes
        self.inp = inp

        self.fc1 = nn.Linear(self.inp, 512)
        self.fc2 = nn.Linear(512, self.num_classes)

        self.bn1 = nn.BatchNorm1d(512)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.fc2(x)
        return x

class CIFAR_MLP_3x(nn.Module):
    def __init__(self, inp, num_classes):
        super(CIFAR_MLP_3x, self).__init__()

        self.num_classes = num_classes
        self.inp = inp

        self.fc1 = nn.Linear(self.inp, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, self.num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.fc3(x)
        x = F.relu(self.bn3(x))
        x = self.fc4(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.num_classes = num_classes

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# def test():
#     net = CIFAR_Inception_Small(10)
#     x = torch.randn(1,3,32,32)
#     y = net(x)
#     print(y.size())