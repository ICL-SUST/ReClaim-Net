import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, input_channel, output_channel):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(output_channel))

    def forward(self, inp):
        return self.layers(inp)


class BackBone(nn.Module):

    def __init__(self, num_channel=64):
        super().__init__()

        self.adjust_input = nn.Sequential(
            nn.Conv2d(3, num_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel)
        )

        self.layers = nn.Sequential(
            ConvBlock(3, num_channel)
        )
        self.layers1_1 = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),

        )
        self.layers1_2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layers1_3 = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
        )
        self.layers1_4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layers2_1 = nn.Sequential(
            ConvBlock(num_channel, num_channel)
        )
        self.layers2_2 = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layers3_1 = nn.Sequential(
            ConvBlock(num_channel, num_channel)
        )
        self.layers3_2 = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
        )
        self.layers4_1 = nn.Sequential(
            ConvBlock(num_channel, num_channel)
        )
        self.layers4_2 = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
            # nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.relu = nn.LeakyReLU(0.2, True)
        self.relu1 = nn.LeakyReLU(0.2, True)
        self.relu2 = nn.LeakyReLU(0.2, True)
        self.relu3 = nn.LeakyReLU(0.2, True)
        self.relu4 = nn.LeakyReLU(0.2, True)
        self.relu5 = nn.LeakyReLU(0.2, True)
        self.maxpool = nn.MaxPool2d(2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(2)
        self.maxpool5 = nn.MaxPool2d(2)


        self.layers5_1 = nn.Sequential(
            ConvBlock(num_channel, num_channel)
        )
        self.layers5_2 = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channel * 5, num_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.layers6_1 = nn.Sequential(
            ConvBlock(num_channel, num_channel)
        )
        self.layers6_2 = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, True),
            # nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.block = nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=2),
            nn.Conv2d(4 * num_channel, num_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gn1 = nn.GroupNorm(32, num_channel)
        self.gn2 = nn.GroupNorm(32, num_channel)
        self.gn3 = nn.GroupNorm(32, num_channel)
        self.gn4 = nn.GroupNorm(32, num_channel)
        # self.gn5 = nn.GroupNorm(32, num_channel)


    def forward(self, inp):
        identity = self.adjust_input(inp)

        res2 =out = self.layers(inp)
        res3 = res1 = identity - out

        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layers1_1(out)
        res4 =res2 = self.block(res2) - out
        out = self.layers1_2(out)
        out = self.layers1_3(out)
        out = self.layers1_4(out)

        res1 = self.layers2_1(res1)
        res1 = self.relu1(res1)
        res1 = self.maxpool1(res1)
        res1 = self.layers2_2(res1)

        res2 = self.layers3_1(res2)
        res4 = res4 - res2
        res2 = self.relu2(res2)
        res2 = self.maxpool2(res2)
        res2 = self.layers3_2(res2)

        res3 = self.layers4_1(res3)
        res3 = self.relu3(res3)
        res3 = self.maxpool3(res3)
        res3 = self.layers4_2(res3)

        res4 = self.layers5_1(res4)
        res4 = self.relu4(res4)
        res4 = self.maxpool4(res4)
        res4 = self.layers5_2(res4)

        res1 = self.gn1(res1)
        res2 = self.gn2(res2)
        res3 = self.gn3(res3)
        res4 = self.gn4(res4)




        out = torch.cat((out,res1, res2,res3,res4), dim=1)

        out = self.conv_layers(out)
        return out