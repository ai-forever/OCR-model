import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super().__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):
    def __init__(self, number_class_symbols, in_channels=3, rnn_size=128):
        super().__init__()
        self.maxpooling_22 = nn.MaxPool2d(2, 2)
        self.maxpooling_21 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.layer1 = self._make_layer(in_channels, 64, 3, 1, 1)
        self.layer2 = self._make_layer(64, 128, 3, 1, 1)
        self.layer3 = self._make_layer(128, 256, 3, 1, 1)
        self.layer4 = self._make_layer(256, 256, 3, 1, 1)
        self.layer5 = self._make_layer(256, 512, 3, 1, 1)
        self.layer6 = self._make_layer(512, 512, 3, 1, 1)
        self.layer7 = self._make_layer(512, 512, 2, 1, 0)
        self.rnn = BidirectionalLSTM(512, rnn_size, number_class_symbols)

    def _make_layer(self, in_channels, out_channels,
                    kernel_size, stride, padding):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size, stride, padding))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpooling_22(x)
        x = self.layer2(x)
        x = self.maxpooling_22(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.maxpooling_21(x)
        x = self.layer5(x)
        x = self.maxpooling_21(x)
        x = self.layer6(x)
        x = self.maxpooling_21(x)

        b, c, h, w = x.size()
        assert h == 1, "The height of conv must be 1"
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)

        output = self.rnn(x)
        output = nn.functional.log_softmax(output, dim=2)
        return output
