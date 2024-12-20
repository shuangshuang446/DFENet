import time
import torch
import warnings
warnings.filterwarnings("ignore")
from model.DSCM import DSCM
from model.SSFR import SSFR
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        conv1 = nn.Conv2d(1, 64, (8, 1), (2, 1), padding=(1, 0))
        mask1 = DSCM(64, 64, output_size=(98, 3), mask_spatial_granularity=(10, 1))

        conv2 = nn.Conv2d(64, 128, (8, 1), (2, 1), padding=(1, 0))
        mask2 = DSCM(128, 128, output_size=(47, 3), mask_spatial_granularity=(5, 1))

        conv3 = nn.Conv2d(128, 256, (8, 1), (2, 1), padding=(1, 0))

        self.conv_module = nn.Sequential(
            conv1,
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            mask1,
            SSFR(64),

            conv2,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            mask2,
            SSFR(128),

            conv3,
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 7))
        self.classifier = nn.Linear(16128, 6)

    def forward(self, x):
        out = self.conv_module(x)
        # out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class DeepConvLSTM(torch.nn.Module):
    def __init__(self, image_channels=1, n_classes=6):
        super(DeepConvLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 64, (8, 1), (2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, (8, 1), (2, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, (8, 1), (2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

        )
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        # self.gru_ = nn.GRU(6912, 128)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        cnn_x = self.cnn(x)
        # print(cnn_x.shape)
        cnn_x = cnn_x.reshape([-1, 21 * 3, 256])
        # lstm_x, (h_n, c_n) = self.lstm(cnn_x)
        lstm_x, (h_n, c_n) = self.lstm(cnn_x)
        # # print(lstm_x.shape)  ([1, 57, 128])
        # print((lstm_x[:, -1, :]).shape)
        out = self.fc(lstm_x[:, -1, :])

        return out

# model = CNN()
model = DeepConvLSTM()
input_data = torch.randn(1, 1, 200, 3)


def test_inference_time(model, input_data, device, num_iterations=100):
    model.to(device)
    model.eval()
    input_data = input_data.to(device)

    with torch.no_grad():
        model(input_data)

    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(input_data)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations
    return avg_time


# CPU
cpu_device = torch.device("cpu")
cpu_inference_time = test_inference_time(model, input_data, cpu_device)
print(f"CPU Inference Time: {cpu_inference_time:.6f} seconds")

# GPU
if torch.cuda.is_available():
    gpu_device = torch.device("cuda:0")
    gpu_inference_time = test_inference_time(model, input_data, gpu_device)
    print(f"GPU Inference Time: {gpu_inference_time:.6f} seconds")
else:
    print("CUDA is not available.")


# Deployment for mobile devices
dummy_input = torch.randn(1, 1, 200, 3).cuda()
torch.onnx.export(model, dummy_input, "./onnx/lstm.onnx", export_params=True, opset_version=11, do_constant_folding=True,
input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
