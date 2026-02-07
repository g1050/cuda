import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(32, 128)

    def forward(self, x):
        x = self.linear(x)
        x = F.sigmoid(x)
        return x

model = Model()
model.eval()

# 示例输入：Linear(32, 128) 需要 shape (N, 32)
x = torch.randn(1, 32)

# output = model(x)
# print(output.shape)

# onnx_path = "model_file/model.onnx"
# torch.onnx.export(model, x, onnx_path, verbose=True, input_names=["input"], output_names=["output"])

torch_script_path = "model_file/model.pt"
traced_model = torch.jit.trace(model, x)
traced_model.save(torch_script_path)