from action_value_model import ActionValueModel as ActionNN
import coremltools as ct
import torch

torch_model = torch.load("./_out/8x8/model_1500r_500g_i1.pt", map_location=torch.device('cpu'))
torch_model.eval()

for log_batch_size in range(4):
    sample = torch.rand(2 ** log_batch_size, 2, 8, 8)

    traced_model = torch.jit.trace(torch_model, sample)

    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=sample.shape)]
    )

    coreml_model.save(f'./_out/8x8/coreml_model_i1_{2 ** log_batch_size}.mlmodel')