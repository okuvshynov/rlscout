from action_model import ActionNN
import coremltools as ct
import torch

torch_model = torch.load("./_out/model_cp_1.pt", map_location=torch.device('cpu'))
torch_model.eval()

for log_batch_size in range(10):
    sample = torch.rand(2 ** log_batch_size, 2, 7, 7)

    traced_model = torch.jit.trace(torch_model, sample)

    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=sample.shape)]
    )

    coreml_model.save(f'./_out/coreml_model_cp_{2 ** log_batch_size}.mlmodel')