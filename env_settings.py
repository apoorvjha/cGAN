import torch

if torch.cuda.is_available():
    DEVICE="cuda"
else:
    DEVICE="cpu"
batch_size=1
learning_rate=2e-5
num_workers=4
epochs=10

