import torch

def pad_tensor(A, pad_to_len, fill_value, dim):
    padding_shape = list(A.size())
    padding_shape[dim] = pad_to_len - A.size(dim)
    padding = torch.ones(padding_shape, dtype=A.dtype)*fill_value
    return torch.cat([A,padding], dim=dim)

def to_device(batch, device):
    return dict((k, v.to(device) if isinstance(v,torch.Tensor) else v)
                for k,v in batch.items())
