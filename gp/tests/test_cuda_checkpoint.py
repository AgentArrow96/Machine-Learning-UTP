"""Integration boundary: persisted optimiser + all RNG streams survive CUDA reload."""
import random
import numpy as np
import pytest
import torch

from floodnet.neural import rng_state, restore_rng, seed_all
from floodnet.training import save_checkpoint


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA checkpoint integration needs an NVIDIA GPU")
def test_cuda_optimiser_and_random_stream_resume(project):
    root, _ = project
    seed_all(42)
    generator = torch.Generator().manual_seed(83)
    model = torch.nn.Linear(3, 2).cuda()
    optimiser = torch.optim.AdamW(model.parameters(), lr=.001)
    loss = model(torch.randn(4, 3, device='cuda')).square().mean()
    loss.backward(); optimiser.step()
    path = root / 'checkpoint.pt'
    save_checkpoint(path, {'model':model.state_dict(),'optimiser':optimiser.state_dict(),'rng':rng_state(generator)})
    expected = (random.random(), np.random.random(), torch.rand(4), torch.rand(4,device='cuda'), torch.rand(4,generator=generator))
    saved = torch.load(path, map_location='cpu', weights_only=False)
    restored_model = torch.nn.Linear(3, 2).cuda()
    restored_model.load_state_dict(saved['model'])
    restored_optimiser = torch.optim.AdamW(restored_model.parameters(), lr=.001)
    restored_optimiser.load_state_dict(saved['optimiser'])
    restore_rng(saved['rng'], generator)
    actual = (random.random(), np.random.random(), torch.rand(4), torch.rand(4,device='cuda'), torch.rand(4,generator=generator))
    assert actual[0] == expected[0] and actual[1] == expected[1]
    for a, b in zip(actual[2:], expected[2:]):
        assert torch.equal(a,b)
    for old, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(old, restored)
    for state in restored_optimiser.state.values():
        assert state['exp_avg'].device.type == 'cuda'
        assert state['exp_avg_sq'].abs().sum() > 0
