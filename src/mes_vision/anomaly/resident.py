"""Engine-owned reference snapshot; exact FP32 chunked L2, no approximate search."""
import numpy as np

from mes_vision.training.data import require
from .bank import checked_features


class ResidentNeighbors:
    def __init__(self, memory, *, device='cuda', query_chunk=256, bank_chunk=2048):
        import torch
        checked_features(memory)
        require(memory.ndim == 2, 'distance feature dimensions mismatch')
        require(type(query_chunk) is int and 1 <= query_chunk <= 1024
                and type(bank_chunk) is int and 1 <= bank_chunk <= 8192, 'invalid distance chunk size')
        require(device in {'cpu', 'cuda'} and (device != 'cuda' or torch.cuda.is_available()),
                'requested distance device unavailable')
        self.query_chunk, self.bank_chunk = query_chunk, bank_chunk
        # An owned copy also isolates CPU tensors from later caller array mutations.
        with torch.inference_mode():
            self._memory = torch.from_numpy(np.array(memory, copy=True, order='C')).to(device)
        self.device = self._memory.device

    @property
    def resident_bytes(self):
        return 0 if self._memory is None else self._memory.numel() * self._memory.element_size()

    def search(self, query):
        import torch
        require(self._memory is not None, 'normal reference search is closed')
        if isinstance(query, np.ndarray):
            checked_features(query)
            query = torch.from_numpy(np.array(query, copy=True, order='C')).to(self.device)
        require(isinstance(query, torch.Tensor) and query.dtype == torch.float32
                and query.ndim == 2 and min(query.shape) > 0
                and query.shape[1] == self._memory.shape[1], 'distance feature dimensions mismatch')
        require(query.device == self.device, 'query and reference devices differ')
        with torch.inference_mode():
            require(torch.isfinite(query).all().item(), 'nonfinite features')
            norms = torch.linalg.vector_norm(query, dim=-1)
            require(torch.isclose(norms, torch.ones_like(norms), atol=2e-4, rtol=1e-5).all().item(),
                    'features must be L2 normalized')
            query = query.contiguous()
            distances = torch.empty(len(query), dtype=torch.float32, device=self.device)
            indices = torch.empty(len(query), dtype=torch.long, device=self.device)
            for start in range(0, len(query), self.query_chunk):
                q = query[start:start+self.query_chunk]
                best = torch.full((len(q),), float('inf'), device=self.device)
                closest = torch.zeros(len(q), dtype=torch.long, device=self.device)
                for bank_start in range(0, len(self._memory), self.bank_chunk):
                    memory = self._memory[bank_start:bank_start+self.bank_chunk]
                    scores = torch.cdist(q, memory, p=2, compute_mode='donot_use_mm_for_euclid_dist')
                    minimum, local_index = scores.min(dim=1)
                    better = minimum < best
                    closest = torch.where(better, local_index + bank_start, closest)
                    best = torch.minimum(best, minimum)
                distances[start:start+len(q)] = best
                indices[start:start+len(q)] = closest
            # Only compact scores/indices leave the device, once per object.
            result, nearest = distances.cpu().numpy(), indices.cpu().numpy()
        require(np.isfinite(result).all() and (result >= 0).all() and (result <= 2.001).all(),
                'invalid nearest-neighbor distances')
        return np.minimum(result, 2), nearest

    def close(self):
        self._memory = None
