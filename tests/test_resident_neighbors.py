import unittest
import numpy as np
from mes_vision.anomaly.resident import ResidentNeighbors
from mes_vision.anomaly.scoring import nearest_neighbors


class ResidentNeighborTests(unittest.TestCase):
    def test_models_release_reference_before_feature_model(self):
        from types import SimpleNamespace
        from mes_vision.operation.engine import Models
        from unittest.mock import Mock
        order=[]
        models=Models('unused',{})
        models.anomaly_engine=SimpleNamespace(close=lambda:order.append('reference'))
        models.features=SimpleNamespace(close=lambda:order.append('features'))
        models.backends=[SimpleNamespace(close=lambda:order.append('detector'))]
        models.inspectors=[Mock()]; models.detector=Mock()
        models.close(); models.close()
        self.assertEqual(order,['reference','detector','features'])
        self.assertIsNone(models.anomaly_engine); self.assertIsNone(models.features)
        self.assertEqual(models.inspectors,[])

    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def vectors(self, count, width=7):
        values=np.random.default_rng(count).normal(size=(count,width)).astype(np.float32)
        return values/np.linalg.norm(values,axis=-1,keepdims=True)

    def test_exact_results_across_chunks_and_tensor_inputs(self):
        import torch
        memory=self.vectors(13); query=self.vectors(9)
        for qc,bc in ((1,1),(4,5),(256,2048)):
            expected=nearest_neighbors(query,memory,query_chunk=qc,bank_chunk=bc)
            search=ResidentNeighbors(memory,device='cpu',query_chunk=qc,bank_chunk=bc)
            for values in (query,torch.from_numpy(query),torch.from_numpy(query.T.copy()).T):
                actual=search.search(values)
                for a,b in zip(expected,actual): np.testing.assert_array_equal(a,b)
            search.close()

    def test_ties_keep_first_reference_across_bank_chunks(self):
        memory=np.array([[1,0],[0,1],[1,0],[0,1]],np.float32)
        search=ResidentNeighbors(memory,device='cpu',bank_chunk=1)
        distance,index=search.search(memory)
        np.testing.assert_array_equal(distance,np.zeros(4))
        np.testing.assert_array_equal(index,[0,1,0,1])

    def test_reference_snapshot_is_owned_and_reused(self):
        memory=self.vectors(13); query=memory.copy(); before=memory.copy()
        search=ResidentNeighbors(memory,device='cpu'); pointer=search._memory.data_ptr()
        memory[:]=0
        for _ in range(3):
            distances,indices=search.search(query)
            np.testing.assert_array_equal(distances,np.zeros(13))
            np.testing.assert_array_equal(indices,np.arange(13))
            self.assertEqual(pointer,search._memory.data_ptr())
        self.assertEqual(search.resident_bytes,before.nbytes)

    def test_independent_banks_and_close(self):
        a=ResidentNeighbors(np.array([[1,0]],np.float32),device='cpu')
        b=ResidentNeighbors(np.array([[0,1]],np.float32),device='cpu')
        query=np.array([[1,0]],np.float32)
        self.assertEqual(a.search(query)[0][0],0)
        self.assertGreater(b.search(query)[0][0],1)
        a.close(); a.close(); self.assertEqual(a.resident_bytes,0)
        with self.assertRaisesRegex(ValueError,'closed'): a.search(query)
        self.assertGreater(b.search(query)[0][0],1)

    def test_invalid_features_and_options_rejected(self):
        import torch
        memory=self.vectors(4)
        for kw in ({'query_chunk':True},{'bank_chunk':0},{'device':'automatic'}):
            with self.assertRaises(ValueError): ResidentNeighbors(memory,**kw)
        search=ResidentNeighbors(memory,device='cpu')
        for query in (np.zeros_like(memory),np.full_like(memory,np.nan),memory.astype(np.float64),
                      memory[:0],memory[:,:2],torch.zeros((2,7)),torch.full((2,7),float('inf')),
                      torch.ones((2,7),dtype=torch.float64)):
            with self.assertRaises(ValueError): search.search(query)
        with self.assertRaises(ValueError): ResidentNeighbors(memory.reshape(2,2,7),device='cpu')


if __name__=='__main__': unittest.main()
