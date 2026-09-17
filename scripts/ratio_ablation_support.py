"""Matched 256 draws/epoch. 0% versus 25% CAD with the same optimizer budget."""
import torch
from torch.utils.data import DataLoader,Sampler
class RatioBatches(Sampler):
    def __init__(self,real,synthetic,weights,mode):
        self.real=real;self.synthetic=synthetic;self.weights=torch.tensor(weights,dtype=torch.double);self.mode=mode;self.epoch=0
    def __len__(self):return 128
    def __iter__(self):
        g=torch.Generator().manual_seed(742+self.epoch);self.epoch+=1
        real=[]
        while len(real)<256:real.extend(self.real[i] for i in torch.randperm(len(self.real),generator=g).tolist())
        if self.mode=='mixed25':
            synthetic=[self.synthetic[i] for i in torch.multinomial(self.weights,64,replacement=True,generator=g).tolist()]
            # Every 4 samples: 3 real, 1 synthetic. Every optimizer step is also 75/25.
            flat=[v for i in range(64) for v in [*real[i*3:i*3+3],synthetic[i]]]
        else:flat=real[:256]
        for i in range(0,256,2):yield flat[i:i+2]
def install_ratio(dm,meta,coco,mode):
    def loader():
        ds=dm._dataset_train;lookup={im['file_name']:item for im,item in zip(coco['images'],meta['items'])};real=[];synthetic=[];weights=[]
        for i,image_id in enumerate(ds.ids):
            item=lookup[ds.coco.imgs[image_id]['file_name']]
            if item['domain']=='real':real.append(i)
            else:synthetic.append(i);weights.append(item['sampling_weight'])
        return DataLoader(ds,batch_sampler=RatioBatches(real,synthetic,weights,mode),collate_fn=dm._collate_fn,num_workers=0,pin_memory=True)
    dm.train_dataloader=loader
def self_test():
    for mode,expected in [('real',0),('mixed25',64)]:
        s=RatioBatches(list(range(55)),list(range(55,980)),[1]*925,mode)
        a=list(s);b=list(s);assert a!=b and len(a)==128
        assert sum(i>=55 for batch in a for i in batch)==expected
        for start in range(0,128,4):assert sum(i>=55 for batch in a[start:start+4] for i in batch)==(2 if expected else 0)
    print('PASS: equal 256 draws,128 batches,32 optimizer steps; exact real-only or 75/25')
if __name__=='__main__':self_test()
