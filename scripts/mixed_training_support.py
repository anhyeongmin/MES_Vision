"""Each batch has one real and one synthetic sample; 128 batches/epoch."""
import torch
from torch.utils.data import DataLoader,Sampler

class MixedBatches(Sampler):
    def __init__(self,real,synthetic,weights):
        self.real=real;self.synthetic=synthetic;self.weights=torch.tensor(weights,dtype=torch.double);self.epoch=0
        assert len(real)>0 and len(synthetic)==len(weights)>0 and (self.weights>0).all()
    def __len__(self):return 128
    def __iter__(self):
        g=torch.Generator().manual_seed(742+ self.epoch);self.epoch+=1
        ri=[]
        while len(ri)<128:ri.extend(torch.randperm(len(self.real),generator=g).tolist())
        si=torch.multinomial(self.weights,128,replacement=True,generator=g).tolist()
        for a,b in zip(ri[:128],si):yield [self.real[a],self.synthetic[b]]

def install_mixing(dm,meta,coco):
    def loader():
        ds=dm._dataset_train;real=[];syn=[];weights=[]
        by_file={im['file_name']:item for im,item in zip(coco['images'],meta['items'])}
        for i,image_id in enumerate(ds.ids):
            item=by_file[ds.coco.imgs[image_id]['file_name']]
            if item['domain']=='real':real.append(i)
            else:syn.append(i);weights.append(item['sampling_weight'])
        sampler=MixedBatches(real,syn,weights)
        return DataLoader(ds,batch_sampler=sampler,collate_fn=dm._collate_fn,num_workers=0,pin_memory=True)
    dm.train_dataloader=loader

def self_test():
    s=MixedBatches(list(range(55)),list(range(55,982)),[1]*927)
    a=list(s);b=list(s)
    assert len(a)==128 and a!=b
    assert all(0<=r<55 and 55<=c<982 for r,c in a+b)
    print('Mixed batches verified: 128 real + 128 synthetic samples per epoch')
if __name__=='__main__':self_test()
