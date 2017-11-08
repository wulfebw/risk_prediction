
import numpy as np

from utils import compute_n_batches, compute_batch_idxs

class DomainAdaptationDataset(object):
    
    def __init__(
            self, 
            xs,ys,ws,xt,yt,wt,
            batch_size=100,
            shuffle=True):
        self.xs = xs
        self.ys = ys
        self.ws = ws
        self.xt = xt
        self.yt = yt
        self.wt = wt
        self.batch_size = batch_size
        self.shuffle = shuffle
          
        self.n_s_samples = len(xs)
        self.n_t_samples = len(xt)
        # each batch will be balanced between source and target
        # so choose the larger of the too as the number of samples
        self.half_batch_size = batch_size // 2
        self.n_batches = compute_n_batches(max(self.n_s_samples, self.n_t_samples), self.half_batch_size)
        
    def _shuffle(self):
        if self.shuffle:
            s_idxs = np.random.permutation(self.n_s_samples)
            self.xs = self.xs[s_idxs]
            self.ys = self.ys[s_idxs]
            t_idxs = np.random.permutation(self.n_t_samples)
            self.xt = self.xt[t_idxs]
            self.yt = self.yt[t_idxs]
        
    def batches(self):
        self._shuffle()
        for bidx in range(self.n_batches):
            s_idxs = compute_batch_idxs(bidx * self.half_batch_size, self.half_batch_size, self.n_s_samples)
            t_idxs = compute_batch_idxs(bidx * self.half_batch_size, self.half_batch_size, self.n_t_samples)
            yield dict(
                xs=self.xs[s_idxs],
                ys=self.ys[s_idxs],
                ws=self.ws[s_idxs],
                xt=self.xt[t_idxs],
                yt=self.yt[t_idxs],
                wt=self.wt[t_idxs]
            )
        