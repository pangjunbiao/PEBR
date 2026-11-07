"""Exploratory receiver-preserving score; not an accepted replacement for PEBR.

No target outside the supplied inner view is used. Whole-word guarantees apply
only to the tested masked views, not final coherence or untouched targets.
"""
from __future__ import annotations
import numpy as np
from transport_topics.proposed.scoring import ActionEvaluator
from transport_topics.proposed.transfer import mixture_weights,mix_views

class ReceiverAnchoredEvaluator(ActionEvaluator):
    def __init__(self,reference,visible,targets,candidate_lists,donors,base_counts,
                 donor_projection=None,batch_size=128):
        super().__init__(reference,visible,targets,candidate_lists,donors,donor_projection,batch_size)
        counts=np.asarray(base_counts,dtype=float).reshape(-1)
        if counts.shape!=(self.visible.shape[1],) or np.any(counts<0) or not np.isfinite(counts).all():
            raise ValueError('Expected one original base-count row')
        total=float(counts.sum());h=self.targets.sum(axis=1);v=total-h
        if total<=0 or np.any(v<=0):raise ValueError('Both parts of each mask must be nonempty')
        if not np.allclose(self.visible*v[:,None]+self.targets,counts[None,:],atol=1e-10,rtol=1e-12):
            raise ValueError('Inner visible/hidden rows do not reconstruct this receiving post')
        self.all_ids=np.flatnonzero(counts);self.hidden_weights=self.targets[:,self.all_ids]
        self.visible_weights=self.visible[:,self.all_ids];self.hidden_shares=h/total
        self.hidden_mass=h;self.diagnostics={}
        baseline=reference.predict(self.visible,self.all_ids)
        self.base_hidden=(self.hidden_weights*np.log(baseline)).sum(axis=1)/h
        self.base_visible=(self.visible_weights*np.log(baseline)).sum(axis=1)
        self.cache={}

    def _diagnostics(self,probability,inactive):
        hidden=(np.log(probability)*self.hidden_weights[None]).sum(axis=2)/self.hidden_mass[None]-self.base_hidden[None]
        visible=(np.log(probability)*self.visible_weights[None]).sum(axis=2)-self.base_visible[None]
        hidden[inactive]=0.;visible[inactive]=0.
        net=self.hidden_shares[None]*hidden-(1-self.hidden_shares[None])*np.maximum(-visible,0.)
        whole=self.hidden_shares[None]*hidden+(1-self.hidden_shares[None])*visible
        if not np.isfinite(net).all():raise ArithmeticError('Nonfinite anchored gain')
        if np.any(whole<net-1e-10):raise ArithmeticError('Receiver-word lower bound failed')
        return hidden,visible,net,whole

    def gains(self,actions):
        result=np.zeros((len(actions),self.views));pending=[]
        for i,a in enumerate(actions):
            if not a.donors:continue
            if a in self.cache:result[i]=self.cache[a]
            else:pending.append((i,a))
        for start in range(0,len(pending),self.batch_size):
            entries=pending[start:start+self.batch_size];chunk=[a for _,a in entries]
            projection=np.empty((len(chunk),self.views,self.reference.k));inactive=np.empty((len(chunk),self.views),bool)
            for i,a in enumerate(chunk):
                available=self._selected_availability(a);alpha,weights=mixture_weights(a,available)
                projection[i]=(1-alpha[:,None])*self.base_projection+weights@self.donor_projection[list(a.donors)]
                inactive[i]=alpha==0
            probabilities=self.reference.from_projection(projection.reshape(-1,self.reference.k),self.all_ids,
                input_factory=lambda ids:self._fallback_inputs(chunk,ids)).reshape(len(chunk),self.views,-1)
            hidden,visible,net,whole=self._diagnostics(probabilities,inactive)
            for (destination,a),hg,vg,ng,wg in zip(entries,hidden,visible,net,whole):
                self.cache[a]=ng.copy();result[destination]=ng
                self.diagnostics[a]={'hidden_gain':hg.copy(),'visible_gain':vg.copy(),
                    'net_gain':ng.copy(),'whole_receiver_gain':wg.copy()}
            self.action_evaluations+=len(chunk)
        return result

    def direct_gains(self,action):
        if not action.donors:return np.zeros(self.views)
        available=self._selected_availability(action)
        mixed,alpha=mix_views(self.visible,action,available,self.donors[list(action.donors)])
        probability=self.reference.predict(mixed,self.all_ids)
        return self._diagnostics(probability[None],(alpha==0)[None])[2][0]
