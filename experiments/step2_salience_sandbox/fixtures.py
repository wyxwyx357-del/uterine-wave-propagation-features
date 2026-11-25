"""Tiny deterministic infrastructure fixtures, not a Dev/Val benchmark."""
import numpy as np

FIXED_NOISE_FLOOR=.0007509593468916695


def case(kind):
    time=np.arange(1001)*.12
    begin,end=time[:-17],time[17:]
    center=(begin+end)/2
    span=end-begin
    s=np.linspace(.1,.9,8)
    a=np.zeros((len(center),2,len(s)))
    truth=[]
    def add(name,direction,at,amplitude,amplitude_class):
        arrivals=at+(1 if direction=='C2F' else -1)*(s-.5)/.04
        def length(t):
            offset=t[:,None]-arrivals
            return 1+amplitude*np.where(abs(offset)<=12,np.exp(-.5*(offset/4)**2),0)
        signal=(length(end)-length(begin))/length(begin)/span[:,None]
        a[:]+=signal[:,None,:]
        truth.append(dict(truth_event_id=name,truth_start_s=float(arrivals.min()-12),
                          truth_end_s=float(arrivals.max()+12),truth_center_s=float(at),
                          truth_direction=direction,truth_sections=list(range(len(s))),
                          truth_amplitude_class=amplitude_class,truth_amplitude=amplitude,truth_speed=.04))
    if kind in ('clear_C2F','artifact'):
        add('clear','C2F',40,.04,'clear')
    elif kind=='clear_F2C':
        add('clear','F2C',40,.04,'clear')
    elif kind=='two':
        add('first','C2F',32,.04,'clear');add('second','F2C',87,.04,'clear')
    elif kind=='mixed':
        add('strong','C2F',32,.06,'clear');add('medium','F2C',87,.015,'medium')
    elif kind=='upstream_miss':
        add('below_upstream','C2F',40,1e-8,'structured_weak')
    elif kind!='zero':
        raise ValueError('Unknown tiny fixture')
    if kind=='artifact':
        a[:,1,-1]+=.03*np.exp(-.5*((center-85)/8)**2)
    maps=dict(A=a,valid_mask=np.ones_like(a,bool),s=s,interval_center_s=center,
              interval_start_s=begin,interval_end_s=end,interval_dt_s=span)
    return maps,truth


KINDS=('zero','clear_C2F','clear_F2C','two','artifact','mixed','upstream_miss')
