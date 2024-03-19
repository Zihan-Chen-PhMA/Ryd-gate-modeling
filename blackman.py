import numpy as np 
import matplotlib.pyplot as plt 


def blackman_window(t, t_rise):
    return (0.42 - 0.5*np.cos(2*np.pi*t/(2*t_rise)) + 
            0.08*np.cos(4*np.pi*t/(2*t_rise)))


def blackman_pulse(t, t_rise, t_gate):
    # if t_gate > 2*t_rise:
    #     raise ValueError("t_gate is too small compared to t_rise")
    ret = (blackman_window(t,t_rise)*np.heaviside(t_rise-t,1) + 
           np.heaviside(t-t_rise,0)*np.heaviside(t_gate-t-t_rise,0) +
           blackman_window(t_gate-t,t_rise)*
           np.heaviside(t_rise-(t_gate-t),1))
    return ret

