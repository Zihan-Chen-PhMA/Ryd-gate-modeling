from qutip import *
import numpy as np
import matplotlib.pyplot as plt

#Function of PSD. para covers all parameters of this function 
def S_white(f,para):
    '''
    white noise with width 'fwidth', intensity 'h0', frequncy begin point 'fbegin'
    para = [h0,fwidth,fbegin]
    '''
    [h0,fwidth,fbegin] = para
    if fbegin <= f <= fbegin+fwidth:
        res = h0 
    else:
        res = 0
    return res

def S_real(f,para):
    '''
    Lorentz line + white noise
    para = [h0,hL,F,f0]
    '''
    [h0,hL,F,f0] = para
    if f<f0:
        res = h0 
    else:
        res = hL/(1 + (f*F/(1.5*10**9))**2)
    return res

def S_servo_freq(f,para):
    [h0,sg,fg,sigmag] = para
    res = h0 + (sg *(fg)*(fg))/(np.sqrt(8*np.pi)*sigmag)* np.exp(-((f - fg)**2)/(2*sigmag**2)) 
    #+ hg* np.exp(-((f + fg)**2)/(2*sigmag**2))
    return res

def noise_coeff(t,args):
    '''
    Function generate the trace of the noise. 
    args includes ['phi','para','sumpara','Sfun']

    'Sfun' is the function of PSD with input [f,para]
    'para' covers all parameters of PSD function
    'sumpara' = [f0,fwidth,N] clearify the center point of summontion f0, the summontion bandwidth fwidth, the fraction number N
    'phi' is phase that used to be random sampled from [0,2pi] 
    '''
    phase_noise = 0
    phi_vec = args['phi']
    para = args['para']
    sumpara = args['sumpara']
    Sfun = args['Sfun']
    f0,fwidth,N = sumpara
    deltaf = fwidth / N
    #sum over the band wideth around point f0
    for i in range(N):
        f = f0 - fwidth/2 +deltaf*i
        PSD = Sfun(f,para)
        phase_noise = phase_noise + 2*np.pi*np.sqrt(PSD * deltaf) * np.sin((f)*2*np.pi*t + phi_vec[i])
    return phase_noise


def noise_para_plot(process_para,Sfun,PSD_para,Plotchoice,Plotrange,sumpara,simnum):
    '''
    process_para:
    process_para = [Omega,q], Omega is the Rabi frequency, q means we check the noise of q-\pi Palse process result.
    
    Sfun:
    Function of PSD, with input [f, para]
    
    PSD_para: 
    The parameter input into Sfun. Order is the same with para
    
    Plotchoice: 
    integer, for Plotchoice = 0, plot the 1-th parameter in PSD_para
    
    Plotrange: 
    Plotrange =  np.linspace(begin,end,plotsegment),  
    Plotrange* Omega /(2*np.pi) will be the plotrange of parameter
    
    sumpara: 
    has the same shape of Plotrange. sumpara[i] =[fcenter,fwidth,N] 
    Allow changing summontion range to [fcenter - fwidth/2, fcenter + fwidth/2]* Omega /(2*np.pi)
    with N segments when parameter changes

    simnum: 
    Simulation numbers
    '''

    [Omega, q] = process_para
    time_scale =  np.pi/Omega
    tau = q *time_scale # simulation time
    index = int((1+(-1)**q)*0.5)
    ptnum = 2 #time segment

    plot_list = Omega * Plotrange /(2*np.pi)
    error_list = np.zeros((len(plot_list),simnum))
    mean_list = np.zeros(len(plot_list))
    dev_list = np.zeros(len(plot_list))

    for i in range(len(plot_list)):
        PSD_para[Plotchoice] = plot_list[i]
        sumparameter = sumpara[i]
        N = sumparameter[-1]
        #defined time-dependent Hamiltonian given sampled noise
        for j in range(simnum):
            phi_vec = 2*np.pi * np.random.random(size = N)
            args = {'phi':phi_vec,'para':PSD_para,'sumpara':sumparameter,'Sfun':Sfun}
            H = [0.5 * Omega * sigmax(),[ sigmaz(),noise_coeff]] 
            time = np.linspace(0,tau,ptnum)
            psi0 = basis(2, 0)
            result = sesolve(H, psi0, time, args = args)
            
            #print error
            print(np.abs(result.states[-1][index,0])**2)
            error_list[i,j] = np.log10(np.abs(result.states[-1][index,0])**2)       
        
    for i in range(len(plot_list)):
        mean_list[i] = np.mean(error_list[i])
        dev_list[i] = np.std(error_list[i])
    print(error_list)

    plt.errorbar(Plotrange,mean_list,yerr = dev_list, fmt = 'o')
    plt.xlabel(r'$parameter /(\Omega/2\pi)$')
    plt.ylabel(r'$log_{10}(error)$')
    plt.title( r'%s $\pi$-Palse Error'%q)
    plt.show()

def PSD_plot(PSD,plotrange,para):
    '''
    This function visually show the shape of PSD function
    PSD: Noise function
    para:parameter of the PSD function
    plotrange: list of [begin point, end point]
    '''
    [begin, end] = plotrange
    plotseg = 1000
    x = np.linspace(begin,end,plotseg)
    y = np.zeros(plotseg)
    for i in range(plotseg):
        y[i] = PSD(x[i],para)
    plt.plot(x,y)
    plt.xlabel('f')
    plt.ylabel(r'$\S_{\delta \nu}(f)$')
    plt.title('parameter = [%s]'%para)
    plt.show()

def decay_plot(PSD,PSD_para,simnum,sumparameter,plot_para):
    '''
    PSD: Function of PSD

    PSD_para: parameters of PSD

    simnum:simulation time

    plot_para:[Omega,N,ptnum ]
    Omega:Rabi
    N : N-pi pulse
    ptnum = time segments

    sumparameter = [f0,fwidth,N] 
    clearify the center point of summontion f0, the summontion bandwidth fwidth, the fraction number N
    '''
    Omega,q,ptnum = plot_para
    tau = q*np.pi/Omega
    index = int((1+(-1)**q)*0.5)
    p = np.zeros((simnum,ptnum))
#defined time-dependent Hamiltonian given sampled noise
    for j in range(simnum):
        N = sumparameter[-1]
        phi_vec = 2*np.pi * np.random.random(size = N)
        args = {'phi':phi_vec,'para':PSD_para,'sumpara':sumparameter,'Sfun':PSD}
        H = [0.5 * Omega * sigmax(),[ sigmaz(),noise_coeff]] 
        time = np.linspace(0,tau,ptnum)
        psi0 = basis(2, 0)
        result = sesolve(H, psi0, time, args = args)
        print(np.abs(result.states[-1][index,0])**2)
            
       #Plot Prob(ground state) changed with time
        for i in range(ptnum):
            p[j,i] = np.abs(result.states[i][0,0])**2
    
    plt.figure()
    plt.grid()
    for j in range(simnum):
        plt.plot(time,p[j],color = 'red')
    
    psum = np.zeros(ptnum)
    for i in range(ptnum):
        psum[i] = np.sum(p[:,i])/simnum
    plt.plot(time,p[j],label = r'$|g \rangle$',color = 'black')
    plt.savefig('savefig_example.png')
    plt.show()
        
def main1():
    #Plot decay pic
    Omega = 10**6 * 2* np.pi #Rabi frequency
    q = 30
    ptnum = 100
    plot_para = Omega,q,ptnum

    h0 = 0
    sigmag = 1400  #sigmag/Omega
    hg = 1100
    fg0 = 10**6
    sg = np.sqrt(8*np.pi) * sigmag * hg /(fg0*fg0) *100
    fg = Omega /(2*np.pi)
    PSD_para = [h0,sg,fg,sigmag]

    simnum = 10

    sumpara = [fg, 4*sigmag,500]

    decay_plot(S_servo_freq,PSD_para,simnum,sumpara,plot_para)

def main2():
    #Plot noise change with noise parameter
    Omega = 10**6 * 2* np.pi #Rabi frequency
    q = 1
    process_para = [Omega, q]

    h0 = 100
    hL = 3000
    F = 10000
    f0 = 5*10**5
    PSD_para = [h0,hL,F,f0]
    Plotchoice = 3
    N = 50
    Plotrange = np.linspace(0.5,5,N) 

    fwidth =  4*f0
    sumpara = [[2*f0, fwidth ,500] for t in range(N)]

    simnum = 30
    #You can use PSD_plot to visually choose the best sum range of frequency
    PSD_plot(S_real,[0,fwidth],PSD_para)
    noise_para_plot(process_para, S_real , PSD_para , Plotchoice , Plotrange , sumpara , simnum)

    

if __name__ == '__main__':
    main1()
    






# plt.figure()
# plt.grid()
# plt.plot(time,phase_noise,label = 'Noise phase')
# plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0)
# plt.show()

