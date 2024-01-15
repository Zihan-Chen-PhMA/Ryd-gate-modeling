
# 白噪声和伺服噪声
伺服噪声的功率谱函数分别为：
![](https://cdn.nlark.com/yuque/__latex/dd71676d72da375144196d17caabefdf.svg#card=math&code=S_%7B%5Cdelta%20%5Cnu%7D%20%3D%20h_0%20%2B%20%5Cfrac%7Bs_g%20f_g%5E2%7D%7B%5Csqrt%7B8%5Cpi%7D%20%5Csigma_g%7D%20e%5E%7B%5Cfrac%7B-%28f%20-%20f_g%29%5E2%7D%7B2%5Csigma_g%5E2%7D%7D%20%2B%20%5Cfrac%7Bs_g%20f_g%5E2%7D%7B%5Csqrt%7B8%5Cpi%7D%20%5Csigma_g%7D%20%20e%5E%7B%5Cfrac%7B-%28f%20%2B%20f_g%29%5E2%7D%7B2%5Csigma_g%5E2%7D%7D&id=biyWP)
当![](https://cdn.nlark.com/yuque/__latex/c7a0d892621f911e9ab0892f5163e5ab.svg#card=math&code=h_g%20%3D%200&id=xhCkA),成为白噪声, 此外可能有多个gauss Bump存在。
数据的典型值:对![](https://cdn.nlark.com/yuque/__latex/ed390549207207d4d83a777b546560a0.svg#card=math&code=%5COmega%20%3D%202%5Cpi%20%5Ctimes1%20MHz&id=cnyfE)的激光，设置![](https://cdn.nlark.com/yuque/__latex/be8ecec7adfd1d401121d87158828652.svg#card=math&code=h_0%20%3D%200%20Hz%2C%20f_g%20%5Cin%20%280.1%2C4%29%20%5Ctimes%20%5Cfrac%7B%5COmega%7D%7B2%5Cpi%7D%2C%5Csigma_g%20%3D%201400%20Hz%2C%20s_g%20%3D%20%5Cfrac%7B%5Csqrt%7B8%5Cpi%20%7D%20%5Csigma_g%20h_g%7D%7Bf_%7Bg0%7D%5E2%7D&id=W5Yo1), 其中![](https://cdn.nlark.com/yuque/__latex/eebcd5f693b9c1406220abdc2ebb1d1a.svg#card=math&code=h_g%20%3D%201100%2Cf_%7Bg0%7D%20%3D%2010%5E6&id=Cb1IU)。
给定功率谱密度，可以用带有均匀随机相位的频率混合来重现噪声。如已知相位噪声的功率谱密度![](https://cdn.nlark.com/yuque/__latex/8961887da95b56c4d3cfad4c1fd58564.svg#card=math&code=S_%7B%5Cphi%7D%28f%29&id=xLpJv),可以设置：
![](https://cdn.nlark.com/yuque/__latex/6b032684297d245767d3cd4bdc52eb2d.svg#card=math&code=%5Cphi%28t%29%20%3D%20%5Csum_%7Bj%20%3D%201%7D%5E%7B%5Cinfty%7D%202%20%5Csqrt%7BS_%7B%20%5Cphi%7D%28f_j%29%20%5CDelta%20f%7D%20%5Ccos%20%282%5Cpi%20f_j%20t%20%2B%20%5Cvarphi_j%20%29%20&id=gbncD)
其中![](https://cdn.nlark.com/yuque/__latex/44b1d3bf6a7f02ea3edc046c85f36e8a.svg#card=math&code=%5Cvarphi_i&id=ixKDQ)是从![](https://cdn.nlark.com/yuque/__latex/1eb56e1b43f922024358d6b4ba2659e3.svg#card=math&code=%5B0%2C2%5Cpi%5D&id=pcwdv)间均匀选取的随机变量。上述随机函数的导数为：
![](https://cdn.nlark.com/yuque/__latex/091de4a62629cd162b9da2cd7c2b7f03.svg#card=math&code=%5Cbegin%7Balign%7D%0A%5Cfrac%7Bd%5Cphi%28t%29%7D%7Bdt%7D%20%26%3D%20-%5Csum_%7Bj%20%3D%201%7D%5E%7B%5Cinfty%7D%204%5Cpi%20f_j%20%5Csqrt%7BS_%7B%20%5Cphi%7D%28f_j%29%20%5CDelta%20f%7D%20%5Csin%20%282%5Cpi%20f_j%20t%20%2B%20%5Cvarphi_j%20%29%20%5C%5C%0A%26%3D%20-%5Csum_%7Bj%20%3D%201%7D%5E%7B%5Cinfty%7D%204%5Cpi%5Csqrt%7BS_%7B%20%5Cdelta%5Cnu%7D%28f_j%29%20%5CDelta%20f%7D%20%5Csin%20%282%5Cpi%20f_j%20t%20%2B%20%5Cvarphi_j%20%29%20%0A%5Cend%7Balign%7D&id=SPY1P)
其中利用了关系：![](https://cdn.nlark.com/yuque/__latex/91a29f4544e836ca1c11412623396835.svg#card=math&code=S_%7B%20%5Cdelta%5Cnu%7D%28f%29%20%3D%20f%5E2%20S_%7B%20%5Cphi%7D%28f%29%20&id=OwO5d).
# 噪声的测量
### PDH直接测量
将PDH锁频误差信号接入频谱仪，即可实现对PDH锁后激光相噪的直接测量。原理如下：
PDH锁定点附近的误差信号近似为一条很陡的直线，其斜率与![](https://cdn.nlark.com/yuque/__latex/6b673fbf3f5261342141ccd09600d152.svg#card=math&code=%5Cdelta%20%5Cnu&id=eR5FX) 成正比。误差信号的斜率会随时间抖动，从中可以得到频率差关于时间的变化![](https://cdn.nlark.com/yuque/__latex/b7f1c4c9462a050f791b2d26f21d4ec9.svg#card=math&code=%5Cdelta%20%5Cnu%28t%29&id=DxcTI).由于![](https://cdn.nlark.com/yuque/__latex/2473f21093ad2d297815d379630121b2.svg#card=math&code=R_%5Cnu%28%5Ctau%29%3D%3C%5Cnu%28t%29%5Cnu%28t%2B%5Ctau%29%3E%2CS_%5Cnu%28f%29%3DF%28R_%5Cnu%28%5Ctau%29%29&id=bobID),我们可以在频谱仪中读出相噪谱![](https://cdn.nlark.com/yuque/__latex/d03c53ec0b0bccf5615e5ac79cacce95.svg#card=math&code=S_%5Cnu%28f%29&id=SCn8h).
然而，由于误差信号的直线段很窄，我们只能读出![](https://cdn.nlark.com/yuque/__latex/6b673fbf3f5261342141ccd09600d152.svg#card=math&code=%5Cdelta%20%5Cnu&id=NuL28)很小时的噪声，即低频噪声。故这种方法一般不采用。
### Self-Heterodyne Spectrum
这种方法将信号分成两束，一束经过了![](https://cdn.nlark.com/yuque/__latex/542db859ba8715e8897d94667091b476.svg#card=math&code=t_d&id=Tthwy)时间的延迟，一束经过了![](https://cdn.nlark.com/yuque/__latex/6f7b500649e1be26d0e01eceae6aabe4.svg#card=math&code=%5Cnu_s&id=GSFc1)的频率移动，接着二者干涉共同打在一个光电二极管上。光电二极管输出一个正比于光强的电流，通过分析这个电流的噪声功率谱，就可以反推出![](https://cdn.nlark.com/yuque/__latex/bd805b32bcbd2bdc4d58605415e80109.svg#card=math&code=S_%5Cphi&id=Z6UhN).
### beat with PDH transmission signal
# 噪声的压制
## 滤波腔
由于腔的透射函数![](https://cdn.nlark.com/yuque/__latex/ec24a7c38a393c10de6ad2ff24d674e6.svg#card=math&code=T%28w%29&id=fJ9aY)为洛伦兹线形，其线宽定义为腔的线宽![](https://cdn.nlark.com/yuque/__latex/f013a947cdf24436d1a3ec1b2f34ce89.svg#card=math&code=%5CGamma%20%3D%20%5Cfrac%7BFSR%7D%7BF%7D&id=bWnd4)，所以腔可以看作一个滤波器，滤去入射光中![](https://cdn.nlark.com/yuque/__latex/8c6e971606ed92bc3cfeb1bcb327cdcd.svg#card=math&code=%5CDelta%20w%20%3E%20%5CGamma%20&id=QLkzX)的部分。使用一个![](https://cdn.nlark.com/yuque/__latex/0449938de709a51d4c215260b39520dd.svg#card=math&code=F%3D3000&id=FyJJl)的腔，其线宽为![](https://cdn.nlark.com/yuque/__latex/0a012ed6a92cd290ad43a0f0a7baa6b0.svg#card=math&code=500kHz&id=o4Akv), 可以滤掉![](https://cdn.nlark.com/yuque/__latex/0f2b19716ec3530d8d9cc8061e4decd2.svg#card=math&code=1MHz&id=OaNQC)以上的高频噪声。其缺点是腔的进光功率阈值较低(为![](https://cdn.nlark.com/yuque/__latex/38e13fcf2d999bb27e44af9055fe8268.svg#card=math&code=mW&id=p6Znq)量级)，在透射后光强会更小，因此常常需要进行注入锁定和光纤放大。
## PDH前馈系统
# ![802230031e6e4234b06e1ef3ba12cd6.png](https://cdn.nlark.com/yuque/0/2024/png/41004299/1705315300087-0f0dff7e-6263-4121-86b8-bd3b78e9d372.png#averageHue=%23fbf5f4&clientId=u2f9816db-30df-4&from=drop&height=395&id=u532aec9f&originHeight=441&originWidth=538&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=52001&status=done&style=none&taskId=u426c5f54-13c3-499f-9d87-d34f1cf293c&title=&width=481.3333435058594)
PDH前馈系统由一个相位延迟光纤，一个移相器(EOM)和一个P电路构成。
PDH前馈的理论推导详见:[https://arxiv.org/abs/2309.09759](https://arxiv.org/abs/2309.09759).
假设入射光含有![](https://cdn.nlark.com/yuque/__latex/bcd578764d08d90add8eb37bb20d9d51.svg#card=math&code=%5Cvarphi%28t%29&id=SaNiC)的相位噪声，前馈系统意图通过光纤的相位延迟来弥补这个相位噪声。PDH锁后的激光线宽远低于腔的线宽，而我们关心的高频相噪线宽远高于腔的线宽。在这个近似下，我们得到在锁定点附近的误差信号与![](https://cdn.nlark.com/yuque/__latex/bcd578764d08d90add8eb37bb20d9d51.svg#card=math&code=%5Cvarphi%28t%29&id=wRWe7)的关系:
![](https://cdn.nlark.com/yuque/__latex/057cad69da49c5850037fa19d358b211.svg#card=math&code=V_%7Berr%7D%20%5Cpropto%20%5Cvarphi%28t%29&id=LXk2v).
我们将误差信号接入一个P电路，并前馈到EOM，可以使EOM出射光产生额外的相位调制：
![](https://cdn.nlark.com/yuque/__latex/743515b2e21ed0a6826f92856950deb4.svg#card=math&code=E_%7Bout%7D%20%3D%20E_0%20exp%28i%28%5Comega_c%20t%2B%20%5Cvarphi%28t%29%2BGsin%28%5Cvarphi%28t%27%29%29%29&id=pnc8n)
调节参数使![](https://cdn.nlark.com/yuque/__latex/1c01ca08f65371b613cd2cdd0dab007b.svg#card=math&code=G%3D-1%2Ct%27%5Capprox%20t&id=yvsz7),由于![](https://cdn.nlark.com/yuque/__latex/c576bca6428131f6a16a768ba26c457b.svg#card=math&code=%7C%5Cvarphi%28t%29%7C%3C%3C1&id=FLea6),我们便可以弥补相位噪声。论文中详细论证了这种方法对低频噪声的处理作用，并证明了它与滤波腔几乎等价。但对于高频噪声，一旦离开了![](https://cdn.nlark.com/yuque/__latex/36782d3824e752431e3fef7827d751a2.svg#card=math&code=Im%28R%28w%29%29&id=kGLHb)的近似直线段，且受限于调制EOM的带宽(![](https://cdn.nlark.com/yuque/__latex/813b1a69188dd3907071deeda849970a.svg#card=math&code=%5Capprox%2020MHz&id=hmNYa))，相噪就难以抑制。其相噪抑制效果图如下：

![70b823ee46948d25d0aeef69abdf75a.png](https://cdn.nlark.com/yuque/0/2024/png/41004299/1705317200527-d52d568a-86f6-4e52-a18f-1f29142462ca.png#averageHue=%23f9f6f5&clientId=u2f9816db-30df-4&from=drop&id=uf8e8b599&originHeight=487&originWidth=742&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=56202&status=done&style=none&taskId=u661fe556-c73f-45c9-8733-830100f3212&title=)
可见其对![](https://cdn.nlark.com/yuque/__latex/e09399716bb1f7c1b63895f8f67f64d4.svg#card=math&code=2MHz&id=NaOUH)以上的高频噪声抑制效果不理想。








# 有相位噪声的单光子过程
在谐振条件下，有相位噪声的哈密顿量可以写为
![](https://cdn.nlark.com/yuque/__latex/f5e0482ba4820fe6e847980104d52e5f.svg#card=math&code=H%20%3D%20%5Cfrac%7B%5Chbar%20%5COmega%7D%7B2%7D%20%5Be%5E%7Bi%5Cphi%28t%29%7D%20%7Ce%5Crangle%20%5Clangle%20g%7C%20%2B%20e%5E%7B-i%5Cphi%28t%29%7D%20%7Cg%5Crangle%20%5Clangle%20e%7C%5D&id=W8GU1)
若使用含时酉变换： ![](https://cdn.nlark.com/yuque/__latex/430d3cfaed92e474653359c9da0c2d24.svg#card=math&code=U%28t%29%20%3D%20e%5E%7B-i%5Cphi%28t%29%20%7Ce%5Crangle%20%5Clangle%20e%7C%7D%20&id=A3C1y)，新的基矢下的哈密顿量为：
![](https://cdn.nlark.com/yuque/__latex/27719052dc7dec9aaddefea08fb46ac9.svg#card=math&code=%5Cbegin%7Balign%7D%0AH%27%20%26%3D%20i%5Chbar%20%5Cfrac%7BdU%7D%7Bdt%7D%20U%5E%5Cdagger%20%2B%20UHU%5E%5Cdagger%20%5C%5C%0A%26%3D%20%5Chbar%20%5Cfrac%7Bd%5Cphi%7D%7Bdt%7D%20%7Ce%5Crangle%5Clangle%20e%7C%20%2B%20%5Cfrac%7B%5Chbar%20%5COmega%7D%7B2%7D%20%5B%20%7Ce%5Crangle%20%5Clangle%20g%7C%20%2B%20%7Cg%5Crangle%20%5Clangle%20e%7C%5D%20%5C%5C%0A%20%26%3D%20%5Cfrac%7B%5Chbar%20%5COmega%7D%7B2%7D%20%5B%20%7Ce%5Crangle%20%5Clangle%20g%7C%20%2B%20%7Cg%5Crangle%20%5Clangle%20e%7C%5D%20-%20%5Csum_%7Bj%20%3D%201%7D%5E%7B%5Cinfty%7D%204%5Cpi%20%5Chbar%20%5Csqrt%7BS_%7B%5Cdelta%20%5Cnu%7D%28f_j%29%20%5CDelta%20f%7D%20%5Csin%20%282%5Cpi%20f_j%20t%20%2B%20%5Cvarphi_j%20%29%20%7Ce%5Crangle%5Clangle%20e%7C%0A%5Cend%7Balign%7D&id=CO5RV)
其中利用了频率噪声的估计表达式。
我们求解：
![](https://cdn.nlark.com/yuque/__latex/c809462ce9c0af5ac0afc2e695f0f59a.svg#card=math&code=%5Cbegin%7Balign%7D%0AH%27%2F%5Chbar%20%20%26%3D%20%5Cfrac%7B1%7D%7B2%7D%20%5COmega%20%5Csigma_x%20%2B%20%5Csum_%7Bj%20%3D%201%7D%5E%7B%5Cinfty%7D%202%5Cpi%20%5Csqrt%7BS_%7B%5Cdelta%20%5Cnu%7D%28f_j%29%20%5CDelta%20f%7D%20%5Csin%20%282%5Cpi%20f_j%20t%20%2B%20%5Cvarphi_j%20%29%20%5Csigma_z%0A%5Cend%7Balign%7D&id=GZr93)
在![](https://cdn.nlark.com/yuque/__latex/47143d1242a2160c949497189492deaf.svg#card=math&code=%5Cfrac%7B%5Csqrt%7BS_%7B%5Cdelta%5Cnu%7D%28f_i%29%20%5CDelta%20f%7D%7D%7B%20%5COmega%7D%20%3A%3D%20%5Cdelta_i%20%5Cll%201&id=zUrqV)的近似下，把误差部分视为微扰，在保留二阶的近似下，可以解析地给出上述哈密顿量的演化结果，详见文献[[链接]()]公式(74)

在模拟中，为了简化，我们设置初态为![](https://cdn.nlark.com/yuque/__latex/92675c2f26f99751dfa90658c603ef19.svg#card=math&code=%5Cket%7Bg%7D&id=cD39x),激光作用使其在基态和激发态间震荡，误差为![](https://cdn.nlark.com/yuque/__latex/26d01865f6e830adaa66117e307e316f.svg#card=math&code=%5Cpi&id=NiYuO)pause后，测量坍缩到基态的概率。
```python
from qutip import *
import numpy as np
import matplotlib.pyplot as plt

#Function of PSD. para covers all parameters of this function 
def S_freq(f,para):
    h0 = para
    res = h0 
    return res

#Function generate the trace of the noise. 
# para covers all parameters of PSD function
#sumpara = [f0,fwidth,N] clearify the begin point of summontion f0, the summontion bandwidth fwidth, the fraction number N
def noise_coeff(t,args):
    phase_noise = 0
    phi_vec = args['phi']
    para = args['para']
    sumpara = args['sumpara']
    f0,fwidth,N = sumpara
    deltaf = fwidth / N
    #sum over the band wideth around point f0
    for i in range(N):
        f = f0 - fwidth/2 +deltaf*i
        PSD = S_freq(f,para)
        phase_noise = phase_noise + 2*np.pi*np.sqrt(PSD * deltaf) * np.sin((f)*2*np.pi*t + phi_vec[i])
    return phase_noise



Omega = 10**6 * 2* np.pi #Rabi frequency
time_scale =  np.pi/Omega
tau = 1*time_scale # simulation time
Delta = 0 # detuning
ptnum = 2 #time segment


# Sampling phase noise and phase noise parameter.
h0 = 1000 #Hz/Hz^2
simnum = 50

fwidth = 0.1*Omega/(2*np.pi) #Hz, the bandwidth of power spectrum of noise phase
N = 500 # cut noise bandwidth into N parts

x = np.linspace(0.01,9,80) 
hcenter_var = Omega * x /(2*np.pi)

plot_list = hcenter_var
error_list = np.zeros((len(plot_list),simnum))
mean_list = np.zeros(len(plot_list))
dev_list = np.zeros(len(plot_list))

for i in range(len(plot_list)):
    parameter = h0
    sumparameter = [plot_list[i],fwidth,N]
    #defined time-dependent Hamiltonian given sampled noise
    for j in range(simnum):
        phi_vec = 2*np.pi * np.random.random(size = N)
        args = {'phi':phi_vec,'para':parameter,'sumpara':sumparameter}
        H = [0.5 * Omega * sigmax(),[ sigmaz(),noise_coeff]] 
        time = np.linspace(0,tau,ptnum)
        psi0 = basis(2, 0)
        result = sesolve(H, psi0, time, args = args)
        
        #print error
        print(np.abs(result.states[-1][0,0])**2)
        error_list[i,j] = np.log10(np.abs(result.states[-1][0,0])**2)       
    
for i in range(len(plot_list)):
    mean_list[i] = np.mean(error_list[i])
    dev_list[i] = np.std(error_list[i])
print(error_list)

plt.errorbar(x,mean_list,yerr = dev_list, fmt = 'o')
plt.xlabel(r'$h_{center} /(\Omega/2\pi)$')
plt.ylabel(r'$log_{10}(error)$')
plt.title(r'$\pi$-Palse Error')
plt.show()

```
![whitenoise_translation.png](https://cdn.nlark.com/yuque/0/2024/png/40467816/1705286579815-54e77337-658e-4955-ad8c-1a36710553d0.png#averageHue=%23fbfbfb&clientId=u7d0b75ef-0575-4&from=drop&height=285&id=u5d913f1a&originHeight=480&originWidth=640&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=20791&status=done&style=none&taskId=uebf95c3c-9c36-45c5-805a-b54d0255593&title=&width=380)![Figure_2.png](https://cdn.nlark.com/yuque/0/2024/png/40467816/1705298565300-aa80c6ee-b69d-45b5-b920-d7d135da296a.png#averageHue=%23fbfbfb&clientId=u3254b533-af6d-4&from=drop&height=266&id=ue8fb3470&originHeight=480&originWidth=640&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=21290&status=done&style=none&taskId=u99f780f6-43e9-45dd-9a11-2f087bbf8b7&title=&width=355)
上图是固定宽度高度的白噪声在有不同起始点![](https://cdn.nlark.com/yuque/__latex/2c5e10a0f94e0fb1982f2b333f121cf6.svg#card=math&code=f_0&id=qaKtw)时噪声带来的误差。其中纵轴是 ![](https://cdn.nlark.com/yuque/__latex/2b018c0572bf1e633151c2b0ecf10596.svg#card=math&code=10%5Ctimes%20log_%7B10%7D%28p%20%5Ctimes%201000%29&id=wWQZG)，是dBm的单位。横轴是![](https://cdn.nlark.com/yuque/__latex/4c32ccde49733b4e386c3e8e671b3b01.svg#card=math&code=2%5Cpi%20f_0%2F%20%5COmega&id=ql2zf) ,![](https://cdn.nlark.com/yuque/__latex/2c5e10a0f94e0fb1982f2b333f121cf6.svg#card=math&code=f_0&id=vaZd9)是白噪声起始点,区间为 ![](https://cdn.nlark.com/yuque/__latex/c590e215e0008be50ef7692d709ae298.svg#card=math&code=%5Bf_0%2C%20f_0%2B10%5E5Hz%5D&id=LT9sz), ![](https://cdn.nlark.com/yuque/__latex/69647a759c307a5bf2e5c03e96db880a.svg#card=math&code=h0%20%3D%201000Hz&id=gVYBG), ![](https://cdn.nlark.com/yuque/__latex/c7f9ad31a0870501baefdce3e4519215.svg#card=math&code=%5COmega%20%3D%202%5Cpi%20%2A10%5E6%20Hz&id=BuvIU)。随着分布逐渐靠向高频噪声，误差在震荡地减小。
更真实的噪声分布具有如下的函数形式，它由一段白噪声和随后的洛伦兹线形的噪声拼接而成
![](https://cdn.nlark.com/yuque/__latex/eaea400dbf89d8d9e554e02285df8848.svg#card=math&code=%5Cbegin%7Balign%7D%0AS_%7B%5Cdelta%20%5Cnu%7D%28f%29%20%3D%20%5Cleft%5C%7B%0A%5Cbegin%7Baligned%7D%0A%26h_0%2C%20%5Cquad%20%20f%20%3C%20f_0%5C%5C%0A%26%5Cfrac%7Bh_L%7D%7B1%20%2B%20%28%5Cfrac%7BfF%7D%7B1.5%20%5Ctimes%2010%5E9%7D%29%5E2%7D%2C%20%5Cquad%20f%20%3E%20f_0%0A%5Cend%7Baligned%7D%0A%5Cright.%0A%5Cend%7Balign%7D&id=LyfPl)
设置参数![](https://cdn.nlark.com/yuque/__latex/37c4dc828a58032026858e9c24cc6e4a.svg#card=math&code=%5COmega%20%3D%202%5Cpi%20%5Ctimes%2010%5E6%2C%20%20F%20%5Cin%20%280.003%2C0.05%29%20%5Ctimes%20%5Cfrac%7B%5COmega%7D%7B2%5Cpi%7D&id=taqHX),固定![](https://cdn.nlark.com/yuque/__latex/d4bfca56b9882ddd1e9ef80cd65e43a4.svg#card=math&code=f_0%20%3D%205%20%5Ctimes%2010%5E5Hz%2C%20h_L%20%20%3D%203000%20Hz%2FHz%5E2%2C%20%20h_0%20%3D%20100%20Hz%5E2%2FHz&id=iyqri),作图如下
![F/(Omega/2pi) = 0.0003](https://cdn.nlark.com/yuque/0/2024/png/40467816/1705301657579-81ea15ac-2148-4f84-9b29-ec6a13fa7695.png#averageHue=%23fbfbfb&clientId=u3254b533-af6d-4&from=drop&height=245&id=u66a6d155&originHeight=480&originWidth=640&originalType=binary&ratio=1.25&rotation=0&showTitle=true&size=22126&status=done&style=none&taskId=uf94b145b-aad3-44c6-a331-739cddaa314&title=F%2F%28Omega%2F2pi%29%20%3D%200.0003&width=326 "F/(Omega/2pi) = 0.0003")![F/(Omega/2pi) = 0.05](https://cdn.nlark.com/yuque/0/2024/png/40467816/1705301665598-ad4d5ea4-6dde-459f-810c-70d87228acf2.png#averageHue=%23fcfcfc&clientId=u3254b533-af6d-4&from=drop&height=243&id=u6f785e9f&originHeight=480&originWidth=640&originalType=binary&ratio=1.25&rotation=0&showTitle=true&size=12321&status=done&style=none&taskId=u325602f6-08e0-43ed-baae-50d3cbfc974&title=F%2F%28Omega%2F2pi%29%20%3D%200.05&width=324 "F/(Omega/2pi) = 0.05")
![realnoise1.png](https://cdn.nlark.com/yuque/0/2024/png/40467816/1705301681286-29193662-ba36-4449-9d4a-3a1611b08ccc.png#averageHue=%23fbfbfb&clientId=u3254b533-af6d-4&from=drop&height=326&id=u673b54ec&originHeight=480&originWidth=640&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=20626&status=done&style=none&taskId=ue85a3bcf-72ed-4313-a7a0-5a714a2a1fe&title=&width=434)

设置参数![](https://cdn.nlark.com/yuque/__latex/75c8304b79e1a6db099cef37fff8b1bd.svg#card=math&code=%5COmega%20%3D%202%5Cpi%20%5Ctimes%2010%5E6%2C%20%20f_0%20%5Cin%20%280.5%2C5%29%20%5Ctimes%20%5Cfrac%7B%5COmega%7D%7B2%5Cpi%7D&id=NuqKj),固定:![](https://cdn.nlark.com/yuque/__latex/a5f99354975c889c77e983cc30ea99e2.svg#card=math&code=F%20%3D%2010%5E4Hz%2C%20h_L%20%20%3D%203000%20Hz%2FHz%5E2%2C%20%20h_0%20%3D%20100%20Hz%5E2%2FHz&id=VPn5P),
作图如下
![f0/(Omega/2pi) = 0.5](https://cdn.nlark.com/yuque/0/2024/png/40467816/1705302266050-afab3e04-7631-4c29-84a5-c1f42cae87c5.png#averageHue=%23fbfbfb&clientId=u3254b533-af6d-4&from=drop&height=267&id=ue4f71c27&originHeight=480&originWidth=640&originalType=binary&ratio=1.25&rotation=0&showTitle=true&size=23546&status=done&style=none&taskId=uda147453-f70e-4aa1-a342-134f2e5f14a&title=f0%2F%28Omega%2F2pi%29%20%3D%200.5&width=356 "f0/(Omega/2pi) = 0.5")![f0/(Omega/2pi) = 0.5](https://cdn.nlark.com/yuque/0/2024/png/40467816/1705302271074-73d4b07e-e16a-423c-ad1b-40a270dce3bf.png#averageHue=%23fbfbfb&clientId=u3254b533-af6d-4&from=drop&height=260&id=u963558ce&originHeight=480&originWidth=640&originalType=binary&ratio=1.25&rotation=0&showTitle=true&size=17556&status=done&style=none&taskId=u9655dde9-485e-4374-b43e-cf50ee4a8a8&title=f0%2F%28Omega%2F2pi%29%20%3D%200.5&width=346 "f0/(Omega/2pi) = 0.5")
![Figure_1.png](https://cdn.nlark.com/yuque/0/2024/png/40467816/1705302686714-bc161a1c-0626-44a3-98ae-11cd9121f280.png#averageHue=%23fbfbfb&clientId=u3254b533-af6d-4&from=drop&height=300&id=u895b3fb1&originHeight=480&originWidth=640&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=17731&status=done&style=none&taskId=u1df61814-dd08-4789-9d45-5fab54645fa&title=&width=400)

此外，除了激光本身的相位抖动，原子本身还有随机的速度，速度的概率分布依赖于原子的温度，有如下函数形式:
![](https://cdn.nlark.com/yuque/__latex/003f7d74eb9b717a624ee3e842ff40df.svg#card=math&code=f%28v%29%20%3D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%20%5Cpi%7D%20v_%7Bth%7D%7D%20e%5E%7B%5Cfrac%7Bv%5E2%7D%7B2%20v_%7Bth%7D%5E2%7D%7D&id=fP7So)
由于多普勒效应，随机的速度将给激光带来随机的detuning，我们也可以模拟这个噪声带来的影响
[待续]
# 双光子过程中的噪声
按照同样的方式模拟三能级系统，可以处理拉曼双光子过程
[待续]
此外，还可将中间态decay的影响考虑在内
[待续]
# 附录1：Servo Bump形式的噪声模拟
为了验证代码的正确性，我们复现论文[5]中的模拟。
下面的代码绘制噪声大小随着 servo bump形式噪声中的![](https://cdn.nlark.com/yuque/__latex/e6466bf5c1bcfdfe4a687332379d5c9f.svg#card=math&code=f_g&id=eptBW)参数的变化
```python
from qutip import *
import numpy as np
import matplotlib.pyplot as plt

def S_servo_freq(f,para):
    [h0,sg,fg,sigmag] = para
    res = h0 + (sg *(fg)*(fg))/(np.sqrt(8*np.pi)*sigmag)* np.exp(-((f - fg)**2)/(2*sigmag**2)) 
    #+ hg* np.exp(-((f + fg)**2)/(2*sigmag**2))
    return res

def noise_coeff(t,args):
    phi_vec = args['phi']
    para = args['para']
    [h0,hg,fg,sigmag] = para
    freq_noise = 0
    for i in range(N):
        f = fg - fwidth/2 + deltaf*i
        PSD = S_servo_freq(f,para)
        freq_noise = freq_noise + 2*np.pi*np.sqrt(PSD * deltaf) * np.sin(f*2*np.pi*t + phi_vec[i])
    return freq_noise

Omega = 10**6 * 2* np.pi #Rabi frequency
time_scale =  np.pi/Omega
tau = 2*time_scale # simulation time
Delta = 0 # detuning
ptnum = 2 #time segment


# Sampling phase noise and phase noise parameter.
h0 = 0 #Hz/Hz^2

sigmag = 1400  #sigmag/Omega
hg = 1100
fg0 = 10**6
sg = np.sqrt(8*np.pi) * sigmag * hg /(fg0*fg0)

simnum = 30


fwidth = 4*sigmag #Hz, the bandwidth of power spectrum of noise phase
N = 500 # cut noise bandwidth into N parts
deltaf = fwidth / N



x = np.linspace(0.01,4,40) 
fg_var = Omega * x /(2*np.pi)

plot_list = fg_var
error_list = np.zeros((len(plot_list),simnum))
mean_list = np.zeros(len(plot_list))
dev_list = np.zeros(len(plot_list))

for i in range(len(plot_list)):
    parameter = [h0,sg,plot_list[i],sigmag]

    #defined time-dependent Hamiltonian given sampled noise
    for j in range(simnum):
        phi_vec = 2*np.pi * np.random.random(size = N)
        args = {'phi':phi_vec,'para':parameter}
        H = [0.5 * Omega * sigmax(),[ sigmaz(),noise_coeff]] 
        time = np.linspace(0,tau,ptnum)
        psi0 = basis(2, 0)
        result = sesolve(H, psi0, time, args = args)
        
        #print error
        print(np.abs(result.states[-1][1,0])**2)
        error_list[i,j] = np.log10(np.abs(result.states[-1][1,0])**2)       
    
for i in range(len(plot_list)):
    mean_list[i] = np.mean(error_list[i])
    dev_list[i] = np.std(error_list[i])
print(error_list)

plt.errorbar(x,mean_list,yerr = dev_list, fmt = 'o')
plt.xlabel(r'$fg /(\Omega/2\pi)$')
plt.ylabel(r'$log_{10}(error)$')
plt.title(r'$2\pi$-Palse Error')
plt.show()
```
设定初态处于基态，在一个![](https://cdn.nlark.com/yuque/__latex/832ca5594f56f377286b23d9f12a15c6.svg#card=math&code=%5Cpi-&id=MbnNh)Palse和![](https://cdn.nlark.com/yuque/__latex/a4bb598529cb12fb8704bf86057e66b1.svg#card=math&code=2%5Cpi-&id=OVEZS)Palse过程后，分别衡量末态处于激发态和基态的概率。在无噪声时二者应当严格为0，故这个值可以衡量噪声的大小。
下面是模拟servo bump噪声的结果以及和文献结果的对比。横轴为高斯型噪声的中心，纵轴是噪声的大小（的对数）。上两图是![](https://cdn.nlark.com/yuque/__latex/26d01865f6e830adaa66117e307e316f.svg#card=math&code=%5Cpi&id=m0NQr)Pulse和![](https://cdn.nlark.com/yuque/__latex/db1683fd2f56c5c0fbdcfb071c76d459.svg#card=math&code=2%5Cpi&id=onaq5)Pulse的结果下面两图是对应文献中的结果，可见有类似的形状。模拟代码见上。
![Figure_1.png](https://cdn.nlark.com/yuque/0/2024/png/40467816/1705284121163-7096c447-a8ef-4083-9a36-1d48d2574e67.png#averageHue=%23fbfbfb&clientId=ub53574f0-2a9e-4&from=drop&height=245&id=uc3546c1a&originHeight=480&originWidth=640&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=16762&status=done&style=none&taskId=ua8617a69-12dd-473a-b41e-b7f9ee09f76&title=&width=327)![Figure_2.png](https://cdn.nlark.com/yuque/0/2024/png/40467816/1705284168855-cc942690-fb05-405f-8fda-ba4eaf955dcc.png#averageHue=%23fbfbfb&clientId=ub53574f0-2a9e-4&from=drop&height=263&id=u386fc6a6&originHeight=480&originWidth=640&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=17406&status=done&style=none&taskId=u5389d978-80a7-4f18-8397-6026081778b&title=&width=350)
![45804b3efd53d8a839d364cd35e3e54.png](https://cdn.nlark.com/yuque/0/2024/png/40467816/1705214550584-2f38b00e-b219-4aa3-b902-ae31f27ed22f.png#averageHue=%23f7f0f2&clientId=uad779efb-02c4-4&from=drop&height=248&id=u4a3c90d6&originHeight=317&originWidth=409&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=18951&status=done&style=none&taskId=u5f4e7ee8-6268-4de3-afeb-cd47b881c2b&title=&width=320)![2973a8eaff5c1966e9c47fb41f41ce0.png](https://cdn.nlark.com/yuque/0/2024/png/40467816/1705214559504-14b9e66b-55ae-4003-9872-694b0b5566fb.png#averageHue=%23f6f3f3&clientId=uad779efb-02c4-4&from=drop&height=272&id=u096a1546&originHeight=339&originWidth=379&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=26430&status=done&style=none&taskId=u812ba8f5-f2e6-4ebe-bfdf-3450b8baeb1&title=&width=304)


## 附录2：更高效的噪声拉比振荡模拟方式
用qutip模拟多个时间分段的过程十分耗时，我们考虑直接手撸解薛定谔方程
```python
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
%matplotlib qt5

# Sampling phase noise and phase noise parameter.单位为Hz
Omega = 1e6 * 2* np.pi #Rabi frequency
h0 = 0 #Hz/Hz^2
#单位:Hz
sigmag = 1400  #sigmag/Omega
hg = 1100
fg0 = 1e6
sg = 10*np.sqrt(8*np.pi) * sigmag * hg /(fg0*fg0)

simnum = 5

fwidth = 4*sigmag #Hz, the bandwidth of power spectrum of noise phase
N = 500 # cut noise bandwidth into N parts
deltaf = fwidth / N



def S_servo_freq(f,para):
    [h0,sg,fg,sigmag] = para
    res = h0 + (sg *(fg)*(fg))/(np.sqrt(8*np.pi)*sigmag)* np.exp(-((f - fg)**2)/(2*sigmag**2)) 
    #+ hg* np.exp(-((f + fg)**2)/(2*sigmag**2))
    return res

def noise_coeff(t,args):
    phi_vec = args['phi']
    para = args['para']
    [h0,hg,fg,sigmag] = para
    freq_noise = 0
    for i in range(N):
        f = fg - fwidth/2 + deltaf*i
        PSD = S_servo_freq(f,para)
        freq_noise += 2*np.pi*np.sqrt(PSD * deltaf) * np.sin(f*2*np.pi*t + phi_vec[i])
    return freq_noise

# 1. 定义哈密顿量

def H(t, args):
    sigma_z = np.array([[1, 0], [0, -1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    
    return 0.5 * Omega * sigma_x + noise_coeff(t, args) * sigma_z

# 2. 定义初始状态psi_0为基态
psi_0 = np.array([1, 0])


# 3. 定义时间演化函数evolution(psi,args)
def evolution(psi, H, dt):
    return np.dot(expm(-1j * H * dt), psi)

# 4. 在时间步长下求解薛定谔方程(求解4pi pulse即可)
dt = 0.1/Omega
npulse = 20
t = np.arange(0, npulse*2*np.pi/Omega, dt)
mean_P = np.zeros(len(t))
error_P = np.zeros(len(t))
parameter = [h0,sg,fg0,sigmag] 
for j in range(simnum):
    phi_vec = 2*np.pi * np.random.random(size = N)
    args = {'phi':phi_vec,'para':parameter} 
    psi_t = np.zeros((len(t), 2), dtype=complex)
    psi_t[0] = psi_0
    #求解哈密顿方程
    for i in range(1, len(t)):
        H_delta = H(t[i], args)
        print(H_delta)
        psi_t[i] = evolution(psi_t[i-1], H_delta, dt)
    
    # 5. 画出激发态概率随时间的变化曲线
    P_excited = np.abs(psi_t[:, 1])**2
    plt.plot(t, P_excited, linewidth = 2.0, color = 'r')#, label=f'delta={delta:.2f}')
    mean_P += P_excited
    
mean_P /= simnum
plt.plot(t, mean_P, linewidth = 2.0, color = 'k')
plt.xlabel('Time')
plt.ylabel('Excited state probability')
plt.legend()
plt.show()

```

模拟30次20个![](https://cdn.nlark.com/yuque/__latex/db1683fd2f56c5c0fbdcfb071c76d459.svg#card=math&code=2%5Cpi&id=yPFav)pulse下的拉比振荡图像如下：![2d4a4f219aab046fb13c7a63990a0d9.png](https://cdn.nlark.com/yuque/0/2024/png/41004299/1705305021012-cedc2e34-47b0-4a54-8ef4-8bf9fdb1cc5f.png#averageHue=%23dacaca&clientId=uce7e5149-bcca-4&from=drop&id=u695d1251&originHeight=690&originWidth=903&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=185638&status=done&style=none&taskId=u71780cd1-e02d-4959-b7e1-87560bc543e&title=)
## 
```python
# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
%matplotlib qt5

# Sampling phase noise and phase noise parameter.单位为Hz
Omega = 10**6 * 2* np.pi #Rabi frequency
h0 = 1000 #Hz/Hz^2
h_center = Omega/(2*np.pi)

simnum = 50
fwidth = 0.1*Omega/(2*np.pi) #Hz, the bandwidth of power spectrum of noise phase
N = 500 # cut noise bandwidth into N parts
deltaf = fwidth / N




#Function of PSD. para covers all parameters of this function 
def S_freq(f,para):
    h0 = para
    res = h0 
    return res

#Function generate the trace of the noise. 
# para covers all parameters of PSD function
#sumpara = [f0,fwidth,N] clearify the begin point of summontion f0, the summontion bandwidth fwidth, the fraction number N
def noise_coeff(t,args):
    phase_noise = 0
    phi_vec = args['phi']
    para = args['para']
    sumpara = args['sumpara']
    f0,fwidth,N = sumpara
    deltaf = fwidth / N
    #sum over the band wideth around point f0
    for i in range(N):
        f = f0 - fwidth/2 +deltaf*i
        PSD = S_freq(f,para)
        phase_noise = phase_noise + 2*np.pi*np.sqrt(PSD * deltaf) * np.sin((f)*2*np.pi*t + phi_vec[i])
    return phase_noise

# 1. 定义哈密顿量

def H(t, args):
    sigma_z = np.array([[1, 0], [0, -1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    
    return 0.5 * Omega * sigma_x + noise_coeff(t, args) * sigma_z

# 2. 定义初始状态psi_0为基态
psi_0 = np.array([1, 0])


# 3. 定义时间演化函数evolution(psi,args)
def evolution(psi, H, dt):
    return np.dot(expm(-1j * H * dt), psi)

# 4. 在时间步长下求解薛定谔方程(求解4pi pulse即可)
dt = 0.1/Omega
npulse = 15
t = np.arange(0, npulse*2*np.pi/Omega, dt)
mean_P = np.zeros(len(t))
error_P = np.zeros(len(t))

parameter = h0
sumparameter = [h_center,fwidth,N]
for j in range(simnum):
    phi_vec = 2*np.pi * np.random.random(size = N)
    args = {'phi':phi_vec,'para':parameter,'sumpara':sumparameter}
    psi_t = np.zeros((len(t), 2), dtype=complex)
    psi_t[0] = psi_0
    #求解哈密顿方程
    for i in range(1, len(t)):
        H_delta = H(t[i], args)
        print(H_delta)
        psi_t[i] = evolution(psi_t[i-1], H_delta, dt)
    
    # 5. 画出激发态概率随时间的变化曲线
    P_excited = np.abs(psi_t[:, 1])**2
    plt.plot(t, P_excited, linewidth = 2.0, color = 'r')#, label=f'delta={delta:.2f}')
    mean_P += P_excited
    
mean_P /= simnum
plt.plot(t, mean_P, linewidth = 2.0, color = 'k')
plt.xlabel('Time')
plt.ylabel('Excited state probability')
plt.legend()
plt.show()
```
以下是进行50次模拟15个![](https://cdn.nlark.com/yuque/__latex/db1683fd2f56c5c0fbdcfb071c76d459.svg#card=math&code=2%5Cpi&id=dHXJV) pulse的拉比振荡图(![](https://cdn.nlark.com/yuque/__latex/37a8c5028cf0ff3324db2a1e56a7a8cb.svg#card=math&code=h_%7Bcenter%7D%20%3D%20%5Cfrac%7B%5COmega%7D%7B2%5Cpi%7D&id=y1uss))：
![3882de006813e1c2e34a3f934e1f192.png](https://cdn.nlark.com/yuque/0/2024/png/41004299/1705310617285-2c5a9902-0b78-4c97-9deb-5903024993bf.png#averageHue=%23e1cdcd&clientId=uf9fe7fd5-e984-4&from=drop&id=ua4c260d7&originHeight=675&originWidth=874&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=170919&status=done&style=none&taskId=u10cd9d5d-ec0c-446e-bf77-714fb4be3b8&title=)

# 附录3：总代码
```python
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
        plt.plot(time,p[j],label = r'$|g \rangle$')
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
    main2()
```
# 
## 相关资料
> [1].Pascal Scholl, **Simulation quantique de modèles de spins avec des grandes matrices d’atomes de Rydberg :**[https://theses.hal.science/tel-03523082/](https://theses.hal.science/tel-03523082/)  
> **（Browaeys组的博士论文）**

> [2].Antoine Browaeys et al.,  **Analysis of imperfections in the coherent optical excitation of single atoms to Rydberg states: **[**https://arxiv.org/abs/1802.10424**](https://arxiv.org/abs/1802.10424)

> [3].Harry Levine et al., **High-fidelity control and entanglement of Rydberg atom qubits:**
> [**https://arxiv.org/abs/1806.04682**](https://arxiv.org/abs/1806.04682)

> [4].Jan Hald and Valentina Ruseva,** Efficient suppression of diode-laser phase noise by optical filtering: **[**https://opg.optica.org/josab/abstract.cfm?uri=josab-22-11-2338**](https://opg.optica.org/josab/abstract.cfm?uri=josab-22-11-2338)
> **（理论计算滤波腔抑制相位噪声）**

> [5].X. Jiang, J. Scott, Mark Friesen, and M. Saffman, **Sensitivity of quantum gate fidelity to laser phase and intensity noise**
> [**https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.042611**](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.042611)
> 



