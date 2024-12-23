import numpy as np
import matplotlib.pyplot as plt
import math
from pycbc.types.timeseries import TimeSeries
from pycbc.types.frequencyseries import FrequencySeries
from pycbc.waveform import get_fd_waveform
from pycbc import types



def Inspiral_phase(m1, m2, X1, X2, delta1, delta2, delta3, delta6):
    df = 1/2**29
    A = 1
    f_ini = 5e-5; f_end = 3e-4; sigma0 = 0.0; tc = 0.0; phic = 0.0;
    
    #the information of the system
#     G = 6.6e-11
#     c = 3e8
#     m1 = m1*G*2e30/c**3
#     m2 = m2*G*2e30/c**3
    
    M = m1+m2
    eta = m1*m2/M**2

    X_eff = (m1*X1+m2*X2)/M
    X_PN = X_eff-38*eta/113*(X1+X2)

    #eplison0 confirmed from the continuity
#     sigma0 = 1.5
#     tc = 0.5
#     phic = 0
   
    #PN conefficients
    delta = (m1-m2)/M
    X_s = (X1+X2)/2
    X_a = (X1-X2)/2

    
    phi0 = 1
    phi1 = 0
    phi2 = 3715/756+55*eta/9
    phi3 = -16*np.pi+113*delta*X_a/3+(113/3-76*eta/3)*X_s
    phi4 = 15293365/508032+27145*eta/504+3085*eta**2/72+(-405/8+200*eta)*X_a**2-405/4*delta*X_a*X_s+(-405/8+5*eta/2)*X_s**2
    
    
    #general parameters
    Matrix_p = [[3931.9, -17395.8, 3132.38, 343966.0, -1.21626*10**6, -70698.0, 1.38391*10**6, -3.96628*10**6, -60017.5, 803515.0, -2.09171*10**6],
            [-40105.5, 112253.0, 23561.7, -3.47618*10**6, 1.13759*10**7, 754313.0, -1.30848*10**7, 3.64446*10**7, 596227.0, -7.42779*10**6, 1.8929*10**7],
            [83208.4, -191238.0, -210916, 8.71798*10**6, -2.69149*10**7, -1.98898*10**6, 3.0888*10**7, -8.39087*10**7, -1.4535*10**6, 1.70635*10**7, -4.27487*10**7 ]]

    def p(i):
        p = Matrix_p[i][0]+Matrix_p[i][1]*eta+(X_PN-1)*(Matrix_p[i][2]+Matrix_p[i][3]*eta+Matrix_p[i][4]*eta**2)+\
           (X_PN-1)**2*(Matrix_p[i][5]+Matrix_p[i][6]*eta+Matrix_p[i][7]*eta**2)+\
           (X_PN-1)**3*(Matrix_p[i][8]+Matrix_p[i][9]*eta+Matrix_p[i][10]*eta**2)
        return p

    ###
    Matrix_gamma = [[0.0069274, 0.0302047, 0.00630802, -0.120741, 0.262716, 0.00341518, -0.107793, 0.27099, 0.000737419, -0.0274962, 0.0733151],
                [1.01034, 0.000899312, 0.283949, -4.04975, 13.2078, 0.103963, -7.02506, 24.7849, 0.030932, -2.6924, 9.60937],
                [1.30816, -0.00553773, -0.0678292, -0.668983, 3.40315, -0.0529658, -0.992379, 4.82068, -0.00613414, -0.384293, 1.75618]]

    def gamma(i):
        gamma = Matrix_gamma[i][0]+Matrix_gamma[i][1]*eta+(X_PN-1)*(Matrix_gamma[i][2]+Matrix_gamma[i][3]*eta+Matrix_gamma[i][4]*eta**2)+\
              (X_PN-1)**2*(Matrix_gamma[i][5]+Matrix_gamma[i][6]*eta+Matrix_gamma[i][7]*eta**2)+\
              (X_PN-1)**3*(Matrix_gamma[i][8]+Matrix_gamma[i][9]*eta+Matrix_gamma[i][10]*eta**2)
        return gamma

    ###
    Matrix_sigma = [[2096.55, 1463.75, 1312.55, 18307.3, -43534.1, -833.289, 32047.3, -108609, 452.251, 8353.44, -44531.3],
                [-10114.1, -44631, -6541.31, -266959, 686328.0, 3405.64, -437508, 1.63182*10**6, -7462.65, -114585, 674402.0],
                [22933.7, 230960.0, 14961.1, 1.19402*10**6, -3.10422*10**6, -3038.17, 1.87203*10**6, -7.30915*10**6, 42738.2, 467502., -3.06485*10**6],
                [-14621.7, -377813, -9608.68, -1.71089*10**6, 4.33292*10**6, -22366.7, -2.50197*10**6, 1.02745*10**7, -85360.3, -570025, 4.39684*10**6]]

    def sigma(i):
        sigma = Matrix_sigma[i][0]+Matrix_sigma[i][1]*eta+(X_PN-1)*(Matrix_sigma[i][2]+Matrix_sigma[i][3]*eta+Matrix_sigma[i][4]*eta**2)+\
              (X_PN-1)**2*(Matrix_sigma[i][5]+Matrix_sigma[i][6]*eta+Matrix_sigma[i][7]*eta**2)+\
              (X_PN-1)**3*(Matrix_sigma[i][8]+Matrix_sigma[i][9]*eta+Matrix_sigma[i][10]*eta**2)
        return sigma

    ###
    Matrix_beta = [[97.8975, -42.6597, 153.484, -1417.06, 2752.86, 138.741, -1433.66, 2857.74, 41.0251, -423.681, 850.359],
               [-3.2827, -9.05138, -12.4154, 55.4716, -106.051, -11.953, 76.807, -155.332, -3.41293, 25.5724, -54.408],
               [-2.51564*10**(-5), 1.97503*10**(-5), -1.83707*10**(-5), 2.18863*10**(-5), 8.25024*10**(-5), 7.15737*10**(-6), -5.578*10**(-5), 1.91421*10**(-4), 5.44717*10**(-6), -3.22061*10**(-5), 7.97402*10**(-5)]]

    def beta(i):
        beta = Matrix_beta[i][0]+Matrix_beta[i][1]*eta+(X_PN-1)*(Matrix_beta[i][2]+Matrix_beta[i][3]*eta+Matrix_beta[i][4]*eta**2)+\
             (X_PN-1)**2*(Matrix_beta[i][5]+Matrix_beta[i][6]*eta+Matrix_beta[i][7]*eta**2)+\
             (X_PN-1)**3*(Matrix_beta[i][8]+Matrix_beta[i][9]*eta+Matrix_beta[i][10]*eta**2)
        return beta
 
    ###
    Matrix_alpha = [[43.3151, 638.633, -32.8577, 2415.89, -5766.88, -61.8546, 2953.97, -8986.29, -21.5714, 981.216, -3239.57],
                [-0.0702021, -0.162698, -0.187251, 1.13831, -2.83342, -0.17138, 1.71975, -4.53972, -0.0499834, 0.606207, -1.68277],
                [9.59881, -397.054, 16.2021, -1574.83, 3600.34, 27.0924, -1786.48, 5152.92, 11.1757, -577.8, 1808.73],
                [-0.0298949, 1.40221, -0.0735605, 0.833701, 0.224001, -0.0552029, 0.566719, 0.718693, -0.0155074, 0.157503, 0.210768],
                [0.997441, -0.00788445, -0.0590469, 1.39587, -4.51663, -0.0558534, 1.75166, -5.99021, -0.0179453, 0.59651, -2.06089]]
    def alpha(i):
        alpha = Matrix_alpha[i][0]+Matrix_alpha[i][1]*eta+(X_PN-1)*(Matrix_alpha[i][2]+Matrix_alpha[i][3]*eta+Matrix_alpha[i][4]*eta**2)+\
              (X_PN-1)**2*(Matrix_alpha[i][5]+Matrix_alpha[i][6]*eta+Matrix_alpha[i][7]*eta**2)+\
              (X_PN-1)**3*(Matrix_alpha[i][8]+Matrix_alpha[i][9]*eta+Matrix_alpha[i][10]*eta**2)
        return alpha
    
    ### 采样需要如下，否则采样点不够，波形结果有问题 ###
    # df    = 1/2**28
    # f_end = 10**-2
    # f_ini = 10**-4
    N_f   = int((f_end-f_ini)/df)
    N_add1 = int((f_ini)/df)
    N_add2 = int((2-f_ini-N_f*df)/df)
    f = np.arange(f_ini, f_ini+N_f*df, df)

    phi5 = (1+np.log(np.pi*M*f))*(38645*np.pi/756-65*np.pi*eta/9+delta*(-732985/2268-140*eta/9)*X_a+(-732985/2268+24260*eta/81+340*eta**2/9)*X_s)
    phi6 = 11583231236531/4694215680-6848*np.euler_gamma/21-640*np.pi*np.pi/3+(-15737765635/3048192+2255*np.pi*np.pi/12)*eta+76055*eta**2/1728-127825*eta**3/1296-6848/63*np.log(64*np.pi*M*f)+2270/3*np.pi*delta*X_a+(2270*np.pi/3-520*np.pi*eta)*X_s
    phi7 = 77096675*np.pi/254016+378515*np.pi*eta/1512-74045*np.pi*eta**2/756+delta*(-25150083775/3048192+26804935*eta/6048-1985*eta**2/48)*X_a+(-25150083775/3048192+10566655595*eta/762048-1042165*eta**2/3024+5345*eta**3/36)*X_s
    
    u = eta**(3/5)*np.pi*(M)*f
    Phi_KRZ_1 = 75/8*u**(-1/3)*eta**(-4/5)*delta1
    Phi_KRZ_2 = -85/(3*eta)*(1+np.log(u))*delta2
    
    theta = np.pi/4
    Phi_KRZ_3 = -(48*2**(-1/3)*M**(5/3)/(eta*np.pi))*delta3*(np.cos(theta))**2*f**(5/3)
    Phi_KRZ_6 = -40*M/eta*(delta6*(np.cos(theta))**2*f**(-1))
    
    Phi_TF2 = 2*np.pi*f*tc-phic-np.pi/4+3/(128*eta)*(np.pi*f*M)**(-5/3)*(phi0+phi1*(np.pi*f*M)**(1/3)+phi2*(np.pi*f*M)**(2/3)+phi3*(np.pi*f*M)**(3/3)+phi4*(np.pi*f*M)**(4/3)+phi5*(np.pi*f*M)**(5/3)+phi6*(np.pi*f*M)**(6/3)+phi7*(np.pi*f*M)**(7/3))
    Phi_inspiral = Phi_TF2+1/eta*(sigma0+sigma(0)*f+3/4*sigma(1)*f**(4/3)+3/5*sigma(2)*f**(5/3)+1/2*sigma(3)*f**2)+Phi_KRZ_1+Phi_KRZ_2+Phi_KRZ_3+Phi_KRZ_6
    h_f1 = A*f**(-7/6)*np.exp(-1j*Phi_inspiral)

    
    return h_f1


    
def save_inspiral_data(h_f1, n_beg=500, n_end=150, output_file="inspiral.txt"):

    # 创建 FrequencySeries 对象
    inarr1 = FrequencySeries(h_f1, delta_f=1/2**29)

    # 转换为 TimeSeries 对象
    outarr1 = inarr1.to_timeseries()

    # 获取数据长度
    n = len(outarr1)

    # 保存选定范围的数据到文件
    np.savetxt(output_file, outarr1[n - n_beg:n - n_end])

    
def Interme_phase(m1, m2, X1, X2, delta1, delta2, delta3, delta6):
    f_ini = 3.8e-3; f_end = 1e-1; beta0 = 0.0; 
    
    df = 1/2**26
    A = 1
    pi = 3.14159

    M = m1+m2
    miu = m1*m2/(M)
    eta = miu/M

    X_eff = (m1*X1+m2*X2)/M
    X_PN = X_eff-38*eta/113*(X1+X2)
    
    #eplison0 confirmed from the continuity
    #beta0 = 0
    
    ###
    Matrix_beta = [[97.8975, -42.6597, 153.484, -1417.06, 2752.86, 138.741, -1433.66, 2857.74, 41.0251, -423.681, 850.359],
               [-3.2827, -9.05138, -12.4154, 55.4716, -106.051, -11.953, 76.807, -155.332, -3.41293, 25.5724, -54.408],
               [-2.51564*10**(-5), 1.97503*10**(-5), -1.83707*10**(-5), 2.18863*10**(-5), 8.25024*10**(-5), 7.15737*10**(-6), -5.578*10**(-5), 1.91421*10**(-4), 5.44717*10**(-6), -3.22061*10**(-5), 7.97402*10**(-5)]]

    def beta(i):
        beta = Matrix_beta[i][0]+Matrix_beta[i][1]*eta+(X_PN-1)*(Matrix_beta[i][2]+Matrix_beta[i][3]*eta+Matrix_beta[i][4]*eta**2)+\
             (X_PN-1)**2*(Matrix_beta[i][5]+Matrix_beta[i][6]*eta+Matrix_beta[i][7]*eta**2)+\
             (X_PN-1)**3*(Matrix_beta[i][8]+Matrix_beta[i][9]*eta+Matrix_beta[i][10]*eta**2)
        return beta
 
    ### first
    # delta1 = 0.0
    
    ### 采样需要如下，否则采样点不够，波形结果有问题 ###
    # df    = 1/2**28
    # f_end = 10**-2
    # f_ini = 10**-4
    N_f   = int((f_end-f_ini)/df)
    f = np.arange(f_ini, f_ini+N_f*df, df)

    u = eta**(3/5)*np.pi*(M)*f
    Phi_KRZ_1 = -75/8*u**(-1/3)*eta**(-4/5)*delta1
    Phi_KRZ_2 = -85/(3*eta)*(1+np.log(u))*delta2
    #######################
#     theta = np.pi/4
    theta = 0.1
    #######################
    
    Phi_KRZ_3 = -(48*2**(-1/3)*M**(5/3)/(eta*np.pi))*delta3*(np.cos(theta))**2*f**(5/3)
    Phi_KRZ_6 = -(40*M/eta)*(delta6*(np.cos(theta))**2*f**(-1))
    
    
    phi_int = 1/eta*(beta0+beta(0)*f+beta(1)*np.log(f)-beta(2)/3*f**(-3))+Phi_KRZ_1+Phi_KRZ_2+Phi_KRZ_3+Phi_KRZ_6
    
    h_f1 = A*f**(-7/6)*np.exp(-1j*phi_int)
    return h_f1


def save_Interme_data(h_f1, n_beg=2000, n_end=10, output_file="Intermediate.txt"):

    # 创建 FrequencySeries 对象
    inarr1 = FrequencySeries(h_f1, delta_f=1/2**26)

    # 转换为 TimeSeries 对象
    outarr1 = inarr1.to_timeseries()

    # 获取数据长度
    n = len(outarr1)

    # 保存选定范围的数据到文件
    np.savetxt(output_file, outarr1[n - n_beg:n - n_end])

    
def run1():
    output_file='h_be.txt'

    # 加载数据
    h_int = np.loadtxt("intermediate.txt")

    # 数据长度
    n_int = len(h_int)

    # 创建时间轴
    t2 = np.linspace(0, n_int, n_int)



    # 保存合并信号到文件
    np.savetxt(output_file, np.column_stack((t2, h_int)))

def run2():
    a = np.loadtxt("h_be.txt")
    t_in = a[:,0]
    h_in = a[:,1]

    h_RD_txt = np.loadtxt("ringdown.txt")
    t_RD = h_RD_txt[:,0][::-1]
    h_RD = h_RD_txt[:,1][::-1]

    # 调整数据间隔：inspiral+intermediate
    max_index_file1 = np.argmax(h_in)  # 最大值位置
    min_index_file1 = np.argmin(h_in)  # 最小值位置
    position_diff_ins = abs(t_in[max_index_file1] - t_in[min_index_file1])

    # 调整数据间隔：ringdown
    max_index_file2 = np.argmax((h_RD))  # 最大值位置
    min_index_file2 = np.argmin((h_RD))  # 最小值位置
    position_diff_int = abs(t_RD[max_index_file2] - t_RD[min_index_file2])

    diff = position_diff_int/position_diff_ins

    #调整数值大小一致(极值)
    h_max_ins = np.max(h_in)
    h_max_int = np.max(h_RD)
    h_change  = h_max_int/h_max_ins


    # 调整时间轴
    t1_scaled = t_in * diff 
    t2_scaled = t_RD

    t = np.append(t1_scaled, np.max(t1_scaled)+t2_scaled)
    h = np.append(h_in*h_change, h_RD)

    np.savetxt('h_all.txt', np.column_stack((t,h)))