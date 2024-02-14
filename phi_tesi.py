import math
from scipy.optimize import newton
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def shift(v_0_1,a_1,b_1,v_0_2,a_2,b_2,f_pivoted,S,r):
    
    def implied_volatilities(call_price, S, K, T, r, 
                             initial_volatility_guess=0.2):
    
        def black_scholes_call(sigma):
            
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * 
                                                                 math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
    
            return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2) \
                - call_price
    
        implied_volatility = newton(black_scholes_call, 
                                    initial_volatility_guess)
    
        return implied_volatility
    
    def norm_cdf(x):
    
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    K = np.array(f_pivoted.columns/100)  
    T = f_pivoted.index.astype(float).to_numpy() 
    call_option_prices = f_pivoted.iloc[:,:].astype(float).to_numpy()/100  
    implied_volatilities_list=np.zeros((len(T),len(K)))
    Iphi=np.zeros((len(T),len(K)))
    phi=np.zeros((len(T),len(K)))
    derivative_values=np.zeros((len(T)+1,len(K)))
    
    for i in range(len(T)):
        for j in range(len(K)):
            
            implied_volatilities_list[i,j]= implied_volatilities(
                call_option_prices[i,j], S, K[j], T[i], r)
            ev1=((v_0_1-b_1)/(a_1*T[i]))*(1-np.exp(-a_1*T[i]))+b_1
            ev2=((v_0_2-b_2)/(a_2*T[i]))*(1-np.exp(-a_2*T[i]))+b_2
            ev=ev1+ev2
            Iphi[i,j]=((implied_volatilities_list[i,j])/100-ev/100  )  *T[i]  
    
    Iphi = RegularGridInterpolator((T, K), Iphi,bounds_error=False,
                                   fill_value=None)
    
    for i in range(len(K)):
        
        derivative_values[1:,i] = np.gradient(Iphi((T,K[i])), T)
    
    T=np.insert(T,0,0)
    derivative_values[0,:]=derivative_values[1,:]
    phi = RegularGridInterpolator((T, K), derivative_values,method='linear',
                                  bounds_error=False,fill_value=None)  
    
    return Iphi,phi

    


























