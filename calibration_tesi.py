import pandas as pd
import numpy as np
from pricing_fct_calibration_tesi import Bates2pp
from scipy.optimize import minimize


def pmts():
    
    data=pd.read_excel('option_eurusd.xlsx')
    data=data.drop(['Bid','Ask','Ticker','IVM','Volm'],axis=1)
    data=data.drop([0,17,34,51,68,84,67,35,18,1])
    arr1=np.ones(15)*22/365
    arr2=np.ones(15)*51/365
    arr3=np.ones(15)*79/365
    arr4=np.ones(15)*171/365
    arr5=np.ones(15)*263/365
    arr1=arr1.tolist()
    arr2=arr2.tolist()
    arr3=arr3.tolist()
    arr4=arr4.tolist()
    arr5=arr5.tolist()
    arr=arr1+arr2+arr3+arr4+arr5
    data['maturity']=arr
    data['Strike']=data['Strike'].astype(float)*100
    data['Last']=data['Last'].astype(float)*100
    data['maturity']=data['maturity'].astype(float)
    f_pivoted = data.pivot(index='maturity', columns='Strike', values='Last')
    S0=1.096
    rd =  np.log(1.03925)
    rf = np.log(1.0525)
    K =(f_pivoted.columns.astype(int).to_numpy())/100
    tau=f_pivoted.index.astype(float).to_numpy()
    P = data['Last'].to_numpy('float')/100
    params = {"alpha1": {"x0": 0.3, "lbub": [0,5]},
              "x1infty": {"x0": 0.05, "lbub": [0,0.1]},
              "Lambda1": {"x0": 0.03, "lbub": [0,1]},
              "rho1": {"x0": -0.2, "lbub": [-1,0]},
              "x10": {"x0": 0.1, "lbub": [0,0.1]},
              "alpha2": {"x0": 0.3, "lbub": [0,5]},
              "x2infty": {"x0": 0.05, "lbub": [0,0.1]},
              "Lambda2": {"x0": 0.03, "lbub": [0,1]},
              "rho2": {"x0": -0.1, "lbub": [-1,0]},
              "x20": {"x0": 0.1, "lbub": [0,0.1]},
              "kappabar": {"x0": 0.1, "lbub": [0,0.1]},
              "delta": {"x0": 0.3, "lbub": [0,0.1]},
              "lamda": {"x0": 0.05, "lbub": [0,0.1]},
              "intPhi": {"x0": 0.01, "lbub": [0,0.3]},}
    x0 = [param["x0"] for key, param in params.items()]
    bnds = [param["lbub"] for key, param in params.items()]

    def SqErr(x):
        
        alpha1,x1infty,Lambda1,rho1,x10,alpha2,x2infty,Lambda2,rho2,\
            x20,kappabar,delta,lamda,intPhi= [param for param in x]
        mod=np.zeros(len(P))
        ind=0
    
        for j in range(len(tau)):
            for i in range(len(K)):
                
                mod[ind]=Bates2pp(alpha1,x1infty,Lambda1,rho1,x10,alpha2,
                                  x2infty ,Lambda2,rho2,x20,kappabar,delta,
                                  lamda,intPhi, rd,rf,tau[j],S0, K[i])
                ind=ind+1
        err = np.sum( (P-mod)**2 /len(P) )
    
        return err 
    
    result = minimize(SqErr, x0, tol = 1e-4, method='SLSQP', 
                      options={'maxiter': 1e4, 'disp':True }, bounds=bnds)
    alpha1,x1infty,Lambda1,rho1,x10,alpha2,x2infty,Lambda2,rho2,\
        x20,kappabar,delta,lamda,intPhi=result['x']
        
    return alpha1,x1infty,Lambda1,rho1,x10,alpha2,x2infty,Lambda2,rho2,\
        x20,kappabar,delta,lamda,intPhi,f_pivoted
        
        























