import numpy as np 
import scipy.integrate as si

def jumpcharfunc(zeta,kappabar,delta,lamda,T,typ):
    
    i=complex(0,1)
    
    if typ == 1:
        
        a = 1
        y = 1+kappabar
        
    else:
        
        a = -1
        y = 1
    
    return (np.exp(lamda*T*y*(np.exp(i*(np.log(1+kappabar)
                                        +0.5*a*delta*delta)*zeta
                                     -0.5*delta*delta*zeta*zeta)-1)))


def Hestoncharfunc(zeta,alpha1,V1infty,Lambda1,rho1,V10,alpha2,V2infty,
                   Lambda2,rho2,V20,r_div,T,logS0,typ):
    
    i=complex(0,1)
    
    if typ == 1:
        
        a = 1
        b = 1
        
    else:
        
        a = -1
        b = 0

    BH,AHpart1 = Hestonaux(zeta,alpha1,V1infty,Lambda1,rho1,T,a,b)
    CH,AHpart2 = Hestonaux(zeta,alpha2,V2infty,Lambda2,rho2,T,a,b)
    AH = r_div*zeta*T*i + AHpart1 + AHpart2
    
    return np.exp(AH + BH*V10 + CH*V20 + zeta*logS0*i)


def  Hestonaux(zeta,alpha,Vinfty,Lamda,rho,T,a,b):
    
    i=complex(0,1)
    c = alpha - (b+zeta*i)*rho*Lamda
    d = np.sqrt(c*c - (a*i-zeta)*zeta*Lamda*Lamda)
    g = (c+d)/(c-d)
    
    BH = (c+d)*(np.exp(-d*T)-1)/((Lamda*Lamda)*(np.exp(-d*T)-g))
    AHpart = alpha*Vinfty*((c-d)*T - 2*np.log((np.exp(-d*T)-g)/(1-g)))/(Lamda
                                                                        *Lamda)
    
    return BH,AHpart
    


def integrand(zeta,alpha1,x1infty,Lambda1,rho1,x10,alpha2,
              x2infty,Lambda2,rho2,x20,kappabar,delta,lamda,intPhi,
              r_div,T,logS0,logK,typ):
    
    i=complex(0,1)
    fH  = Hestoncharfunc(zeta,alpha1,x1infty,Lambda1,rho1,x10,alpha2,
                         x2infty,Lambda2,rho2,x20,r_div-lamda*kappabar,
                         T,logS0,typ)
    fB  = jumpcharfunc(zeta,kappabar,delta,lamda,T,typ)
    
    if typ == 1:
      a = 1
      
    else:
      a = -1
    
    fpp = np.exp(0.5*zeta*(a*i-zeta)*intPhi)
    QI  = (np.exp(-i*zeta*logK)*(fH*fpp*fB)/(i*zeta)).real
    
    return QI



def Q(alpha1,x1infty,Lambda1,rho1,x10,alpha2,x2infty,Lambda2,rho2,x20,
      kappabar,delta,lamda,intPhi,r_div,T,logS0,logK,typ):
    
    Q =  0.5 + 1.0/np.pi*si.quad(integrand,0,np.inf,args=(
        alpha1,x1infty,Lambda1,rho1,x10,alpha2,x2infty,Lambda2,rho2,x20,
        kappabar,delta,lamda,intPhi,r_div,T,logS0,logK,typ))[0]
    
    return Q



def Hplusplus(alpha1,x1infty,Lambda1,rho1,x10,alpha2,x2infty,Lambda2,
              rho2,x20,kappabar,delta,lamda,intPhi,r,div,T,S0,K):
    
    Q1    = Q(alpha1,x1infty,Lambda1,rho1,x10,alpha2,x2infty,
              Lambda2,rho2,x20,kappabar,delta,lamda,intPhi,
              r-div,T,np.log(S0),np.log(K),1)
    Q2    = Q(alpha1,x1infty,Lambda1,rho1,x10,alpha2,x2infty,
              Lambda2,rho2,x20,kappabar,delta,lamda,intPhi,r-div,
              T,np.log(S0),np.log(K),2)
    
    call = max(S0*np.exp(-div*T)*Q1 - K*Q2*np.exp(-r*T),0)
    put = max(K*(1-Q2)*np.exp(-r*T)-S0*np.exp(-div*T)*(1-Q1),0)
    
    return call,put























