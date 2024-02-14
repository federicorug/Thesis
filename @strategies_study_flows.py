import numpy as np 
from characteristic_fct import  Hplusplus
from phi_copia import shift
from calibration_thesis_copia import pmts
import matplotlib.pyplot as plt

## DATA

s_0=1.096
rd =  np.log(1.03925)
rf = np.log(1.0525)
r_0=rd-rf
T=0.7
n_sim=100
accuracy=360
dt=T/accuracy
t=np.arange(0,T+dt,dt)

#initialization phi

a_1,b_1,sigma_vol_1,rho_1,v_0_1,a_2,b_2,sigma_vol_2,rho_2,v_0_2,k_j,\
    sigma_j,lamda,fi,f_pivoted=pmts()
Iphi,phi=shift(v_0_1,a_1,b_1,v_0_2,a_2,b_2,f_pivoted,s_0,r_0)

#poisson process

k_=np.exp(np.random.normal(np.log(1+k_j)- 0.5 * sigma_j**2,sigma_j,
                           size=(len(t),n_sim)))-1
jumps = np.random.poisson(lamda * dt,size=(len(t),n_sim))
jumps=jumps * k_

#CIR 1

rv1=np.random.normal(size=(len(t),n_sim))
cir1_sim=np.ones((len(t),n_sim))
cir1_sim[0,:]=v_0_1

#CIR 2

rv2=np.random.normal(size=(len(t),n_sim))
cir2_sim=np.ones((len(t),n_sim))
cir2_sim[0,:]=v_0_2

#contractual data


strike_s=np.arange(0.8,1.3,0.05)
fixings=[0,2/12,4/12,6/12,8/12]
leverage=1.5
target=0.45
tarn_superrep=0
sim_s=np.ones_like(cir2_sim)

#feller condition

assert 2 * a_1*b_1 > sigma_vol_1**2  
assert 2 * a_2*b_2 > sigma_vol_2**2 
dyn_hedging0=[]
dyn_hedging1=[]
dyn_hedging2=[]
dyn_hedging3=[]

superrep0=[]
superrep1=[]
superrep2=[]
superrep3=[]

payoff0=np.zeros(len(strike_s))
payoff1=[]
payoff2=[]
payoff3=[]
payoff4=[]
## PRICING SUPERREPLICATION

super_prezzo=[]

for strike in strike_s: 
    strike=round(strike,6)
    np.random.seed(seed=12)
    sim_s[0,:]=s_0
    tarn_superrep=0
        
    for fixing in fixings[1:]:
        
        low= Hplusplus(a_1,b_1,sigma_vol_1,rho_1,v_0_1,a_2,b_2,sigma_vol_2,
                       rho_2,v_0_2,k_j,sigma_j,lamda,Iphi((fixing,strike)),rf,rd,
                       fixing,s_0,strike)
        c_low=low[0]
        p_low=low[1]
        high= Hplusplus(a_1,b_1,sigma_vol_1,rho_1,v_0_1,a_2,b_2,sigma_vol_2,rho_2,
                        v_0_2,k_j,sigma_j,lamda,Iphi((fixing,strike)),rf,rd,
                        fixing,s_0,strike+target)
        c_high=high[0]
        tarn_superrep=p_low*leverage+c_high-c_low+tarn_superrep
    
    supdist=tarn_superrep
    super_prezzo.append(tarn_superrep)
    
    ## DYNAMIC HEDGING CASH FLOW
    
    for j in range(n_sim):
        for i in range(1,len(t)):
            
            cir1_sim[i,j]=abs(cir1_sim[i-1,j]+a_1*(b_1-cir1_sim[i-1,j])
                              *dt+sigma_vol_1*np.sqrt(cir1_sim[i-1,j])*rv1[i,j]
                              *np.sqrt(dt))
            cir2_sim[i,j]=abs(cir2_sim[i-1,j]+a_2*(b_2-cir2_sim[i-1,j])*dt
                              +sigma_vol_2*np.sqrt(cir2_sim[i-1,j])*rv2[i,j]*
                              np.sqrt(dt))
    
    
    corr_1=rho_1#*np.sqrt(cir2_sim/(cir2_sim+phi((TT,strike))))
    corr_2=rho_2
    wiener_12=np.sqrt(dt)*(rv1*corr_1+(1-corr_1)*
                           np.random.normal(size=(len(t),n_sim)))
    wiener_22=np.sqrt(dt)*(rv2*corr_2+(1-corr_2)*
                           np.random.normal(size=(len(t),n_sim)))
    
    for j in range(n_sim):
        for i in range(1,len(t)):
            
            u=phi((t[i],strike))
            sim_s[i,j]=sim_s[i-1,j]+(r_0-k_j*lamda)*sim_s[i-1,j]\
            *dt+np.sqrt(cir1_sim[i,j]+u)*sim_s[i-1,j]*wiener_12[i,j]\
            +np.sqrt(cir2_sim[i,j])*sim_s[i-1,j]*wiener_22[i,j]+jumps[i,j]\
            *sim_s[i-1,j]      
    
    payoff_mat=np.zeros_like(sim_s)
    target_payoff=np.zeros_like(sim_s[int(fixings[2]),:])+target
    
    for i in range(1,len(fixings)): 
        for j in range(len(sim_s[3,:])):
            
            if target_payoff[j]>0:
                if sim_s[int(fixings[i]*accuracy),j]>strike:
                    
                    payoff_mat[int(fixings[i]*accuracy),j]= (
                        sim_s[int(fixings[i]*accuracy),j]-strike)
                    
                    if target_payoff[j]-\
                        (sim_s[int(fixings[i]*accuracy),j]-strike)<0:
                            
                        payoff_mat[int(fixings[i]*accuracy),j]=  target_payoff[j]               
                    target_payoff[j]=target_payoff[j]\
                        -(sim_s[int(fixings[i]*accuracy),j]-strike)
                        
                else:
                    
                    payoff_mat[int(fixings[i]*accuracy),j]= (
                        sim_s[int(fixings[i]*accuracy),j]-strike)*leverage
                    
    
    payoff_mat=np.sum(payoff_mat,axis=1)/n_sim
    payoff_mat=payoff_mat[payoff_mat!=0]
    
    if len(np.array(payoff_mat))<1:
        payoff1.append(0)
        
    else:
        payoff1.append(float(payoff_mat[0]))
    

    if len(np.array(payoff_mat))<2:
        payoff2.append(0)
        
    else:
        payoff2.append(float(payoff_mat[1]))
  
    
    if len(np.array(payoff_mat))<3:
        payoff3.append(0)
        
    else:
        payoff3.append(float(payoff_mat[2]))

    if len(np.array(payoff_mat))<4:
        payoff4.append(0)
        
    else:
        payoff4.append(float(payoff_mat[3]))
                    
    disc_matrix = np.exp(-r_0*t)
    disc_matrix=np.atleast_2d(disc_matrix).T
    disc_matrix=np.repeat(disc_matrix, n_sim ,axis=1)
    payoff_mat=payoff_mat*disc_matrix
    payoff_mat=np.sum(payoff_mat,axis=1)/n_sim
    payoff1.append(sum(payoff_mat))
     
    ## DEHEDGING THE SUPERREPLICATION and DYNAMIC HEDGE
    
    targ=sim_s-strike
    targ[targ<0]=0
    new_targ=np.zeros_like(targ)
    targ[0,:]=0
    
    for i in range(1,len(fixings)):
        for j in range(n_sim):
            
            new_targ[int(fixings[i]*accuracy),j]=targ[
                int(fixings[i]*accuracy),j] + new_targ[int(fixings[i-1]*
                                                           accuracy),j]   
                                                           
            if new_targ[int(fixings[i]*accuracy),j]>=target:  
                
                new_targ[int(fixings[i]*accuracy),j]=target
                
    call_dehedge_mat=np.zeros((len(t),n_sim))    
    count=np.zeros(n_sim)
    
    for j in range(len(sim_s[3,:])):
        for i in range(1,len(fixings)): 
            
            if new_targ[int(fixings[i-1]*accuracy),j]==target:
                
                continue
            
            for date in range(i,len(fixings)-1):
                
                if  sim_s[int(fixings[i]*accuracy),j]> strike :
                    
                    strike_hi=strike+target-new_targ[int(fixings[i-1]*accuracy),j]
                    
                    c_higer= Hplusplus(
                        a_1,b_1,sigma_vol_1,rho_1,cir1_sim[int(fixings[i]*accuracy)
                        ,j],a_2,b_2,sigma_vol_2,rho_2,cir2_sim[int(fixings[i]
                        *accuracy),j],k_j,sigma_j,lamda,
                        Iphi((T,strike_hi))-Iphi((fixings[date],strike_hi)),rf,rd,
                        fixings[date+1],sim_s[int(fixings[i]*accuracy),j],strike_hi)
                                                                   
                    c_lower= Hplusplus(a_1,b_1,sigma_vol_1,rho_1,cir1_sim[
                        int(fixings[i]*accuracy),j],a_2,b_2,sigma_vol_2,rho_2,
                        cir2_sim[int(fixings[i]*accuracy),j],k_j,sigma_j,lamda,
                        Iphi((T,strike))-Iphi((fixings[date],strike)),
                        rf,rd,fixings[date+1],sim_s[int(fixings[i]*accuracy),j],
                        strike+target-new_targ[int(fixings[i]*accuracy),j])
                    
                    call_dehedge_mat[int(fixings[i]*accuracy),j]=call_dehedge_mat[
                        int(fixings[i]*accuracy),j]+c_lower[0]-c_higer[0]
                    
                    if new_targ[int(fixings[i]*accuracy),j]==target and \
                        new_targ[int(fixings[i-1]*accuracy),j]!=target:
                            
                        p= Hplusplus(a_1,b_1,sigma_vol_1,rho_1,cir1_sim[
                            int(fixings[i]*accuracy),j],a_2,b_2,sigma_vol_2,rho_2,
                            cir2_sim[int(fixings[i]*accuracy),j],k_j,sigma_j,lamda,
                            Iphi((T,strike))-Iphi((fixings[date],strike)),rf,rd,
                            fixings[date+1],sim_s[int(fixings[i]*accuracy
                                                               ),j],strike)[1]
                        
                        call_dehedge_mat[int(fixings[i]*accuracy),j]=\
                            call_dehedge_mat[int(fixings[i]*accuracy),j]-leverage*p  
    
    de_hedge=np.sum(call_dehedge_mat,axis=1)/n_sim


    de_hedge=de_hedge[de_hedge>0]
    if len(np.array(de_hedge))<1:
        superrep1.append(0)
        
    else:
        superrep1.append(-float(de_hedge[0]))

    if len(np.array(de_hedge))<2:
        superrep2.append(0)
        
    else:
        superrep2.append(-float(de_hedge[1]))

    
    if len(np.array(de_hedge))<3:
        superrep3.append(0)
        
    else:
        superrep3.append(-float(de_hedge[2]))
    
    dyn_hed=np.zeros_like(targ)    
    
    for i in range(1,len(fixings)-1): 
        for j in range(len(sim_s[3,:])):
            if new_targ[int(fixings[i-1]*accuracy),j]==target :
                continue
            else:
                c_hi= Hplusplus(a_1,b_1,sigma_vol_1,rho_1,cir1_sim[int(fixings[i]*
                    accuracy),j],a_2,b_2,sigma_vol_2,rho_2,cir2_sim[int(fixings[i]*
                    accuracy),j],k_j,sigma_j,lamda,Iphi((T,strike))-Iphi((fixings[i]
                    ,strike)),rf,rd,fixings[i+1],sim_s[int(fixings[i]*accuracy),j],
                    strike+target-new_targ[int(fixings[i]
                                                                    *accuracy),j])
                                                                        
                oth= Hplusplus(a_1,b_1,sigma_vol_1,rho_1,cir1_sim[int(fixings[i]*
                    accuracy),j],a_2,b_2,sigma_vol_2,rho_2,cir2_sim[int(fixings[i]*
                    accuracy),j],k_j,sigma_j,lamda,Iphi((T,strike))-Iphi((fixings[i]
                    ,strike)),rf,rd,fixings[i+1],sim_s[int
                                                (fixings[i]*accuracy),j],strike)
                                                                               
                c_l=oth[0]
                p_l=oth[1]
                dyn_hed[int(fixings[i]*accuracy),j]=+c_hi[0]+p_l*leverage-c_l
                
    dyn_hed[0,:]=+ Hplusplus(a_1,b_1,sigma_vol_1,rho_1,v_0_1,a_2,b_2,sigma_vol_2,
            rho_2,v_0_2,k_j,sigma_j,lamda,Iphi((T,strike)),rf,rd,fixings[1],s_0,
            strike+target)[0]
    
    dyn_hed[0,:]=dyn_hed[0,:]- Hplusplus(a_1,b_1,sigma_vol_1,rho_1,v_0_1,a_2,b_2,
       sigma_vol_2,rho_2,v_0_2,k_j,sigma_j,lamda,Iphi((T,strike)),rf,rd,
       fixings[1],s_0,strike)[0]
    
    dyn_hed[0,:]=dyn_hed[0,:]+leverage* Hplusplus(a_1,b_1,sigma_vol_1,rho_1,v_0_1,
        a_2,b_2,sigma_vol_2,rho_2,v_0_2,k_j,sigma_j,lamda,Iphi((T,strike)),rf,
        rd,fixings[1],s_0,strike)[1]
    

    dyn=np.sum(dyn_hed,axis=1)/n_sim
    dyn=dyn[dyn!=0]
    if len(np.array(dyn))<1:
        dyn_hedging0.append(0)
        
    else:
        dyn_hedging0.append(float(dyn[0]))

    if len(np.array(dyn))<2:
        dyn_hedging1.append(0)
        
    else:
        dyn_hedging1.append(float(dyn[1]))

    if len(np.array(dyn))<3:
        dyn_hedging2.append(0)
        
    else:
        dyn_hedging2.append(float(dyn[2]))

    if len(np.array(dyn))<4:
        dyn_hedging3.append(0)
        
    else:
        dyn_hedging3.append(float(dyn[3]))
    

fig,ax=plt.subplots(facecolor=".7",figsize=(10, 5),sharex=True)


ax.plot(strike_s,np.array(superrep0),'C1',label='de-hedged super-replication')
ax.plot(strike_s,dyn_hedging0,'deepskyblue',label='dynamic replication')

ax.plot(strike_s,np.zeros_like(dyn_hedging0),color='r',linestyle='--')

ax.set_title('TARF replication strategy flows in t0',fontsize=16)
ax.grid(visible=1,linewidth=0.4,linestyle='--')

ax.set_xlabel('strike')
ax.set_ylabel('flow')
plt.legend()

plt.show()



fig,ax=plt.subplots(facecolor=".7",figsize=(10, 5),sharex=True)


ax.plot(strike_s,-np.array(superrep1),'C1',label='de-hedged super-replication')
ax.plot(strike_s,dyn_hedging1,'deepskyblue',label='dynamic replication')

ax.plot(strike_s,np.zeros_like(dyn_hedging0),color='r',linestyle='--')

ax.set_title('TARF replication strategy flows in t1',fontsize=16)
ax.grid(visible=1,linewidth=0.4,linestyle='--')

ax.set_xlabel('strike')
ax.set_ylabel('flow')
plt.legend()

plt.show()


fig,ax=plt.subplots(facecolor=".7",figsize=(10, 5),sharex=True)


ax.plot(strike_s,-np.array(superrep2),'C1',label='de-hedged super-replication')
ax.plot(strike_s,dyn_hedging2,'deepskyblue',label='dynamic replication')

ax.plot(strike_s,np.zeros_like(dyn_hedging0),color='r',linestyle='--')

ax.set_title('TARF replication strategy flows in t2',fontsize=16)
ax.grid(visible=1,linewidth=0.4,linestyle='--')

ax.set_xlabel('strike')
ax.set_ylabel('flow')
plt.legend()

plt.show()

fig,ax=plt.subplots(facecolor=".7",figsize=(10, 5),sharex=True)


ax.plot(strike_s,-np.array(superrep3),'C1',label='de-hedged super-replication')
ax.plot(strike_s,dyn_hedging3,'deepskyblue',label='dynamic replication')

ax.plot(strike_s,np.zeros_like(dyn_hedging0),color='r',linestyle='--')

ax.set_title('TARF replication strategy flows in t3',fontsize=16)
ax.grid(visible=1,linewidth=0.4,linestyle='--')

ax.set_xlabel('strike')
ax.set_ylabel('flow')
plt.legend()

plt.show()










