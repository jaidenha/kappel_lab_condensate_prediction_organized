#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math


def get_kappa(seq, type1, type2, threshold=0.05, blob_size=5):
    # Check frequency threshold
    freq1 = sum(seq.count(res) for res in type1) / len(seq)
    freq2 = sum(seq.count(res) for res in type2) / len(seq)
    if freq1 <= threshold or freq2 <= threshold:
        return 0.0

    kappab = []
    b = blob_size
    count1 = sum(seq.count(res) for res in type1)
    count2 = sum(seq.count(res) for res in type2)
    sigAll = ((count1/len(seq)) - (count2/len(seq)))**2 / ((count1/len(seq)) + (count2/len(seq)))

    sigX = []
    for x in range(0, len(seq) - b + 2):
        subseq = seq[x:x+b]
        c1 = sum(subseq.count(res) for res in type1)
        c2 = sum(subseq.count(res) for res in type2)
        if c1 + c2 == 0:
            sigX.append(0)
        else:
            sigX.append(((c1/b) - (c2/b))**2 / ((c1/b) + (c2/b)))

    asym = [(s - sigAll)**2 for s in sigX]
    kappab.append(np.mean(asym))
    return np.mean(kappab)



def get_kappa_zscore(seq, type1, type2, num_scrambles=10000):
    real_score = get_kappa(seq, type1, type2)

    scramble_scores = []
    for _ in range(num_scrambles):
        scrambled = ''.join(random.sample(seq, len(seq)))
        score = get_kappa(scrambled, type1, type2)
        scramble_scores.append(score)

    scramble_scores = np.array(scramble_scores)

    # === Safeguard: Check if values are suitable for Gamma fit ===
    if np.all(scramble_scores == 0) or np.any(scramble_scores <= 0) or np.std(scramble_scores) == 0:
        return np.nan

    try:
        # Fit a gamma distribution
        fit_alpha, fit_loc, fit_beta = stats.gamma.fit(scramble_scores, floc=0)
        mean = stats.gamma.mean(fit_alpha, fit_loc, fit_beta)
        var = stats.gamma.var(fit_alpha, fit_loc, fit_beta)
        std = math.sqrt(var) if var > 0 else 1e-6
        z = (real_score - mean) / std
    except Exception as e:
        print("⚠️ Gamma fit failed:", e)
        z = np.nan

    return z

def get_omega(seq, type1, threshold=0.05, blob_size=5):
    # Check frequency threshold
    freq = sum(seq.count(res) for res in type1) / len(seq)
    if freq <= threshold:
        return 0.0

    omegab = []
    b = blob_size
    count = sum(seq.count(res) for res in type1)
    sigAll = ((count/len(seq)) - (1 - count/len(seq)))**2

    sigX = []
    for x in range(0, len(seq) - b + 2):
        subseq = seq[x:x+b]
        c = sum(subseq.count(res) for res in type1)
        sigX.append(((c/b) - (1 - c/b))**2)

    asym = [(s - sigAll)**2 for s in sigX]
    omegab.append(np.mean(asym))
    return np.mean(omegab)


def get_omega_zscore(seq, type1, num_scrambles=10000):
    real_score = get_omega(seq, type1)
    scramble_scores = [get_omega(''.join(random.sample(seq, len(seq))), type1)
                       for _ in range(num_scrambles)]
    scramble_scores = np.array(scramble_scores)

    if np.all(scramble_scores == 0) or np.any(scramble_scores <= 0) or np.std(scramble_scores) == 0:
        return np.nan

    try:
        fit_alpha, fit_loc, fit_beta = stats.gamma.fit(scramble_scores, floc=0)
        mean = stats.gamma.mean(fit_alpha, fit_loc, fit_beta)
        var = stats.gamma.var(fit_alpha, fit_loc, fit_beta)
        std = math.sqrt(var) if var > 0 else 1e-6
        z = (real_score - mean) / std
    except Exception as e:
        print("⚠️ Gamma fit failed:", e)
        z = np.nan

    return z

                
        
        
def get_org_seq_vals(myseq,typeall,fracsall):
    
    org_seq_arr = np.zeros((len(typeall),len(typeall)))
    
    for count1 in range(0,len(typeall)):
        type1 = typeall[count1]

        for count2 in range(count1,len(typeall)):
            type2 = typeall[count2]

            if type1 == type2 and fracsall[count1]>0.12:
                org_seq_arr[count1, count2]=get_omega(myseq,type1)
                
            if type1 != type2 and fracsall[count1]>0.12 and fracsall[count2]>0.12:
                org_seq_arr[count1, count2]=get_kappa(myseq,type1,type2)
    
    org_seq_1d=org_seq_arr.reshape([1, len(typeall)**2])
    
    return org_seq_1d


def get_scramble_seqs_vals(myseq,num_seqs,typeall,fracsall):
    
    currseq=[]
    allseqs=[]
    scr_vals=np.zeros((num_seqs,len(typeall)**2))
    
    for x in range(0,num_seqs):
        currseq=''.join(random.sample(myseq,len(myseq)))
        
        scr_seq_arr = np.zeros((len(typeall),len(typeall)))
    
        for count1 in range(0,len(typeall)):
            type1 = typeall[count1]

            for count2 in range(count1,len(typeall)):
                type2 = typeall[count2]

                if type1 == type2 and fracsall[count1]>0.12:
                    scr_seq_arr[count1, count2]=get_omega(currseq,type1)

                if type1 != type2 and fracsall[count1]>0.12 and fracsall[count2]>0.12:
                    scr_seq_arr[count1, count2]=get_kappa(currseq,type1,type2)
        
        scr_vals[x,0:len(typeall)**2] = scr_seq_arr.reshape([1, len(typeall)**2])
        allseqs.append(currseq)
    
    #fit to a gamma distribution and obtain mean and variance
    alpha=[]
    beta=[]
    amean=[]
    avar=[]
    #for column in scr_vals:
    #for x in range(0,len(scr_vals[1])): #0 to 63
    
    scr_vals_t=scr_vals.transpose()
    scr_vals_row = scr_vals_t.shape[0]
    for i in range(0,scr_vals_row):   
        fit_alpha, fit_loc, fit_beta = stats.gamma.fit(scr_vals_t[i,:], floc=0)


        cmean = stats.gamma.mean(fit_alpha,fit_loc,fit_beta)
        cvar = stats.gamma.var(fit_alpha,fit_loc,fit_beta)
        alpha.append(fit_alpha)
        beta.append(fit_beta)
        amean.append(cmean)
        avar.append(cvar)
 
    return [alpha,amean,avar,scr_vals,allseqs]
    

####### SCRIPT STARTS HERE ####### 

if __name__ == "__main__":
        
    #input sequence
    orthseqs=['IEQEKDVTKPQRPSLNQSIKTHNQSVPKREPKREEPQQQNTVSRHTSQPA']
    
    num_seqs=100000
    pol=['S','T','N','Q','C','H']
    hyd=['I','L','M','V']
    pos=['R','K']
    neg=['E','D']
    aro=['F','W','Y']
    ala=['A']
    pro=['P']
    gly=['G']
    
    typeall=[pol,hyd,pos,neg,aro,ala,pro,gly]
    
    zvec=np.zeros((len(orthseqs),int(len(typeall)+(len(typeall)*(len(typeall)-1))/2)))
    zvecdb=np.zeros((len(orthseqs),len(typeall)**2))
    zvecdbscr=np.zeros((len(orthseqs),len(typeall)**2))
    
    countseqs=-1
    for myseq in orthseqs:
        #print(myseq)
        fracsall=[]
        countseqs=countseqs+1
        for type1 in typeall:
            mycount=0
            for res in type1:
                mycount=mycount+myseq.count(res)
            fracsall.append(mycount/len(myseq))
    
        #print(fracsall)
        myarr=get_org_seq_vals(myseq,typeall,fracsall)
    
        # Returns mean of scrambles, std of scrambles, all values in a number of scramble x 64 list, and all scramble sequences 
        [alpha,amean,avar,allscrvals,allscrseqs]=get_scramble_seqs_vals(myseq,num_seqs,typeall,fracsall)
        
        # Get difference of scramble from input sequence
        difffromseq=[]
        for x in range(0,len(allscrvals)):
            difffromseq.append(sum(abs(myarr[0]-allscrvals[x]))) # if care about everything
            #difffromseq.append(abs(myarr[0][19]-allscrvals[x][19])) # if just care about kappa
            #print(sum(abs(myarr[0]-allscrvals[x]))) # if care about everything
            #print(abs(myarr[0][19]-allscrvals[x][19])) # if just care about kappa
        
        #Find most similar scramble
        val, idx = min((val, idx) for (idx, val) in enumerate(difffromseq))
        
        # Create 8x8 matrix of original sequence
        for x in range(0,myarr.shape[1]):
            if myarr[0,x]==0:
                zvecdb[countseqs,x]=0
            else:
                zvecdb[countseqs,x]=(myarr[0,x]-amean[x])/math.sqrt(avar[x])
    
        # Plot orginal sequence z-matrix            
        fig, ax = plt.subplots(1,1)
        img = ax.imshow(np.array(zvecdb[0,:]).reshape([len(typeall), len(typeall)]),vmin=-3, vmax=3, cmap='bwr', aspect='auto')
        fig.colorbar(img)
        x_label_list = ['µ', 'h', '+', '-','π','A','P','G']
        ax.set_xticks([0,1,2,3,4,5,6,7])
        ax.set_xticklabels(x_label_list)
        ax.set_yticks([0,1,2,3,4,5,6,7])
        ax.set_yticklabels(x_label_list)
        
        # Create 8x8 matrix of most similar scramble
        for x in range(0,len(allscrvals[idx])):
            if allscrvals[idx,x]==0:
                zvecdbscr[countseqs,x]=0
            else:
                zvecdbscr[countseqs,x]=(allscrvals[idx,x]-amean[x])/math.sqrt(avar[x])
    
        # Plot similar sequence z-matrix            
        fig, ax = plt.subplots(1,1)
        img = ax.imshow(np.array(zvecdbscr[0,:]).reshape([len(typeall), len(typeall)]),vmin=-3, vmax=3, cmap='bwr', aspect='auto')
        fig.colorbar(img)
        x_label_list = ['µ', 'h', '+', '-','π','A','P','G']
        ax.set_xticks([0,1,2,3,4,5,6,7])
        ax.set_xticklabels(x_label_list)
        ax.set_yticks([0,1,2,3,4,5,6,7])
        ax.set_yticklabels(x_label_list)
        
        print(allscrseqs[idx])
    
    
    
    
    
        
