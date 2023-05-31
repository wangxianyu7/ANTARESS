import numpy as np
from utils import stop,npint,np_interp,np_where1D
import matplotlib.pyplot as plt
from copy import deepcopy
from utils_plots import custom_axis
from MCMC_routines import calc_HDI



##Inverted CDFs: retourne valeurs a partir d'un nombre aleatoire tire sur une distribution definie sur la grille xgrid
def invert_CDF(xgrid,CDFtab,rand_draw):

    ##Interpole xtab sur la table de la CDF au niveau de rand
    return np_interp(rand_draw,CDFtab,xgrid)


#Compute the cumulative distribution function of the chosen distribution
#    - on calcule la distribution du parametre a partir des chaines, puis la CDF a partir de la distribution obtenue
#    - the CDF is rescaled so that it goes from 0 to 1: 
# si on a : 0 <= y0 < CDF_temp < y1 
#           0 < CDF_temp-y0 < y1-y0 
#           0 < (CDF_temp-y0)/(y1-y0)
#    - ensuite quand on veut tirer au hasard un parametre sur la distrib on interpole la table de la CDF en fonction de la CDF au random demande
#    - si on a plusieurs parametres cette approche ne fonctionne pas s'ils sont correles, c'est pour ca qu'on utilise directement les chaines
def hist_cdf(chain_val):
    
    hist_val, bin_edg_val = np.histogram(chain_val, bins=ngrid,density=True)
    grid_val = 0.5*(bin_edg_val[0:-1]+bin_edg_val[1:])
    cdf_val = np.cumsum(hist_val)
    cdf_val = (cdf_val-np.min(cdf_val))/(np.max(cdf_val)-np.min(cdf_val))
    
    return grid_val,hist_val,cdf_val

# #Draw distribution to compare with original one
# plt.plot(grid_lamb_b*180./np.pi,hist_lamb_b,drawstyle='steps-mid',color='dodgerblue')
# rand_draw = np.random.uniform(low=0.0, high=1.0, size=10000)
# lamb_rand_pts = invert_CDF(grid_lamb_b,cdf_lamb_b,rand_draw)
# hist_lamb_test, bin_edg_lamb_test = np.histogram(lamb_rand_pts, bins=ngrid,density=True)
# grid_lamb_test = 0.5*(bin_edg_lamb_test[0:-1]+bin_edg_lamb_test[1:])
# plt.plot(grid_lamb_test*180./np.pi,hist_lamb_test,drawstyle='steps-mid',color='orange')
# plt.show()
# stop()


def print_val(chain_ang,frac_search,nbins_par,dbins_par,bw_fact,HDI_lev):
        
    #Median value and HDI intervals
    med_par = np.median(chain_ang)
    HDI_interv= [] 
    HDI_interv_txt=''
    HDI_interv_txt,HDI_frac=calc_HDI(chain_ang,nbins_par,dbins_par,bw_fact,frac_search,HDI_interv,HDI_interv_txt)
    print('  > med :'+"{0:.8e}".format(med_par))  
    print('  > HDI '+HDI_lev+' : '+str(HDI_interv_txt))
    for irange in range(len(HDI_interv)):print('  > err : -'+"{0:.3e}".format(med_par-HDI_interv[irange][0])+' +'+"{0:.3e}".format(HDI_interv[irange][1]-med_par))  
    return med_par,HDI_interv
   
def phase_fold(x_mid,lambda_chain):
    
    #Fold between +-180 degrees around given value
    lambda_temp=deepcopy(lambda_chain)+180.-x_mid
    w_gt_360=(lambda_temp > 360.)
    if True in w_gt_360:lambda_chain[w_gt_360]=np.mod(lambda_temp[w_gt_360],360.)-180.+x_mid
    w_lt_0=(lambda_temp < 0.)
    if True in w_lt_0:
        i_mod=npint(np.abs(lambda_temp[w_lt_0])/(360.))+1.
        lambda_chain[w_lt_0] = lambda_temp[w_lt_0]+i_mod*360.-180.+x_mid

    return lambda_chain    



if __name__ == '__main__':
    
    #General options
    font_size=18
    ngrid=1000
    HDI_lev='1s'
    fig_path = '/Users/bourrier/Travaux/ANTARESS/HD3167/Plots_dyna/'
    
    
    
    #Fractions associated with each confidence level
    if HDI_lev=='1s':
        frac_search=0.6826894921370859
    elif HDI_lev=='2s':
        frac_search=0.954499736103642
    elif HDI_lev=='3s':
        frac_search=0.997300204  
    
    
    
    
    
    
    
    ##-------------------------------------------------------------
    print('Uploading')
    
    #Chains for all planets
    chains_dic={}
    pl_list = ['HD3167_b','HD3167_c']
    
    # data_load=np.load('/Users/bourrier/Travaux/ANTARESS/En_cours/HD3167_b_Saved_data/Intr_data_prop/HRRM_loose_priors_Cdeg1_FWHMdeg0_vsini2.6/mcmc/merged_deriv_chains_walk30_steps5000_HD3167_b.npz',allow_pickle=True)['data'].item()
    # var_par_list = data_load['var_par_list']
    # chains_dic['b']={'lambda_deg':data_load['merged_chain'][:,np_where1D(var_par_list=='lambda_deg')[0]]}
    
    # data_load=np.load('/Users/bourrier/Travaux/ANTARESS/En_cours/HD3167_c_Saved_data/Intr_data_prop/HRRM_loose_priors_Cdeg1_FWHMdeg0/mcmc/merged_deriv_chains_walk20_steps4000_HD3167_c.npz',allow_pickle=True)['data'].item()  
    # var_par_list = data_load['var_par_list']
    # chains_dic['c']={'lambda_deg':data_load['merged_chain'][:,np_where1D(var_par_list=='lambda_deg')[0]]}
    
    #Global fit
    data_load=np.load('/Users/bourrier/Travaux/ANTARESS/En_cours/HD3167_bHD3167_c_Saved_data/Intr_data_prop/Fit_osamp5_n51/Fit_C2_ystar2/mcmc_postproc/merged_deriv_chains_walk39_steps6000_HD3167_bHD3167_c.npz',allow_pickle=True)['data'].item()  
    var_par_list = data_load['var_par_list']
    print('Par. list :',var_par_list)
    for pl_loc in pl_list:
        chains_dic[pl_loc]={'lambda_deg':data_load['merged_chain'][:,np_where1D(var_par_list=='lambda_deg__'+pl_loc)[0]]}
    chains_dic['istar']=data_load['merged_chain'][:,np_where1D(var_par_list=='istar')[0]]
        
    ##-------------------------------------------------------------
     
    fold=True   #& False
    
    if fold:
        print('Phase-folding')   
    
        chains_dic['HD3167_b']['lambda_deg']=phase_fold(80.,chains_dic['HD3167_b']['lambda_deg'])
        chains_dic['HD3167_c']['lambda_deg']=phase_fold(-110.,chains_dic['HD3167_c']['lambda_deg'])
        
        #Repeat PDF to avoid edge effects, when the chain goes up to the domain boundaries
        #lambda_temp=deepcopy(chains_dic['b']['lambda_deg'])
        #chains_dic['b']['lambda_deg']=np.concatenate((lambda_temp-2.*np.pi,lambda_temp,lambda_temp+2.*np.pi))     #unfold distribution to avoid edge effects
        
    
    ##-------------------------------------------------------------
    
    thin=True   & False
    
    if thin:
        print('Thinning')
        
        #Identify shortest chain
        pl_ref = ''
        n_samp = 1e10
        pl_dic={}
        for pl_loc in pl_list:
            pl_dic[pl_loc]={'nsamp': len(chains_dic[pl_loc]['lambda_deg'])}
            print('  Samples for '+pl_loc+' :',pl_dic[pl_loc]['nsamp'])    
            if pl_dic[pl_loc]['nsamp']<n_samp:
                  n_samp=pl_dic[pl_loc]['nsamp']
                  pl_ref=pl_loc
        print('  Shortest chain : ',pl_ref)
        
        #Thin other chains to the same length
        for pl_loc in pl_list:
            if pl_loc !=pl_ref:
                nsub = int(pl_dic[pl_loc]['nsamp']/pl_dic[pl_ref]['nsamp'])
                nrem = pl_dic[pl_loc]['nsamp']-pl_dic[pl_ref]['nsamp']*nsub
                for key in chains_dic[pl_loc]:
                    chains_dic[pl_loc][key]=chains_dic[pl_loc][key][:-nrem]            
                    chains_dic[pl_loc][key]=chains_dic[pl_loc][key][::nsub]
                pl_dic[pl_loc]['nsamp'] =     len(chains_dic[pl_loc]['lambda_deg'])
            
        for pl_loc in pl_list:
            pl_dic[pl_loc]['nsamp'] =     len(chains_dic[pl_loc]['lambda_deg'])
            print('  Samples for '+pl_loc+' :',pl_dic[pl_loc]['nsamp'])
    
        ##-------------------------------------------------------------
        print('Shuffling and associating samples')    
        #We shuffle the secondary chains to remove correlations, and for each shuffling associate to the suffled chain the samples from the reference chain   
        nshuff = 1   #20
        
        
        final_chains_dic={}
        for pl_loc in pl_list:
            final_chains_dic[pl_loc]={}
            for key in chains_dic[pl_loc]:final_chains_dic[pl_loc][key]=np.zeros(0,dtype=float)
        
        for ishuff in range(nshuff):
            for pl_loc in pl_list:
                for key in chains_dic[pl_loc]:
                    if pl_loc == pl_ref:final_chains_dic[pl_loc][key]=np.append(final_chains_dic[pl_loc][key],chains_dic[pl_loc][key])
                    else:
                        temp_chain = deepcopy(chains_dic[pl_loc][key])
                        np.random.shuffle(temp_chain)
                        final_chains_dic[pl_loc][key]=np.append(final_chains_dic[pl_loc][key],temp_chain)
    
    else:
        final_chains_dic = deepcopy(chains_dic)    
    
    n_chain  = len(final_chains_dic[pl_list[0]]['lambda_deg'])
    
    ##-------------------------------------------------------------
    check_dist=True   & False
    
    if check_dist:
        print('Checking distributions')   
    
        # key_plot = 'lambda_deg'
        # ngrid_plot=100
    
        # for pl_loc in pl_list:
        #     hist_val, bin_edg_val = np.histogram(final_chains_dic[pl_loc][key_plot], bins=ngrid_plot,density=True)
        #     grid_val = 0.5*(bin_edg_val[0:-1]+bin_edg_val[1:])
        #     plt.plot(grid_val,hist_val,drawstyle='steps-mid')
        # plt.show()
        # stop()
    
        key_plot = 'istar'
        ngrid_plot=100
    
        for pl_loc in pl_list:
            hist_val, bin_edg_val = np.histogram(final_chains_dic[key_plot], bins=ngrid_plot,density=True)
            grid_val = 0.5*(bin_edg_val[0:-1]+bin_edg_val[1:])
            plt.plot(grid_val,hist_val,drawstyle='steps-mid')
        plt.show()
        stop()
    
    
    
    
    
    
    
    
    
    
    ################################################################################################
    #Calculation of delta_lambda
    print('Calculating lambda_'+pl_list[0]+' - lambda_'+pl_list[1])
    dlamb = final_chains_dic[pl_list[0]]['lambda_deg'] - final_chains_dic[pl_list[1]]['lambda_deg']
    
    nbins_par=None #20
    dbins_par=None
    bw_fact=None
    med_par,HDI_interv = print_val(dlamb,frac_search,nbins_par,dbins_par,bw_fact,HDI_lev)
    
    
    #Plot
    plot_act=True & False
    
    if plot_act:
        
        x_range=[-150.,320.]
        y_range=[0,0.085]
        ngrid_plot=100
        
        plt.ioff()        
        fig = plt.figure(figsize=(10,6))    
        ax=plt.gca()
    
        hist_val, bin_edg_val = np.histogram(dlamb, bins=ngrid_plot,density=True)
        grid_val = 0.5*(bin_edg_val[0:-1]+bin_edg_val[1:])
        plt.plot(grid_val,hist_val,drawstyle='steps-mid',color='black')    
        ax.axvline(med_par, ls="-", color='black')
        for HDI_sub in HDI_interv:
            ax.axvline(HDI_sub[0], ls=":", color='black') 
            ax.axvline(HDI_sub[1], ls=":", color='black')
            
        hist_val, bin_edg_val = np.histogram(final_chains_dic[pl_list[0]]['lambda_deg'], bins=ngrid_plot,density=True)
        grid_val = 0.5*(bin_edg_val[0:-1]+bin_edg_val[1:])        
        plt.plot(grid_val,hist_val,drawstyle='steps-mid',color='dodgerblue')
    
        hist_val, bin_edg_val = np.histogram(final_chains_dic[pl_list[1]]['lambda_deg'], bins=ngrid_plot,density=True)
        grid_val = 0.5*(bin_edg_val[0:-1]+bin_edg_val[1:])
        plt.plot(grid_val,hist_val,drawstyle='steps-mid',color='orange')
    
        custom_axis(plt,ax=ax,position=[0.15,0.15,0.95,0.95],x_range=x_range,y_range=y_range,
                    #xmajor_int=0.5,xminor_int=0.1,ymajor_int=0.5,yminor_int=0.1,
                    #xmajor_form='%.1f',ymajor_form='%.1f',
                    # x_title='$\lambda_{b}$ - $\lambda_{c}$ ($^\circ$)',
                    x_title='Obliquities ($^\circ$)',
                    y_title='Density',
                    font_size=font_size,xfont_size=font_size,yfont_size=font_size)
    
        plt.savefig(fig_path+'Obliquities_PDFs.pdf') 
        plt.close()    
        stop()    
    
    
    
    
    
    
    
    
    ################################################################################################
    #Calculation of mutual inclination
    #    - cos(i_m)=cos(i_b)*cos(i_c)+cos(Omega)*sin(i_b)*sin(i_c)
    #      Omega = om_c - om_b     difference between longitudes of ascending nodes
    #      taking the sky-projected node line of the star as a reference for the longitude of the ascending node, there are two possible
    # solutions corresponding to the same RM signal:
    # om = lambda, i = ip
    # om = -lambda, i = 180 - ip
    #      for the two planets, there are thus four combinations (considering that i_m ranges between 0 and 180):
    # A : (i_b,l_b) and (i_c,l_c)
    #     cos(i_mA)=cos(i_b)*cos(i_c)+cos(l_c - l_b)*sin(i_b)*sin(i_c)    
    # B : (180-i_b,-l_b) and (i_c,l_c)
    #     cos(i_mB)=cos(180 - i_b)*cos(i_c)+cos(l_c + l_b)*sin(180 - i_b)*sin(i_c)  
    #              =-cos(i_b)*cos(i_c)+cos(l_c + l_b)*sin(i_b)*sin(i_c)  
    # C : (i_b,l_b) and (180-i_c,-l_c) 
    #     cos(i_mC)=cos(i_b)*cos(180 - i_c)+cos(-l_c - l_b)*sin(i_b)*sin(180 - i_c)
    #              =-cos(i_b)*cos(i_c)+cos(l_c + l_b)*sin(i_b)*sin(i_c)
    #              =cos(i_mB)
    # D : (180-i_b,-l_b) and (180-i_c,-l_c) 
    #     cos(i_mD)=cos(180 - i_b)*cos(180 - i_c)+cos(-l_c + l_b)*sin(180 - i_b)*sin(180 - i_c)     
    #              =cos(i_b)*cos(i_c)+cos(l_c - l_b)*sin(i_b)*sin(i_c) 
    #              =cos(i_mA)
    #So in the end there are only two possible solutions (A=D, B=C)
    print('Calculating i_mut')
    hn_chain1 = int(n_chain/2)
    hn_chain2 = n_chain-hn_chain1
    ip_chains = {}
    ip_grid = {}
    ip_hist = {}
    ip_cdf = {}
    
    #Distributions of ib and ic
    #    - we use the published error bars to define gaussian profiles with different wings
    #    - we draw half the required number of points in the distribution on either side of the median
    ip_mean = {'HD3167_b':83.4*np.pi/180.       , 'HD3167_c':89.3*np.pi/180.}
    ip_high = {'HD3167_b':7.7   *np.pi/180.     , 'HD3167_c': 0.5    *np.pi/180.} 
    ip_low  = {'HD3167_b': 4.6*np.pi/180.    , 'HD3167_c':0.96*np.pi/180.}
    # ib_mean = 82.6*np.pi/180.
    # ib_high=3.67   *np.pi/180.  
    # ib_low = 4.35*np.pi/180.
    
    for pl_loc in pl_list:
        rand_draw_right = np.random.normal(loc=ip_mean[pl_loc], scale=ip_high[pl_loc], size=2*n_chain)
        rand_draw_right = rand_draw_right[rand_draw_right>ip_mean[pl_loc]]
        rand_draw_right = rand_draw_right[0:hn_chain1]
        rand_draw_left = np.random.normal(loc=ip_mean[pl_loc], scale=ip_low[pl_loc], size=2*n_chain)
        rand_draw_left = rand_draw_left[rand_draw_left<=ip_mean[pl_loc]]
        rand_draw_left = rand_draw_left[0:hn_chain2]
        ip_chains[pl_loc] = np.append(rand_draw_left,rand_draw_right)
        ip_grid[pl_loc],ip_hist[pl_loc],ip_cdf[pl_loc] = hist_cdf(ip_chains[pl_loc])   
    
    
    #Plot
    plot_act=True  & False
    
    if plot_act:
        pl_loc='HD3167_b'
        
        # print(len(chain_ib))
        # plt.plot(grid_ib*180./np.pi,cdf_ib,drawstyle='steps-mid',color='black')
        # plt.show()
        # stop()
        
        #Draw distribution to compare with original one
        plt.plot(ip_grid[pl_loc],ip_hist[pl_loc],drawstyle='steps-mid',color='dodgerblue')
        rand_draw = np.random.uniform(low=0.0, high=1.0, size=len(ip_chains[pl_loc]))
        irand_pts = invert_CDF(ip_grid[pl_loc],ip_cdf[pl_loc],rand_draw)
        hist_itest, bin_edg_itest = np.histogram(irand_pts, bins=ngrid,density=True)
        grid_itest = 0.5*(bin_edg_itest[0:-1]+bin_edg_itest[1:])
        plt.plot(grid_itest,hist_itest,drawstyle='steps-mid',color='orange')
        plt.show()
        stop()
    
    #Mutual inclination for the two degenerate cases
    im_chainA = np.arccos( np.cos(ip_chains[pl_list[0]])*np.cos(ip_chains[pl_list[1]])+np.cos((final_chains_dic[pl_list[1]]['lambda_deg'] - final_chains_dic[pl_list[0]]['lambda_deg'])*np.pi/180.)*np.sin(ip_chains[pl_list[0]])*np.sin(ip_chains[pl_list[1]]) )*180./np.pi
    im_chainB = np.arccos(-np.cos(ip_chains[pl_list[0]])*np.cos(ip_chains[pl_list[1]])+np.cos((final_chains_dic[pl_list[1]]['lambda_deg'] + final_chains_dic[pl_list[0]]['lambda_deg'])*np.pi/180.)*np.sin(ip_chains[pl_list[0]])*np.sin(ip_chains[pl_list[1]]) )*180./np.pi
    
    
    #Combined distributions
    #    - the two configurations are equiprobable, and have the same number of samples, so we just combine them in a common distribution
    im_chain_glob = np.append(im_chainA,im_chainB)
    
    
    #Histograms for the plots
    ngrid_chain=200
    hist_imA, bin_edg_imA = np.histogram(im_chainA, bins=ngrid_chain,density=True)
    grid_imA = 0.5*(bin_edg_imA[0:-1]+bin_edg_imA[1:])
    hist_imB, bin_edg_imB = np.histogram(im_chainB, bins=ngrid_chain,density=True)
    grid_imB = 0.5*(bin_edg_imB[0:-1]+bin_edg_imB[1:])
    hist_im_glob, bin_edg_im_glob = np.histogram(im_chain_glob, bins=ngrid_chain,density=True)
    grid_im_glob = 0.5*(bin_edg_im_glob[0:-1]+bin_edg_im_glob[1:])
    
    #Best-fit results
    nbins_par=None #20
    dbins_par=None
    bw_fact=None
    print('  Case A')
    med_imA,HDI_interv_imA = print_val(im_chainA,frac_search,nbins_par,dbins_par,bw_fact,HDI_lev)
    print('  Case B')
    med_imB,HDI_interv_imB = print_val(im_chainB,frac_search,nbins_par,dbins_par,bw_fact,HDI_lev)
    print('  Case global')
    med_im_glob,HDI_interv_im_glob = print_val(im_chain_glob,frac_search,nbins_par,dbins_par,bw_fact,HDI_lev)
    
    
    
    
    
    
    #Plot
    plot_act=True  #  & False
    
    if plot_act:
        
        font_size=20
        x_range=[0.,180.]
        y_range=np.array([ 0. , np.max(np.concatenate((hist_imA,hist_imB))) ])*1.05
        
        
        plt.ioff()        
        fig = plt.figure(figsize=(10,6))  
        ax=plt.gca()
        plt.plot(grid_imA,hist_imA,drawstyle='steps-mid',color='dodgerblue')
        # plt.plot(grid_imB,hist_imB,drawstyle='steps-mid',color='dodgerblue')
        # plt.plot(grid_im_glob,hist_im_glob,drawstyle='steps-mid',color='black')
        ax.axvline(med_imA, ls='-', color='dodgerblue')
        # ax.axvline(med_imB, ls='-', color='dodgerblue')
        # ax.axvline(med_im_glob, ls='-', color='black')
    
        for HDI_sub in HDI_interv_imA:
            ax.axvline(HDI_sub[0], ls=":", color='dodgerblue') 
            ax.axvline(HDI_sub[1], ls=":", color='dodgerblue')
        # for HDI_sub in HDI_interv_imB:
        #     ax.axvline(HDI_sub[0], ls=":", color='orange') 
        #     ax.axvline(HDI_sub[1], ls=":", color='orange')
        # for HDI_sub in HDI_interv_im_glob:
        #     ax.axvline(HDI_sub[0], ls=":", color='black') 
        #     ax.axvline(HDI_sub[1], ls=":", color='black')
    
        custom_axis(plt,ax=ax,position=[0.15,0.15,0.95,0.95],x_range=x_range,
                    y_range=y_range,
                    #xmajor_int=0.5,xminor_int=0.1,ymajor_int=0.5,yminor_int=0.1,
                    #xmajor_form='%.1f',ymajor_form='%.1f',
                    x_title='i$_\mathrm{mut}$ ($^\circ$)',
                    y_title='Density',
                    font_size=font_size,xfont_size=font_size,yfont_size=font_size)
    
        plt.savefig(fig_path+'Mutual_inc_PDFs.pdf') 
        plt.close()    
        
    
    
    
    ################################################################################################
    #Calculation of psi angle
    #    - psi = acos(sin(istar)*cos(lamba)*sin(ip) + cos(istar)*cos(ip))
    calc_psi=True
    
    if calc_psi:
        print('Calculating psi')
        
        istar_chain = final_chains_dic['istar']*np.pi/180. 
        for pl_loc in pl_list:
            final_chains_dic[pl_loc]['psi_deg'] = np.arccos(np.sin(istar_chain)*np.cos(final_chains_dic[pl_loc]['lambda_deg']*np.pi/180.)*np.sin(ip_chains[pl_loc]) + np.cos(istar_chain)*np.cos(ip_chains[pl_loc]))*180./np.pi
    
        #Histograms for the plots
        ngrid_chain=200
        hist_psi_b, bin_edg_psi_b = np.histogram(final_chains_dic['HD3167_b']['psi_deg'], bins=ngrid_chain,density=True)
        grid_psi_b = 0.5*(bin_edg_psi_b[0:-1]+bin_edg_psi_b[1:])
        hist_psi_c, bin_edg_psi_c = np.histogram(final_chains_dic['HD3167_c']['psi_deg'], bins=ngrid_chain,density=True)
        grid_psi_c = 0.5*(bin_edg_psi_c[0:-1]+bin_edg_psi_c[1:])
    
        #Best-fit results
        nbins_par=None #20
        dbins_par=None
        bw_fact=None
        print('b')
        med_psi_b,HDI_interv_psi_b = print_val(final_chains_dic['HD3167_b']['psi_deg'],frac_search,nbins_par,dbins_par,bw_fact,HDI_lev)
        print('c')
        med_psi_c,HDI_interv_psi_c = print_val(final_chains_dic['HD3167_c']['psi_deg'],frac_search,nbins_par,dbins_par,bw_fact,HDI_lev)    
    
    #Plot
    plot_act=True #   & False
    
    if plot_act:
        
        font_size=20
        x_range=[0.,180.]
        y_range=[0,0.1]
        
        
        plt.ioff()        
        fig = plt.figure(figsize=(10,6))  
        ax=plt.gca()
        plt.plot(grid_psi_b,hist_psi_b,drawstyle='steps-mid',color='limegreen')
        plt.plot(grid_psi_c,hist_psi_c,drawstyle='steps-mid',color='orange')
        ax.axvline(med_psi_b, ls='-', color='limegreen')
        ax.axvline(med_psi_c, ls='-', color='orange')
    
        for HDI_sub in HDI_interv_psi_b:
            ax.axvline(HDI_sub[0], ls=":", color='limegreen') 
            ax.axvline(HDI_sub[1], ls=":", color='limegreen')
        for HDI_sub in HDI_interv_psi_c:
            ax.axvline(HDI_sub[0], ls=":", color='orange') 
            ax.axvline(HDI_sub[1], ls=":", color='orange')
    
        custom_axis(plt,ax=ax,position=[0.15,0.15,0.95,0.95],x_range=x_range,
                    y_range=y_range,
                    #xmajor_int=0.5,xminor_int=0.1,ymajor_int=0.5,yminor_int=0.1,
                    #xmajor_form='%.1f',ymajor_form='%.1f',
                    x_title='$\Psi$ ($^\circ$)',
                    y_title='Density',
                    font_size=font_size,xfont_size=font_size,yfont_size=font_size)
    
        plt.savefig(fig_path+'Psi_PDFs.pdf') 
        plt.close()    
    
    
    
    
    
    

