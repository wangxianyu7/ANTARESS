"""
AUTHOR : Michael Cretignier 
EDITOR : Khaled Al Moulla

Collection of classes.
"""

###
### MODULES


import matplotlib.pyplot             as     plt
import numpy                         as     np
import pandas                        as     pd

from   matplotlib.collections        import LineCollection
from   scipy.interpolate             import interp1d
from   statsmodels.stats.weightstats import DescrStatsW

from ..ANTARESS_conversions.KitCat import Kitcat_functions as myf
from ..ANTARESS_conversions.KitCat.Kitcat_functions import rm_outliers as rm_out


#np.warnings.filterwarnings('ignore', category=RuntimeWarning)

###
### CLASSES

class tableXY(object):

    def __init__(self, x, y, *yerr):
        self.stats = pd.DataFrame({},index=[0])
        self.y = np.array(y)  #vector of y

        if x is None:# for a fast table initialisation
            x = np.arange(len(y))
        self.x = np.array(x)  #vector of x
        
        try:
            np.sum(self.y) 
        except: #in case of None
            self.y = np.zeros(len(self.x))
            yerr = [np.ones(len(self.y))]
                
        if len(x)!=len(y):
            print('X et Y have no the same lenght')

        if len(yerr)!=0:
            if len(yerr)==1:
                self.yerr = np.array(yerr[0])
                self.xerr =  np.zeros(len(self.x))
            elif len(yerr)==2:
                self.xerr = np.array(yerr[0])
                self.yerr = np.array(yerr[1])
        else :
            if sum(~np.isnan(self.y.astype('float'))):
                self.yerr = np.ones(len(self.x))*myf.mad(rm_out(self.y.astype('float'),m=2,kind='sigma')[1])
                if not np.sum(abs(self.yerr)):
                    self.yerr = np.ones(len(self.x))
            else:
                self.yerr = np.ones(len(self.x))
            self.xerr =  np.zeros(len(self.x))

    def clip(self, min=[None,None], max=[None,None], replace=True, invers=False):
        """This function seems sometimes to not work without any reason WARNING"""
        min2 = np.array(min).copy() ; max2 = np.array(max).copy()
        if min2[1] == None:
            min2[1] = np.nanmin(self.y)-1
        if max2[1] == None:
            max2[1] = np.nanmax(self.y)+1
        masky = (self.y<=max2[1])&(self.y>=min2[1])
        if min2[0] == None:
            min2[0] = np.nanmin(self.x)-1
        if max2[0] == None:
            max2[0] = np.nanmax(self.x)+1
        maskx = (self.x<=max2[0])&(self.x>=min2[0])
        mask = maskx&masky
        try:
            self.clip_mask = self.clip_mask&mask
        except:    
            self.clip_mask = mask

        if invers:
            mask = ~mask
        self.clipped = tableXY(self.x[mask],self.y[mask],self.xerr[mask],self.yerr[mask])
        if replace==True:
            self.x = self.x[mask] ; self.y = self.y[mask] ; self.yerr = self.yerr[mask] ; self.xerr= self.xerr[mask]
        else:
            self.clipx = self.x[mask] ; self.clipy=self.y[mask] ; self.clipyerr = self.yerr[mask] ; self.clipxerr=self.xerr[mask]

    def copy(self):
        return tableXY(self.x.copy(),self.y.copy(),self.xerr.copy(),self.yerr.copy())

    def fit_line(self, perm=1000, Draw=False, color='k', info=False, fontsize=13, label=True, compute_r=True, offset=True, recenter=True, info_printed=['r','s','i','rms']):
        k = perm
        self.yerr[self.yerr==0] = [np.min(self.yerr),0.1][np.min(self.yerr)==0] #to avoid 0 value
        
        w = 1/self.yerr**2    
        if offset:    
            A = np.array([(self.x-np.mean(self.x)*int(recenter)),np.ones(len(self.x))]).T
        else:
            A = np.array([self.x]).T

        A = A *np.sqrt(w)[:,np.newaxis]
        B = np.array([self.y]*(k+1)).T
        noise = np.random.randn(np.shape(B)[0],np.shape(B)[1])/np.sqrt(w)[:,np.newaxis] ; noise[:,0] = 0
        B = B + noise
        Bmean = np.sum(B*w[:,np.newaxis],axis=0)/np.sum(w)*int(recenter)
        Brms = np.sqrt(np.sum(((B-Bmean)**2*w[:,np.newaxis]),axis=0)/np.sum(w))
        B = B*np.sqrt(w)[:,np.newaxis]
        Cmean = np.sum(self.x*w,axis=0)/np.sum(w)*int(recenter)
        Crms = np.sqrt(np.sum(((self.x-Cmean)**2*w),axis=0)/np.sum(w))

        self.s = np.linalg.lstsq(A,B,rcond=None)[0][0]
        if offset:
            self.i = np.linalg.lstsq(A,B,rcond=None)[0][1]      
        else:
            self.i = self.s*0

        self.lin_slope_w = np.mean(self.s)
        self.lin_errslope_w = np.std(self.s)
        
        self.lin_intercept_w = np.mean(self.i)
        self.lin_errintercept_w = np.std(self.i)

        self.stats['lin_slope_w'] = self.lin_slope_w
        self.stats['lin_slope_w_std'] = self.lin_errslope_w
        self.stats['lin_intercept_w'] = self.lin_intercept_w
        self.stats['lin_intercept_w_std'] = self.lin_errintercept_w     

        if compute_r:
            self.r = self.s*Crms/Brms        
            self.r_pearson_w = np.mean(self.r)
            self.r_errpearson_w = np.std(self.r)
        else:
            self.r_pearson_w = np.inf
            self.r_errpearson_w = np.inf
            
        self.stats['r_pearson_w'] = self.r_pearson_w
        self.stats['r_pearson_w_std'] = self.r_errpearson_w
        
        
        temp = tableXY(self.x, self.y-((self.x-np.mean(self.x)*int(recenter))*self.lin_slope_w+self.lin_intercept_w), self.yerr)
        temp.rms_w()
        self.fit_line_rms = temp.rms
        
        if Draw:
            if label:
                lbl = ''
                if perm==1:
                    symb = '|'
                    lbl = ' $\\mathcal{R}$=%.2f %s S=%.2f %s I=%.2f %s rms=%.2f'%(self.r_pearson_w,symb,self.lin_slope_w,symb,self.lin_intercept_w,symb,temp.rms)
                else:
                    symb = '\n'
                    txt = {
                        'r':['$\\mathcal{R}$',self.r_pearson_w, self.r_errpearson_w],
                        's':['S',self.lin_slope_w, self.lin_errslope_w],
                        'i':['I',self.lin_intercept_w, self.lin_errintercept_w],
                        'rms':['rms',temp.rms,0]}
                    for n,prin in enumerate(info_printed):
                        l1,l2,l3 = txt[prin]
                        if n!=0:
                            lbl = lbl+'%s'%(symb)
                        if l3!=0:
                            lbl = lbl+' '+l1+' = %.2f $\\pm$ %.2f'%(l2,l3)
                        else:
                            lbl = lbl+' '+l1+' = %.2f'%(l2)

                plt.plot(self.x,(self.x-np.mean(self.x)*int(recenter))*self.lin_slope_w+self.lin_intercept_w,color=color,ls='-.',label=lbl)
            else:
                plt.plot(self.x,(self.x-np.mean(self.x)*int(recenter))*self.lin_slope_w+self.lin_intercept_w,color=color,ls='-.')

        if info&Draw:
            plt.legend(fontsize=fontsize)

    def fit_poly(self, Draw = False, d = 2, color='r',cov=True):
        if np.sum(self.yerr)!=0:
            weights=self.yerr
        else :
            weights =np.ones(len(self.x))
        if cov:
            coeff, V = np.polyfit(self.x, self.y, d, w=1/weights,cov=cov)
            self.cov = V
            self.err = np.sqrt(np.diag(V))
        else:
            coeff= np.polyfit(self.x, self.y, d, w=1/weights,cov=cov)
        self.poly_coefficient = coeff
        self.chi2 = np.sum((self.y-np.polyval(coeff,self.x))**2)/np.sum(self.yerr**2)
        self.bic = self.chi2+(d+1)*np.log(len(self.x))
        if Draw==True:
            new_x = np.linspace(self.x.min(),self.x.max(),10000)
            plt.plot(new_x, np.polyval(coeff, new_x), linestyle='-.', color=color, linewidth=1)
            #uncertainty = np.sqrt(np.sum([(err[j]*new_x**j)**2 for j in range(len(err))],axis=0))
            #plt.fill_between(new_x,np.polyval(coeff, new_x)-uncertainty/2,np.polyval(coeff, new_x)+uncertainty/2,alpha=0.4,color=color)

    def interpolate(self, new_grid = 'auto', method = 'cubic', replace = True, interpolate_x=True, fill_value='extrapolate', scale='lin'):
        
        if scale!='lin':
            self.inv()
        
        if type(new_grid)==str:
            new_grid = np.linspace(self.x.min(),self.x.max(),10*len(self.x))
        if type(new_grid)==int:
            new_grid = np.linspace(self.x.min(),self.x.max(),new_grid*len(self.x))
        
        if np.sum(new_grid!=self.x)!=0:
            if replace:
                self.x_backup = self.x.copy()
                self.y_backup = self.y.copy()
                self.xerr_backup = self.xerr.copy()
                self.yerr_backup = self.yerr.copy()  
                self.y = interp1d(self.x, self.y, kind = method, bounds_error = False, fill_value = fill_value)(new_grid)
                if np.sum(abs(self.yerr)):
                    self.yerr = interp1d(self.x, self.yerr, kind = method, bounds_error = False, fill_value = fill_value)(new_grid)
                else:
                    self.yerr = np.zeros(len(new_grid))
                if (interpolate_x)&(bool(np.sum(abs(self.xerr)))):
                    self.xerr = interp1d(self.x, self.xerr, kind = method, bounds_error = False, fill_value = fill_value)(new_grid)  
                else:
                    self.xerr = np.zeros(len(new_grid))
                self.x = new_grid
                
                if scale!='lin':
                    self.inv()
      
            else:
                self.y_interp = interp1d(self.x, self.y, kind = method, bounds_error = False, fill_value = fill_value)(new_grid)
                if np.sum(abs(self.yerr)):
                    self.yerr_interp = interp1d(self.x, self.yerr, kind = method, bounds_error = False, fill_value = fill_value)(new_grid)
                else:
                    self.yerr_interp = np.zeros(len(new_grid))
                if (interpolate_x)&(bool(np.sum(abs(self.xerr)))):
                    self.xerr_interp = interp1d(self.x, self.xerr, kind = method, bounds_error = False, fill_value = fill_value)(new_grid)        
                else:
                    self.xerr_interp = np.zeros(len(new_grid))
                self.x_interp = new_grid
                self.interpolated = tableXY(self.x_interp,self.y_interp,self.xerr_interp,self.yerr_interp)

                if scale!='lin':
                    self.interpolated.inv()
                    self.inv()

    def masked(self,mask,replace=True):
        if replace:
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.xerr = self.xerr[mask]
            self.yerr = self.yerr[mask]
        else:
            return tableXY(self.x[mask],self.y[mask],self.xerr[mask],self.yerr[mask])

    def my_bisector(self,kind='cubic',oversampling=1,weighted=True,num_outpoints='none',between_max=False,vic=10):
        """Compute the bisector of a line if the table(XY) is a table(wavelength/flux) ---"""
        self.order()
        maxi, flux = myf.local_max(self.y,vicinity=vic)        
        maxi_left = maxi[maxi<len(self.x)/2]
        flux_left = flux[maxi<len(self.x)/2]    
        maxi_right = maxi[maxi>len(self.x)/2]
        flux_right = flux[maxi>len(self.x)/2]
        
        maxi1 = 0        
        maxi2 = len(self.x)-1
        
        if between_max:
            if len(flux_left)>0:
                maxi1 = int(maxi_left[flux_left.argmax()])
            
            if len(flux_right)>0:
                maxi2 = int(maxi_right[flux_right.argmax()])
        
        selfx = self.x[maxi1:maxi2+1]
        selfy = self.y[maxi1:maxi2+1]

        if between_max:
            self.xerr = self.xerr[maxi1:maxi2+1]
            self.yerr = self.yerr[maxi1:maxi2+1]
            self.x = self.x[maxi1:maxi2+1]
            self.y = self.y[maxi1:maxi2+1]        
        normalisation = abs(selfx[selfy.argmin()-1]-selfx[selfy.argmin()+1])
        Interpol = interp1d(selfx, selfy, kind=kind, bounds_error=False, fill_value='extrapolate')
        new_x = np.linspace(selfx.min(),selfx.max(),oversampling*(len(selfx)-1)+1)
        new_y = Interpol(new_x)
        
        min_idx = new_y.argmin()
        liste1 = (new_y[0:min_idx+1])[::-1] ; liste2 = new_y[min_idx:]
        left = (new_x[0:min_idx+1])[::-1] ; right = new_x[min_idx:]
        save = myf.match_nearest(liste1,liste2)
        bisector_x = [] ; bisector_y = [] ; bisector_xerr = []
        for num in np.arange(len(save[1:-1,0]))+1:
            j = save[num,0].astype('int')
            k = save[num,1].astype('int')
            bisector_y.append(np.mean([liste1[j],liste2[k]]))
            bisector_x.append(np.mean([left[j],right[k]]))
            if weighted:
                diff_left = (liste1[j-1]-liste1[j+1])/(left[j-1]-left[j+1])
                diff_right = (liste2[k-1]-liste2[k+1])/(right[k-1]-right[k+1])
                diff = np.max([abs(diff_left),abs(diff_right)])
                bisector_xerr.append(1/diff)
                
        if weighted:
            bisector_xerr = np.array(bisector_xerr) ; bisector_xerr = bisector_xerr*normalisation/bisector_xerr[-1]/2
        else:
            bisector_xerr = np.zeros(len(bisector_y))
        bisector = np.vstack([bisector_x,bisector_y,bisector_xerr]).T
        bisector = np.insert(bisector,0,[new_x[min_idx],new_y[min_idx],np.max(bisector_xerr)],axis=0)
        Interpol = interp1d(bisector[:,1], bisector[:,0], kind=kind, bounds_error=False, fill_value='extrapolate')
        if type(num_outpoints)==str:
            num_outpoints = len(bisector[:,1]) 
        new_x = np.linspace(bisector[:,1].min(),bisector[:,1].max(),num_outpoints)
        new_y = Interpol(new_x)
        Interpol = interp1d(bisector[:,1], bisector[:,2], kind=kind, bounds_error=False, fill_value='extrapolate')
        new_yerr = Interpol(new_x)
        bisector = np.vstack([new_y,new_x,new_yerr]).T
        self.bisector = bisector.copy()
        self.bis = tableXY(bisector[:,1],bisector[:,0],bisector[:,2])

    def night_stack(self,db=0,bin_length=1,replace=False):
        
        jdb = self.x
        vrad = self.y
        vrad_std = self.yerr.copy()

        if not np.sum(vrad_std): #to avoid null vector
            vrad_std+=1    

        vrad_std[vrad_std==0] = np.nanmax(vrad_std[vrad_std!=0]*10)    
        
        weights = 1/(vrad_std)**2


        if bin_length:
            groups = ((jdb-db)//bin_length).astype('int')
            groups -= groups[0]
            group = np.unique(groups)
        else:
            group = np.arange(len(jdb))
            groups = np.arange(len(jdb))
            
        mean_jdb = []
        mean_vrad = []
        mean_svrad = []
        
        for j in group:
            g = np.where(groups==j)[0]
            mean_jdb.append(np.sum(jdb[g]*weights[g])/np.sum(weights[g]))
            mean_svrad.append(1/np.sqrt(np.sum(weights[g])))
            mean_vrad.append(np.sum(vrad[g]*weights[g])/np.sum(weights[g]))
            
        mean_jdb = np.array(mean_jdb)
        mean_vrad = np.array(mean_vrad)
        mean_svrad = np.array(mean_svrad)
        
        if replace:
            self.x, self.y, self.xerr,self.yerr = mean_jdb, mean_vrad ,0*mean_svrad, mean_svrad
        else:
            self.stacked = tableXY(mean_jdb,mean_vrad,mean_svrad)

    def order(self, order=None):
        if order is None:
            order = self.x.argsort()
        self.order_liste = order
        self.x = self.x[order]
        self.y = self.y[order]
        self.xerr = self.xerr[order]
        self.yerr = self.yerr[order]

    def plot(self, Show=False, color='k', label='', ls='', lw=2, offset=0, mask=None, capsize=3, fmt='o', markersize=6, zorder=1, species=None, alpha=1, modulo=None, modulo_norm=False, cmap=None, new=False, phase_mod=0, periodic=False, frac=1, yerr=True,xerr=True, sp=None, highlight_seasons=False,cmin=None,cmax=None):
        
        '''For the mask give either the first and last index in a list [a,b] or the mask boolean'''
        
        if modulo==100000: #default value in YARARA
            modulo=None
        
        if (modulo is not None)&(cmap is None):
            try:
                cmap = {'k':'viridis','b':'Blues','r':'Reds','g':'Greens'}[color]
            except:
                cmap = 'viridis'
        if (len(self.x)>20000)&(ls=='')&(modulo is None):
            ls='-'
        
        if species is None:
            species = np.ones(len(self.x))

        if highlight_seasons:
            self.split_seasons(min_gap=highlight_seasons,Plot=False)
            species = self.seasons_species

        if len(np.unique(species))==1:
            colors_species = [color]
        else:
            colors_species = ['k']+['C%.0f'%(i) for i in range(1,1+len(np.unique(species)))]

        for num, selection in enumerate(np.unique(species)):
            
            color = colors_species[num]
            
            if num!=0:
                label=None
            
            if mask is None:
                mask2 = np.ones(len(self.x)).astype('bool')
            elif type(mask[0])==int:
                mask2 = np.zeros(len(self.x)).astype('bool')
                mask2[mask[0]:mask[1]]=True
            else:
                mask2 = mask

            loc = np.where(species[mask2]==selection)[0]
            
            sel = np.arange(len(loc))
            if frac!=1:
                sel = np.random.choice(np.arange(len(loc)),size=int(frac*len(loc)),replace=False)
                
            if new:
                plt.figure()
                
            if sp is not None:
                plt.subplot(sp)
            
            if ls!='':
                if color is not None:
                    if len(color)==len(self.x):
                        points   = np.array([self.x, self.y]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        if cmin is None:
                            cmin = np.nanpercentile(color,5)
                        if cmax is None:
                            cmax = np.nanpercentile(color,95)
                        norm = plt.Normalize(cmin, cmax)    
                        lc = LineCollection(segments, cmap=cmap, norm=norm,lw=lw)
                        lc.set_array(color)
                        line = plt.gca().add_collection(lc)
                        #axc = fig.colorbar(line, ax=plt.gca())
                    else:
                        plt.plot(self.x[mask2][loc][sel],self.y[mask2][loc][sel]+offset,ls=ls,lw=lw,zorder=zorder,label=label,color=color,alpha=alpha)
                else:
                    plt.plot(self.x[mask2][loc][sel],self.y[mask2][loc][sel]+offset,ls=ls,lw=lw,zorder=zorder,label=label,color=color,alpha=alpha)
            else:
                if modulo is not None:
                    norm = ((1-modulo_norm)+modulo_norm*modulo)
                    new_x = ((self.x[mask2][loc]-phase_mod)%modulo)/norm
                    plt.errorbar(new_x[sel], self.y[mask2][loc][sel]+offset, xerr=self.xerr[mask2][loc][sel], yerr=self.yerr[mask2][loc][sel]*int(yerr), fmt=fmt, color=color, alpha=alpha, capsize=capsize, label=label, markersize=markersize,zorder=zorder)
                    plt.scatter(new_x[sel], self.y[mask2][loc][sel]+offset, marker='o', c=self.x[mask2][loc][sel], cmap=cmap, s=markersize,zorder=zorder*100,vmin=np.nanpercentile(self.x[mask2][loc][sel],16),vmax=np.nanpercentile(self.x[mask2][loc][sel],84))
                    if periodic:
                        for i in range(int(np.float(periodic))):
                            plt.errorbar(new_x[sel] - (i+1)*modulo/norm, self.y[mask2][loc][sel]+offset, xerr=self.xerr[mask2][loc][sel], yerr=self.yerr[mask2][loc][sel], fmt=fmt, color=color, alpha=0.3, capsize=capsize, markersize=markersize, zorder=zorder)
                            plt.errorbar(new_x[sel] + (i+1)*modulo/norm, self.y[mask2][loc][sel]+offset, xerr=self.xerr[mask2][loc][sel], yerr=self.yerr[mask2][loc][sel], fmt=fmt, color=color, alpha=0.3, capsize=capsize, markersize=markersize, zorder=zorder)
                else:
                    plt.errorbar(self.x[mask2][loc][sel], self.y[mask2][loc][sel]+offset, xerr=self.xerr[mask2][loc][sel]*int(xerr), yerr=self.yerr[mask2][loc][sel]*int(yerr), fmt=fmt, color=color, alpha=alpha, capsize=capsize, label=label, markersize=markersize,zorder=zorder)
        if Show==True:
            plt.legend()
            plt.show()

    def recenter(self, who='both',weight=False):
        if (who=='X')|(who=='both')|(who=='x'):
            self.xmean = np.nanmean(self.x)
            self.x = self.x - np.nanmean(self.x)
        if (who=='Y')|(who=='both')|(who=='y'):
            self.ymean = np.nanmean(self.y)
            self.y = self.y - np.nanmean(self.y)

    def rm_outliers(self, who='Y', m=2, kind='inter', bin_length=0, replace=True):
        vec = self.copy()
        if bin_length:
            self.night_stack(bin_length=bin_length,replace=False)
            vec_binned = self.stacked.copy()
        else:
            vec_binned = self.copy()
            
        if who=='Xerr':
            mask = rm_out(self.xerr, m=m, kind=kind)[0]
            vec_binned = vec_binned.xerr
            vec = vec.xerr
        if who=='Yerr':
            mask = rm_out(self.yerr, m=m, kind=kind)[0]
            vec_binned = vec_binned.yerr
            vec = vec.yerr
        if who=='Y':
            mask = rm_out(self.y, m=m, kind=kind)[0]
            vec_binned = vec_binned.y
            vec = vec.y
        if who=='X':
           mask = rm_out(self.x, m=m, kind=kind)[0]
           vec_binned = vec_binned.x
           vec = vec.x
         
        if bin_length:
            outputs = rm_out(vec_binned,m=m,kind=kind,return_borders=True)
            mask = (vec>=outputs[-1])&(vec<=outputs[-2])
           
        if who =='both':
            mask = rm_out(self.x, m=m, kind=kind)[0] & rm_out(self.y, m=m, kind=kind)[0] 
        self.mask = mask
        if replace:
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.yerr = self.yerr[mask]
            self.xerr = self.xerr[mask]

    def rms_w(self):
        if len(self.x)>1:
            self.rms = DescrStatsW(self.y,weights=1./self.yerr**2).std
            self.weighted_average = DescrStatsW(self.y,weights=1./self.yerr**2).mean
            self.stats['rms'] = self.rms
        else:
            self.rms=0
            self.weighted_average = self.y[0]