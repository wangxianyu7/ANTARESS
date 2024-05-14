#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,ScalarFormatter
import matplotlib.colors as colors
import numpy as np
from copy import deepcopy
from ..ANTARESS_general.utils import stop


#%% Symbols
degree_sign= u'\N{DEGREE SIGN}'


#%% Routines

def custom_axis(plt,ax=None,fig = None, x_range=None,y_range=None,z_range=None,position=None,colback=None,
		    x_mode=None,y_mode=None,z_mode=None,
		    x_title=None,y_title=None,z_title=None,x_title_dist=None,y_title_dist=None,z_title_dist=None,
		    no_xticks=False,no_yticks=False,no_zticks=True,
		    top_xticks=None,right_yticks=None,   
            hide_xticks = False,hide_yticks = False,		
		    xmajor_int=None,xminor_int=None,ymajor_int=None,yminor_int=None,zmajor_int=None,zminor_int=None,
		    xmajor_form=None,ymajor_form=None,zmajor_form=None,
		    xmajor_length=None,ymajor_length=None,zmajor_length=None,xminor_length=None,yminor_length=None,zminor_length=None,
		    font_size=None,font_thick=None,xfont_size=None,yfont_size=None,zfont_size=None,
		    axis_thick=None,xmajor_thick=None,xminor_thick=None,ymajor_thick=None,yminor_thick=None,zmajor_thick=None,zminor_thick=None,
		    dir_x=None,dir_y=None,dir_z=None,
            xtick_pad=None,ytick_pad=None,ztick_pad=None,
            xlab_col=None,ylab_col=None,zlab_col=None,
            hide_axis=None,
            right_axis=False,secy_title=None,secy_range=None,secy_title_dist=None,no_secyticks=None,secymajor_int=None,dir_secy=None,secyfont_size=None,
            secymajor_length=None,secymajor_thick=None,secyminor_length=None,secyminor_thick=None,secymajor_form=None,secyminor_int=None,secylab_col=None):
    r"""**Plot axis.**
    
    General routines to set up plot axis to default or user-selected values.  
    
    Args:
        TBD
    
    Returns:
        None
    
    """ 
		
    #Font
    #    - thick: 'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'					
    font_size_loc=font_size if font_size is not None else 10.
    font_thick_loc=font_thick if font_thick is not None else 'normal'
    plt.rc('font', size=font_size_loc,weight=font_thick_loc,**{'family':'sans-serif','sans-serif':['Helvetica']})
# 	plt.rc('text', usetex=True)
    plt.rcParams['pdf.fonttype'] = 42

    #Axis
    if ax==None:ax=plt.gca()

    #Axis frame position
    #    - corresponds to the corners of the image
    if position is not None:
        if fig is None:plt.subplots_adjust(left=position[0],bottom=position[1],right=position[2],top=position[3]) 
        else:fig.subplots_adjust(left=position[0],bottom=position[1],right=position[2],top=position[3]) 

    #Axis ranges
    if x_range is not None:ax.set_xlim([x_range[0], x_range[1]])
    else:x_range = ax.get_xlim()
    dx_range = x_range[1]-x_range[0]
    if y_range is not None:ax.set_ylim([y_range[0], y_range[1]])
    else:y_range = ax.get_ylim()
    dy_range = y_range[1]-y_range[0]
    if z_range is not None:ax.set_zlim([z_range[0], z_range[1]])

    #Set axis to log mode if required
    if x_mode=='log':
        ax.set_xscale('log')
    if y_mode=='log':
        ax.set_yscale('log')
    if z_mode=='log':
        ax.set_zscale('log')


    #Axis titles	
    xfont_size_loc=xfont_size if xfont_size is not None else 10.
    yfont_size_loc=yfont_size if yfont_size is not None else 10. 
    zfont_size_loc=zfont_size if zfont_size is not None else 10. 
    if x_title is not None:
            ax.set_xlabel(x_title,fontsize=xfont_size_loc,weight=font_thick_loc)
    if y_title is not None:
            if (right_yticks=='on'):  #set title to right axis          
                ax.set_ylabel(y_title,fontsize=yfont_size_loc,rotation=270,labelpad=22,weight=font_thick_loc)
            else:            
                ax.set_ylabel(y_title,fontsize=yfont_size_loc,weight=font_thick_loc) 
    if z_title is not None:   
            ax.set_zlabel(z_title,fontsize=zfont_size_loc,weight=font_thick_loc)
                
    #Axis title distance
    if x_title_dist is not None:ax.xaxis.labelpad = x_title_dist
    if y_title_dist is not None:ax.yaxis.labelpad = y_title_dist
    if z_title_dist is not None:ax.zaxis.labelpad = z_title_dist
    	
    #Axis background color
    if colback is not None:ax.set_facecolor(colback)

    #Axis thickness
    axis_thick_loc=axis_thick if axis_thick is not None else 1.
    if ('bottom' in list(ax.spines.keys())):ax.spines['bottom'].set_linewidth(axis_thick_loc)
    if ('top' in list(ax.spines.keys())):   ax.spines['top'].set_linewidth(axis_thick_loc)
    if ('left' in list(ax.spines.keys())):  ax.spines['left'].set_linewidth(axis_thick_loc)
    if ('right' in list(ax.spines.keys())): ax.spines['right'].set_linewidth(axis_thick_loc)

    #X ticks and label on top
    if (top_xticks=='on'):
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 	
    #Y ticks and label on right
    if (right_yticks=='on'):
         ax.yaxis.tick_right()
         ax.yaxis.set_label_position('right')
  
    #------------------------------------------------------------------------
    #X ticks (on by default)		
    if (no_xticks==False):
      
        #Interval between major ticks	
        if xmajor_int is not None:
            n_ticks = int(dx_range/xmajor_int)
            if n_ticks>100:xmajor_int = dx_range/100.
            ax.xaxis.set_major_locator(MultipleLocator(xmajor_int))

		#Direction of ticks
        if dir_x==None:dir_x='in' 

	    #Major ticks
        xmajor_length_loc=xmajor_length if xmajor_length is not None else 7
        xmajor_thick_loc=xmajor_thick if xmajor_thick is not None else 1.5		
        xtick_pad_loc=xtick_pad if xtick_pad is not None else 5  
        ax.tick_params('x', length=xmajor_length_loc, which='major',width=xmajor_thick_loc,
					  direction=dir_x, pad=xtick_pad_loc,labelsize=xfont_size_loc,top=True)
       
        #Minor ticks		
        xminor_length_loc=xminor_length if xminor_length is not None else xmajor_length_loc/2.	
        xminor_thick_loc=xminor_thick if xminor_thick is not None else 1.5		
        ax.tick_params('x', length=xminor_length_loc, which='minor',width=xminor_thick_loc,
					  direction=dir_x,labelsize=xfont_size_loc,top=True)
        
        #Major ticks label format		
        if xmajor_form is not None:ax.xaxis.set_major_formatter(FormatStrFormatter(xmajor_form))
        if x_mode=='log':
            if xmajor_form is not None:ax.xaxis.set_major_formatter(ScalarFormatter(xmajor_form))
            else:ax.xaxis.set_major_formatter(ScalarFormatter())
	
	    #Interval between minor ticks	
        if x_mode=='log':xminor_int=None
        if xminor_int is not None:ax.xaxis.set_minor_locator(MultipleLocator(xminor_int))		

        #Ticks labels color
        if xlab_col is not None:[i_col.set_color(xlab_col) for i_col in ax.get_xticklabels()]

        #Hide top ticks
        if hide_xticks:ax.xaxis.set_ticks_position('bottom')

    else:
        ax.set_xticks([])	
		
	#-----------------	
    #Y ticks (on by default)	
    if (no_yticks==False):

        #Interval between major ticks		
        if ymajor_int is not None:
            n_ticks = int(dy_range/ymajor_int)
            if n_ticks>100:ymajor_int = dy_range/100.
            ax.yaxis.set_major_locator(MultipleLocator(ymajor_int))	
       
		#Direction of ticks
        if dir_y==None:dir_y='in'         

	    #Major ticks
        ymajor_length_loc=ymajor_length if ymajor_length is not None else 7
        ymajor_thick_loc=ymajor_thick if ymajor_thick is not None else 1.5		
        ytick_pad_loc=ytick_pad if ytick_pad is not None else 5 		
        ax.tick_params('y', length=ymajor_length_loc, which='major',width=ymajor_thick_loc,
        					  direction=dir_y, pad=ytick_pad_loc,labelsize=yfont_size_loc, right=True)

	    #Minor ticks	
        yminor_length_loc=yminor_length if yminor_length is not None else ymajor_length_loc/2.
        yminor_thick_loc=yminor_thick if yminor_thick is not None else 1.5			
        ax.tick_params('y', length=yminor_length_loc, which='minor',width=yminor_thick_loc,
        					  direction=dir_y,labelsize=yfont_size_loc, right=True)

   	    #Major ticks label format		
        if ymajor_form is not None:ax.yaxis.set_major_formatter(FormatStrFormatter(ymajor_form))	
        if y_mode=='log':
            if ymajor_form is not None:ax.yaxis.set_major_formatter(ScalarFormatter(ymajor_form))
            else:ax.yaxis.set_major_formatter(ScalarFormatter())    

	    #Interval between minor ticks	
        if y_mode=='log':yminor_int=None
        if yminor_int is not None:
            n_ticks = int(ymajor_int/yminor_int)
            if n_ticks>50:yminor_int = ymajor_int/50.
            ax.yaxis.set_minor_locator(MultipleLocator(yminor_int))		

	    #Ticks labels color
        if ylab_col is not None:[i_col.set_color(ylab_col) for i_col in ax.get_yticklabels()]
        
        #Hide right ticks
        if hide_yticks:ax.yaxis.set_ticks_position('left')

    else:
        ax.set_yticks([])	

	#-----------------	
    #Z ticks (off by default)	
    if (no_zticks==False):
		
        #Interval between major ticks		
        if zmajor_int is not None:ax.zaxis.set_major_locator(MultipleLocator(zmajor_int))	

		#Direction of ticks
        if dir_z==None:dir_z='in' 

	      #Major ticks length	
        zmajor_length_loc=zmajor_length if zmajor_length is not None else 7
        zmajor_thick_loc=zmajor_thick if zmajor_thick is not None else 1.		
        ztick_pad_loc=ztick_pad if ztick_pad is not None else 5 		
        ax.tick_params('z', length=zmajor_length_loc, which='major',width=zmajor_thick_loc,
					  direction=dir_z, pad=ztick_pad_loc,labelsize=zfont_size_loc)
	
	      #Minor ticks length	
        zminor_length_loc=zminor_length if zminor_length is not None else zmajor_length_loc/2.
        zminor_thick_loc=zminor_thick if zminor_thick is not None else 1.5			
        ax.tick_params('z', length=zminor_length_loc, which='minor',width=zminor_thick_loc,
					  direction=dir_z,labelsize=zfont_size_loc)
	
	      #Major ticks label format		
        if zmajor_form is not None:
            majorFormatter = FormatStrFormatter(zmajor_form)
            ax.zaxis.set_major_formatter(majorFormatter)		
			
	      #Interval between minor ticks			
        if zminor_int is not None:	
            minorLocator   = MultipleLocator(zminor_int)
            ax.zaxis.set_minor_locator(minorLocator)

	      #Ticks labels color
        if zlab_col is not None:[i_col.set_color(zlab_col) for i_col in ax.get_zticklabels()]

    # else:
    #     ax.set_zticks([])	
		
    #------------------------------------------------------------------------
    #Secondary axis (right side)
    if right_axis==True:
         if right_yticks=='on':stop('Nominal axis already set to right axis')
         newaxvert = ax.twinx()
         newaxvert.yaxis.tick_right()
         newaxvert.yaxis.set_label_position('right') 

         #Title 
         if secy_title is not None:
             yfont_size_loc=secyfont_size if secyfont_size is not None else 10. 
             newaxvert.set_ylabel(secy_title,fontsize=yfont_size_loc,rotation=0) 
         if secy_title_dist is not None:newaxvert.yaxis.labelpad = secy_title_dist

         #Range              
         if secy_range is not None:newaxvert.set_ylim([secy_range[0], secy_range[1]])
    
    	   #-----------------	
         if (no_secyticks==None) or (no_secyticks==''):
             
             #Interval between major ticks
             if secymajor_int is not None:
                newaxvert.yaxis.set_major_locator(MultipleLocator(secymajor_int))	
                
             #Direction of ticks
             if dir_secy==None:dir_secy='in' 
    
    	      #Major ticks length	
             secymajor_length_loc=secymajor_length if secymajor_length is not None else 10
             secymajor_thick_loc=secymajor_thick if secymajor_thick is not None else 1.5			
             newaxvert.tick_params('y', length=secymajor_length_loc, which='major',width=secymajor_thick_loc,
    					  direction=dir_secy)
    	
    	      #Minor ticks length	
             secyminor_length_loc=secyminor_length if secyminor_length is not None else secymajor_length_loc/2.
             secyminor_thick_loc=secyminor_thick if secyminor_thick is not None else 1.5			
             newaxvert.tick_params('y', length=secyminor_length_loc, which='minor',width=secyminor_thick_loc,
    					  direction=dir_secy)
    	
    	      #Major ticks label format		
             if secymajor_form is not None:newaxvert.yaxis.set_major_formatter(FormatStrFormatter(secymajor_form))		
    			
    	      #Interval between minor ticks			
             if secyminor_int is not None:newaxvert.yaxis.set_minor_locator(MultipleLocator(secyminor_int))
    
    	      #Ticks labels color
             if secylab_col is not None:[i_col.set_color(secylab_col) for i_col in newaxvert.get_yticklabels()]
    
         else:
             newaxvert.set_yticks([])	

	#-------------------------------------------
      #Hide all axis
    if hide_axis:
          ax.xaxis.set_visible(False)
          ax.yaxis.set_visible(False)
          ax.spines['bottom'].set_color('white')  
          ax.spines['top'].set_color('white')  
          ax.spines['left'].set_color('white')  
          ax.spines['right'].set_color('white')  
    
              
    return None 
	

def scaled_title(sc_fact10,y_title):
    r"""**Title scaling.**
    
    Applies power-of-ten scaling to title value.  
    
    Args:
        sc_fact10 (int): power of ten scaling
        y_title (str): title
    
    Returns:
        y_title (str): title preceded by scaling value
    
    """ 
    if sc_fact10!=0.:
        sc_sign='-' if sc_fact10<0. else ''
        y_title='10$^{'+sc_sign+'%i' % abs(sc_fact10)+'}$ '+y_title
    return y_title                        



def autom_range_ext(ax_range_in,ax_min,ax_max,ext_fact=0.05):  
    r"""**Automatic range.**
    
    Defines axis range automatically.  
    
    Args:
        ax_range_in (list): axis range to be set, if user-defined
        ax_min (float): minimum value on the axis
        ax_max (float): maximum value on the axis
        ext_fact (float): extension of axis on both sides compared to min and max values
    
    Returns:
        ax_range (list): axis range
    
    """ 

    if ax_range_in is None:
        ax_range = np.array([ax_min,ax_max])   
        dx_range=ax_range[1]-ax_range[0]
        ax_range[0]-=ext_fact*dx_range
        ax_range[1]+=ext_fact*dx_range
        dx_range=ax_range[1]-ax_range[0]
    else:ax_range=ax_range_in
    return ax_range


def autom_tick_prop(dax_range):
    r"""**Automatic ticks.**
    
    Defines tick spacings and format automatically.  
    
    Args:
        dax_range (float): axis extension
    
    Returns:
        axmajor_int (float): major ticks spacing
        axminor_int (float): minor ticks spacing
        axmajor_form (str): major tick format
    
    """ 
    if   dax_range>1e11+0.1:axmajor_int,axminor_int,axmajor_form=5e10,1e10,'%.1e' 
    elif dax_range>1e10+0.1:axmajor_int,axminor_int,axmajor_form=5e9,1e9,'%.1e' 
    elif dax_range>1e9+0.1: axmajor_int,axminor_int,axmajor_form=5e8,1e8,'%.1e'     
    elif dax_range>1e8+0.1: axmajor_int,axminor_int,axmajor_form=5e7,1e7,'%.1e'     
    elif dax_range>1e7+0.1: axmajor_int,axminor_int,axmajor_form=5e6,1e6,'%.1e' 
    elif dax_range>1e6+0.1: axmajor_int,axminor_int,axmajor_form=5e5,1e5,'%.1e' 
    elif dax_range>1e5+0.1: axmajor_int,axminor_int,axmajor_form=5e4,1e4,'%.1e' 
    elif dax_range>1e4+0.1: axmajor_int,axminor_int,axmajor_form=5000.,1000.,'%.1e' 
    elif dax_range>5e3+0.1: axmajor_int,axminor_int,axmajor_form=1000.,500.,'%i'     
    elif dax_range>1e3+0.1: axmajor_int,axminor_int,axmajor_form=500.,100.,'%i' 
    elif dax_range>500.1:   axmajor_int,axminor_int,axmajor_form=200.,50.,'%i' 
    elif dax_range>200.1:   axmajor_int,axminor_int,axmajor_form=100.,10.,'%i'     
    elif dax_range>100.1:   axmajor_int,axminor_int,axmajor_form=50.,10.,'%i' 
    elif dax_range>50.1:    axmajor_int,axminor_int,axmajor_form=10.,1.,'%i' 
    elif dax_range>20.1:    axmajor_int,axminor_int,axmajor_form=5.,1.,'%i' 
    elif dax_range>15.1:    axmajor_int,axminor_int,axmajor_form=4.,1,'%i'    
    elif dax_range>10.1:    axmajor_int,axminor_int,axmajor_form=2.,1.,'%i'
    elif dax_range>6.1:     axmajor_int,axminor_int,axmajor_form=2.,1.,'%i' 
    elif dax_range>3.5:     axmajor_int,axminor_int,axmajor_form=1.,0.5,'%i'  
    elif dax_range>3.1:     axmajor_int,axminor_int,axmajor_form=1.,0.5,'%i'     
    elif dax_range>1.1:     axmajor_int,axminor_int,axmajor_form=5e-1,1e-1,'%.1f'     
    elif dax_range>7.1e-1:  axmajor_int,axminor_int,axmajor_form=2e-1,5e-2,'%.1f'     
    elif dax_range>5.1e-1:  axmajor_int,axminor_int,axmajor_form=2e-1,5e-2,'%.1f'    
    elif dax_range>3.1e-1:  axmajor_int,axminor_int,axmajor_form=1e-1,5e-2,'%.1f'     
    elif dax_range>1.1e-1:  axmajor_int,axminor_int,axmajor_form=5e-2,1e-2,'%.2f'
    elif dax_range>5.1e-2:  axmajor_int,axminor_int,axmajor_form=2e-2,1e-2,'%.2f'       
    elif dax_range>4.1e-2:  axmajor_int,axminor_int,axmajor_form=1e-2,5e-3,'%.2f'
    elif dax_range>3.1e-2:  axmajor_int,axminor_int,axmajor_form=1e-2,5e-3,'%.2f' 
    elif dax_range>1.1e-2:  axmajor_int,axminor_int,axmajor_form=5e-3,1e-3,'%.3f'
    elif dax_range>5.1e-3:  axmajor_int,axminor_int,axmajor_form=2e-3,1e-3,'%.3f'    
    elif dax_range>2.1e-3:  axmajor_int,axminor_int,axmajor_form=1e-3,5e-4,'%.3f'   
    elif dax_range>1.1e-3:  axmajor_int,axminor_int,axmajor_form=5e-4,1e-4,'%.4f'     
    elif dax_range>5.1e-4:  axmajor_int,axminor_int,axmajor_form=5e-4,1e-4,'%.4f' 
    elif dax_range>1.1e-4:  axmajor_int,axminor_int,axmajor_form=2e-4,1e-4,'%.4f' 
    else:axmajor_int,axminor_int,axmajor_form=None,None,None 
    return axmajor_int,axminor_int,axmajor_form    
    



def adjust_isosize(real_bounds,xpos0,ypos0,xpos1,ypos1,max_window_size):
    r"""**2D isotropic plot scaling.**
    
    Adjusts the size and position of the plot window, maintaining isotropy.
    
    Variables along the X and Y axis must have the same units.
    
    The lower left corner of the plot at `(xpos0,ypos0)` is taken as reference. 
    The `(x_pos1,y_pos1)` defines the maximum extension of the plot at the top right corner. 
    Depending on the respective width/height of the original plot, the larger side (within the axes) is scaled to `max_window_size` while the smallest side is then scaled by isotropy.    
    
    Args:
        TBD
    
    Returns:
        TBD
    
    """     

    #We use a square plot
    height=max_window_size					
    width=max_window_size	

    #Real ranges of the plot
    #    - real_bounds must be [min_x,max_x,min_y,max_y]
    #    - xpos0,ypos0,xpos1,ypos1 define the position of the axis limits within the plot window defined by width,height
    sub_width_set=(real_bounds[1]-real_bounds[0])
    sub_height_set=(real_bounds[3]-real_bounds[2])	
   
    #-----------------------
 
    #We calculate the position of the upper side of the axis, assuming isotropy
    #    - here we calculate the position of the axis (not the overall box) in fraction of 1
    #    - (sub_width_set) <-> (xpos1-xpos0) and (sub_height_set) <-> (ypos1-ypos0)		
    ypos1_temp=ypos0+(xpos1-xpos0)*(sub_height_set/sub_width_set)	
 
    #If the new upper side is higher than the required limit
    if (ypos1_temp>ypos1):   
	    #We set the upper limit to ypos1 and adjust instead the right side  
	    #    - in fraction of 1				
	    xpos1=xpos0+(ypos1-ypos0)*(sub_width_set/sub_height_set)			
    else:
	    ypos1=ypos1_temp
					
    return width,height,[xpos0,ypos0,xpos1,ypos1]


def adjust_3D_isosize(ax,x_range,y_range,z_range):
    r"""**3D isotropic plot scaling.**
    
    Adjusts the size of the plot window, maintaining isotropy.
      
    
    Args:
        TBD
    
    Returns:
        TBD

    """  
    max_range = np.array([x_range[1]-x_range[0],y_range[1]-y_range[0], z_range[1]-z_range[0]]).max() / 2.0	
    mean_x = 0.5*(x_range[1]+x_range[0])
    mean_y = 0.5*(y_range[1]+y_range[0])
    mean_z = 0.5*(z_range[1]+z_range[0])
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)
    
    return ax							


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    r"""**Colormap truncation.**
    
    Limits colormap to the chosen range for the plotted variable.
    
    Args:
        TBD
    
    Returns:
        TBD
    
    """   
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def stackrel(val,sub,sup,form):
    r"""**Stackrel truncation.**
    
    Print value with upper and lower error bars.
    
    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    form = deepcopy("{"+form+"}")
    return r"$"+form.format(val)+"\genfrac{}{}{0}{}{+"+form.format(sup)+"}{-"+form.format(sub)+"}$"



def mscatter(plt ,x, y, ax=None, m=None, **kw):
    r"""**Multiple markers.**
    
    Overrides plt.scatter to have various markers in one command.
    
    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    import matplotlib.markers as mmarkers

    if not ax: ax = plt.gca()

    sc = ax.scatter(x, y, **kw)
    
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
        
    return sc





def plot_shade_range(ax,shade_range,x_range_loc,y_range_loc,mode='fill',facecolor='grey',zorder=-1,alpha=0.2,compl=False):
    r"""**Shaded ranges.**
    
    Shades list of ranges.
    
    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    for i_int,bd_int in enumerate(shade_range):
        if compl:
            if (i_int==0 and bd_int[0]>x_range_loc[0]):
                bd_int_loc=[x_range_loc[0],bd_int[0]] #shade area before first interval                 
                if mode=='span':ax.axvspan(bd_int_loc[0],bd_int_loc[1], facecolor=facecolor, alpha=alpha,zorder=zorder)
                elif mode=='fill':ax.fill([bd_int_loc[0],bd_int_loc[1],bd_int_loc[1],bd_int_loc[0]],[y_range_loc[0],y_range_loc[0],y_range_loc[1],y_range_loc[1]], fill=True,color='dodgerblue',alpha=alpha,zorder=zorder,ls='')  
                elif mode=='hatch':ax.fill([bd_int_loc[0],bd_int_loc[1],bd_int_loc[1],bd_int_loc[0]],[y_range_loc[0],y_range_loc[0],y_range_loc[1],y_range_loc[1]], fill=False, hatch='\\',color=facecolor,zorder=zorder)         
            if (i_int==len(shade_range)-1 and bd_int[1]<x_range_loc[1]):
                bd_int_loc=[bd_int[1],x_range_loc[1]] #shade area after last interval
                if mode=='span':ax.axvspan(bd_int_loc[0],bd_int_loc[1], facecolor=facecolor, alpha=alpha,zorder=zorder)
                elif mode=='fill':ax.fill([bd_int_loc[0],bd_int_loc[1],bd_int_loc[1],bd_int_loc[0]],[y_range_loc[0],y_range_loc[0],y_range_loc[1],y_range_loc[1]], fill=True,color='dodgerblue',alpha=alpha,zorder=zorder,ls='')
                elif mode=='hatch':ax.fill([bd_int_loc[0],bd_int_loc[1],bd_int_loc[1],bd_int_loc[0]],[y_range_loc[0],y_range_loc[0],y_range_loc[1],y_range_loc[1]], fill=False, hatch='\\',color=facecolor,zorder=zorder)      
            if i_int>0:
                bd_int_loc=[shade_range[i_int-1][1],bd_int[0]]          #shade area between current and previous interval
                if mode=='span':ax.axvspan(bd_int_loc[0],bd_int_loc[1], facecolor=facecolor, alpha=alpha,zorder=zorder)
                elif mode=='fill':ax.fill([bd_int_loc[0],bd_int_loc[1],bd_int_loc[1],bd_int_loc[0]],[y_range_loc[0],y_range_loc[0],y_range_loc[1],y_range_loc[1]], fill=True,color='dodgerblue',alpha=alpha,zorder=zorder,ls='')
                elif mode=='hatch':ax.fill([bd_int_loc[0],bd_int_loc[1],bd_int_loc[1],bd_int_loc[0]],[y_range_loc[0],y_range_loc[0],y_range_loc[1],y_range_loc[1]], fill=False, hatch='\\',color=facecolor,zorder=zorder)        
        else:
            if (bd_int[1]>x_range_loc[0]) & (bd_int[0]<x_range_loc[1]):
                bd_int_loc=[np.max([x_range_loc[0],bd_int[0]]),np.min([x_range_loc[1],bd_int[1]])]
                if mode=='span':ax.axvspan(bd_int_loc[0],bd_int_loc[1], facecolor=facecolor, alpha=alpha,zorder=zorder)
                elif mode=='fill':ax.fill([bd_int_loc[0],bd_int_loc[1],bd_int_loc[1],bd_int_loc[0]],[y_range_loc[0],y_range_loc[0],y_range_loc[1],y_range_loc[1]], fill=True,color='dodgerblue',alpha=alpha,zorder=zorder,ls='')  
                elif mode=='hatch':ax.fill([bd_int_loc[0],bd_int_loc[1],bd_int_loc[1],bd_int_loc[0]],[y_range_loc[0],y_range_loc[0],y_range_loc[1],y_range_loc[1]], fill=False, hatch='\\',color=facecolor,zorder=zorder)         
  
    return None



