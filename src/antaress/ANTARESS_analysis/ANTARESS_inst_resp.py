#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import astropy.convolution.convolve as astro_conv
from copy import deepcopy
import bindensity as bind
from ..ANTARESS_general.utils import npint,stop,gen_specdopshift
from ..ANTARESS_general.constant_data import c_light


def flag_err_inst(inst): 
    r"""**Spectrograph error flag**

    Returns boolean tracing whether error tables are provided with an instrumental dataset.

    Args:
        inst (str) : ANTARESS name for the considered instrument configuration
    
    Returns:
        err_flag (bool) : True if errors are provided
    
    """     
    err_flag = {
        'CARMENES_VIS':True,
        'CARMENES_VIS_CCF':True,
        'CORALIE':False,  
        'ESPRESSO':True,
        'ESPRESSO_MR':True,
        'EXPRES':True,     
        'HARPN':True   ,
        'HARPS':True, 
        'IGRINS2_Blue':True,
        'IGRINS2_Red':True,
        'MAROONX_Blue':True,
        'MAROONX_Red':True,
        'MIKE_Blue':True,
        'MIKE_Red':True,
        'NIGHT':True,
        'NIRPS_HA':True,
        'NIRPS_HE':True,
        'SOPHIE_HE':False,
        'SOPHIE_HR':False, 
    } 
    if inst not in err_flag:stop('ERROR : define error status for '+inst+' in ANTARESS_inst_resp.py > flag_err_inst()')
    return err_flag[inst]

def return_spec_root(inst): 
    r"""**Spectrograph root name**

    Returns generic name of an instrument for a given configuration.

    Args:
        inst (str) : ANTARESS name for the considered instrument configuration
    
    Returns:
        spec_root (str) : generic instrument name
    
    """     
    spec_root = {
        'CARMENES_VIS':'CARMENES',
        'CORALIE':'CORALIE', 
        'ESPRESSO':'ESPRESSO',
        'ESPRESSO_MR':'ESPRESSO',
        'EXPRES':'EXPRES',    
        'HARPN':'HARPN',
        'HARPS':'HARPS',  
        'IGRINS2_Blue':'IGRINS2',
        'IGRINS2_Red':'IGRINS2',
        'MAROONX_Blue':'MAROONX',
        'MAROONX_Red':'MAROONX',
        'MIKE_Blue':'MIKE',
        'MIKE_Red':'MIKE',
        'NIRPS_HA':'NIRPS',
        'NIRPS_HE':'NIRPS',
        'SOPHIE_HE':'SOPHIE',
        'SOPHIE_HR':'SOPHIE', 
    } 
    if inst not in spec_root:stop('ERROR : define instrument root name for '+inst)
    return spec_root[inst]


def return_spec_nord(inst): 
    r"""**Number of orders**

    Returns number of spectral orders in spectrographs.

    Args:
        TBD
    
    Returns:
        TBD
    
    """     
    spec_nord = {
        'CARMENES_VIS':61,
        'CARMENES_VIS_CCF':61,
        'CORALIE':69,         
        'ESPRESSO':170,
        'ESPRESSO_MR':85,
        'EXPRES':86,
        'HARPN':69,
        'HARPS':71,  
        'IGRINS2_Blue':25,
        'IGRINS2_Red':22,
        'MAROONX_Blue':30,
        'MAROONX_Red':28,  
        'MIKE_Blue':37,
        'MIKE_Red':34,
        'NIGHT':71,     #UPDATE TO 1 WHEN FORMAT FINALIZED
        'NIRPS_HA':71,
        'NIRPS_HE':71,
        'SOPHIE_HE':39,
        'SOPHIE_HR':39,
    } 
    if inst not in spec_nord:stop('ERROR : define number of spectral orders in '+inst)
    return spec_nord[inst]
 

def return_SNR_orders(inst): 
    r"""**Orders for S/N ratio**

    Returns typical spectrograph order used for S/N measurements.

    Args:
        TBD
    
    Returns:
        TBD
    
    """       
    SNR_orders = {
        'CARMENES_VIS':[40],        
        'ESPRESSO_MR':[39],
        'ESPRESSO':[102,103], 
        'EXPRES':[14],           #562 nm         
        'HARPS':[49],
        'HARPN':[46],
        'NIRPS_HA':[57],
        'NIRPS_HE':[57],        #H band, 1.63 mic, order not affected by tellurics thus stable for SNR measurement
    }
    if inst not in SNR_orders:stop('ERROR : define spectral orders for S/R in '+inst)
    return SNR_orders[inst]

def return_cen_wav_ord(inst): 
    r"""**Orders central wavelength**

    Returns central wavelength of each spectrograph order.

    Args:
        inst (str) : ANTARESS name for the considered instrument configuration.
    
    Returns:
        cen_wav_ord (float, array) : central wavelength of each spectrograph order (in A)
    
    """       
    cen_wav_ord = {
        'ESPRESSO':10.*np.repeat(np.flip([784.45 ,774.52, 764.84, 755.40, 746.19, 737.19, 728.42 ,719.85, 711.48, 703.30, 695.31, 687.50, 679.86, 672.39, 665.08,
                                      657.93 ,650.93 ,644.08, 637.37, 630.80, 624.36, 618.05, 611.87, 605.81, 599.87, 594.05, 588.34, 582.74, 577.24, 571.84,
                                      566.55, 561.35, 556.25, 551.24, 546.31, 541.48, 536.73, 532.06, 527.48, 522.97, 522.97, 518.54, 514.18, 509.89, 505.68,
                                      501.53, 497.46, 493.44, 489.50 ,485.61, 481.79 ,478.02, 474.32, 470.67, 467.08, 463.54 ,460.05, 456.62, 453.24, 449.91,
                                      446.62, 443.39, 440.20, 437.05, 433.95 ,430.90, 427.88 ,424.91, 421.98, 419.09, 416.24, 413.43, 410.65, 407.91, 405.21,
                                      402.55, 399.92, 397.32, 394.76, 392.23, 389.73, 387.26, 384.83, 382.42, 380.04]),2),
        'HARPN':np.array([3896.9385, 3921.912 , 3947.2075, 3972.8315, 3998.7905, 4025.0913,4051.7397, 4078.7437, 4106.11  , 4133.8457, 4161.9595, 4190.458 ,
                          4219.349 , 4248.641 , 4278.3433, 4308.464 , 4339.011 , 4369.9946,4401.424 , 4433.3086, 4465.6587, 4498.484 , 4531.796 , 4565.6045,
                          4599.922 , 4634.759 , 4670.127 , 4706.0396, 4742.509 , 4779.548 ,4817.169 , 4855.388 , 4894.2188, 4933.675 , 4973.773 , 5014.5273,
                          5055.955 , 5098.074 , 5140.9004, 5184.452 , 5228.747 , 5273.8066,5319.6494, 5366.297 , 5413.7686, 5462.088 , 5511.2773, 5561.3613,
                          5612.3633, 5664.311 , 5717.2285, 5771.1436, 5826.086 , 5882.084 ,5939.1685, 5997.3726, 6056.7285, 6117.271 , 6179.0366, 6242.062 ,
                          6306.3867, 6372.0503, 6439.0957, 6507.568 , 6577.511 , 6648.9746,6722.0083, 6796.6646, 6872.9976]),
        'HARPS': np.array([3824.484, 3848.533, 3872.886, 3897.549, 3922.529, 3947.831, 3973.461,
                            3999.426, 4025.733, 4052.388, 4079.398, 4106.771, 4134.515, 4162.635, 4191.141,
                            4220.039, 4249.338, 4279.048, 4309.175, 4339.73 , 4370.722, 4402.16 , 4434.053,
                            4466.411, 4499.245, 4532.564, 4566.384, 4600.709, 4635.555, 4670.933, 4706.854,
                            4743.333, 4780.382, 4818.015, 4856.243, 4895.084, 4934.552, 4974.66 , 5015.426,
                            5056.866, 5098.997, 5141.835, 5185.398, 5229.708, 5274.78 , 5367.266, 5414.749,
                            5463.079, 5512.279, 5562.374, 5613.388, 5665.346, 5718.276, 5772.203, 5827.157,
                            5883.167, 5940.266, 5998.481, 6057.852, 6118.408, 6180.188, 6243.229, 6307.567,
                            6373.246, 6440.308, 6508.795, 6578.756, 6650.235, 6723.287, 6797.961, 6874.313]),                
        'CARMENES_VIS': np.array([5185.683800259293,5230.002514675239,5275.085301397565,5320.952091123464,5367.623513832012,
                                  5415.120929724138,5463.466461819996,5512.683030318242,5562.794388829332,5613.825162603174,5665.800888880403,5718.748059506192,5772.694165955971,
                                  5827.667746933892,5883.6984387171215,5940.817028432618,5999.055510467573,6058.447146230714,6119.026527498995,6180.829643603155,6243.893952726336,
                                  6308.2584576125455,6373.9637860064695,6441.052276173239,6509.5680678764,6579.557199224864,6651.067709835355,6724.149750796099,6798.855701960672,
                                  6875.24029714847,6953.36075788067,7033.276936338339,7115.051468293246,7198.749936832545,7284.4410477767,7372.196817776685,7462.09277617269,
                                  7554.208181803426,7648.626256074004,7745.4344337228085,7844.724632875495,7946.5935461392,8051.142954674568,8158.480067389824,8268.717887632938,
                                  8381.975610018162,8498.379050316224,8618.061111667314,8741.162290748554,8867.83122794844,8998.225306077602,9132.51130268578,9270.866101669406,
                                  9413.477470553611,9560.54491063029,9712.280588045605,9868.910354974394,10030.674871216968,10197.83083793165,10370.652356804127,10549.432429789025]),   
        'IGRINS2_Blue': np.array([14748.5,14865.8,14985.0,15106.3,15229.7,15355.2,15483.0,15613.0,15745.3,
                                  15880.0,16017.1,16156.8,16299.0,16443.9,16591.5,16741.9,16895.2,17051.5,
                                  17210.8,17373.2,17538.9,17707.9,17880.3,18056.3,18235.9]),
        'IGRINS2_Red': np.array([19396.5,19605.0,19818.1,20036.2,20259.2,20487.4,20720.9,2096.01,21204.9,
                                 21455.7,21712.7,21976.1,22246.2,22523.1,22807.2,23098.7,2339.80,23705.3,
                                 24021.0,24345.3,24678.8,25021.7]),
        'MAROONX_Blue': np.array([]),
        'MAROONX_Red': np.array([]),
        'MIKE_Blue': np.array([]),
        'MIKE_Red': np.array([]),
        'NIGHT': np.array([   9793.31830725 , 9859.95301593 , 9927.49931744 , 9995.97622395,10065.40380975, 10135.80171767, 10207.19048503, 10279.59121586,
                                10353.02577856, 10427.51660496, 10503.0864558 , 10579.75913144,10657.5589196 , 10736.51096413, 10816.64116324, 10897.97569896,
                                10980.54199941, 11064.36827222, 11149.48372245, 11235.91830631,11323.70300177, 11412.86989695, 11503.45207046, 11595.48314928,
                                11688.99793186, 11784.03276228, 11880.62559736, 11978.81474198,12078.63947148, 12180.1413448 , 12283.36401067, 12388.35148928,
                                12495.14844764, 12603.80175443, 12714.36064267, 12826.87609104,12941.40020952, 13057.98722241, 13176.69424781, 13297.58039703,
                                13420.70477935, 13546.12854527, 13673.91830748, 14072.1783805,14210.13779528, 14350.82710154, 14494.33106043, 14640.73394106,
                                14790.12380228, 14942.59417032, 15098.24089619, 15257.16474828,15419.46907017, 15585.26310369, 15754.66203085, 15927.78376522,
                                16104.75265749, 16285.69620041, 16470.75172016, 16660.06255469,16853.77645383, 17052.04937274, 17255.0418695 , 17462.92163421,
                                17675.87345983, 17894.08476988, 18117.75049316, 18347.07795686,18582.28588236, 18823.6030827 , 19071.27048963]),                        #UPDATE WHEN FINAL FORMAT DEFINED          
        'NIRPS_HA': np.array([   9793.31830725 , 9859.95301593 , 9927.49931744 , 9995.97622395,10065.40380975, 10135.80171767, 10207.19048503, 10279.59121586,
                                10353.02577856, 10427.51660496, 10503.0864558 , 10579.75913144,10657.5589196 , 10736.51096413, 10816.64116324, 10897.97569896,
                                10980.54199941, 11064.36827222, 11149.48372245, 11235.91830631,11323.70300177, 11412.86989695, 11503.45207046, 11595.48314928,
                                11688.99793186, 11784.03276228, 11880.62559736, 11978.81474198,12078.63947148, 12180.1413448 , 12283.36401067, 12388.35148928,
                                12495.14844764, 12603.80175443, 12714.36064267, 12826.87609104,12941.40020952, 13057.98722241, 13176.69424781, 13297.58039703,
                                13420.70477935, 13546.12854527, 13673.91830748, 14072.1783805,14210.13779528, 14350.82710154, 14494.33106043, 14640.73394106,
                                14790.12380228, 14942.59417032, 15098.24089619, 15257.16474828,15419.46907017, 15585.26310369, 15754.66203085, 15927.78376522,
                                16104.75265749, 16285.69620041, 16470.75172016, 16660.06255469,16853.77645383, 17052.04937274, 17255.0418695 , 17462.92163421,
                                17675.87345983, 17894.08476988, 18117.75049316, 18347.07795686,18582.28588236, 18823.6030827 , 19071.27048963]),
        'EXPRES': np.array([3827.18212341, 3851.28596202, 3875.69435186, 3900.41312958 ,3925.44828033,3950.80594266 ,3976.49241363 ,4002.51415404 ,4028.87779393 ,4055.59013823,
                              4082.65817269 ,4110.08906999 ,4137.89019612 ,4166.069117   ,4194.63360538,4223.591648   ,4252.95145307 ,4282.72145806 ,4312.91033776 ,4343.52701277,
                              4374.58065826 ,4406.08071317 ,4438.03688981 ,4470.45918379 ,4503.3578845,4536.74358599 ,4570.62719834 ,4605.01995956 ,4639.93344803 ,4675.37959548,
                              4711.3707006  ,4747.91944324 ,4785.03889938 ,4822.74255663 ,4861.04433066,4899.95858228 ,4939.50013539 ,4979.68429586 ,5020.52687121 ,5062.04419143,
                              5104.2531307  ,5147.17113026 ,5190.81622248 ,5235.20705609 ,5280.3629228,5326.3037852  ,5373.05030624 ,5420.62388022 ,5469.04666543 ,5518.34161856,
                              5568.53253104 ,5619.6440673  ,5671.70180522 ,5724.73227878 ,5778.76302323,5833.82262268 ,5889.94076065 ,5947.14827335 ,6005.47720621 ,6064.96087378,
                              6125.63392315 ,6187.53240127 ,6250.69382638 ,6315.15726382 ,6380.96340664,6448.15466126 ,6516.77523859 ,6586.87125105 ,6658.49081593 ,6731.68416551,
                              6806.50376456 ,6883.00443578 ,6961.24349375 ,7041.2808881  ,7123.17935679,7207.00459002 ,7292.82540601 ,7380.71393943 ,7470.74584357 ,7563.00050759,
                              7657.56128994 ,7754.51576965 ,7853.9560168  ,7955.97888421 ,8060.68632206,8168.18571772])
    }
    cen_wav_ord['NIRPS_HE'] = cen_wav_ord['NIRPS_HA'] 
    cen_wav_ord['CARMENES_VIS_CCF'] = cen_wav_ord['CARMENES_VIS']     
    if inst not in cen_wav_ord:stop('ERROR : define central wavelength of orders for '+inst)
    return cen_wav_ord[inst]

def return_edge_wav_ord(inst): 
    r"""**Orders edge wavelengths**

    Returns edge wavelengths of each spectrograph order.

    Args:
        inst (str) : ANTARESS name for the considered instrument configuration.
    
    Returns:
        edge_wav_ord (float, array) : edge wavelengths of each spectrograph order (in A)
    
    """       
    edge_wav_ord = {
        'ESPRESSO':10.*np.vstack((np.repeat(np.flip([778.98,769.11,759.48,750.10,740.95,732.01,723.29,714.78,706.46,698.34,690.40,682.63,675.04,667.62,660.36,653.26,646.31,639.50,632.84,626.31,619.92,613.65,607.52,601.50,595.60,589.82,584.14,578.58,
                                                    573.12,567.76,562.50,557.34,552.27,547.30,542.41,537.61,532.89,528.25,523.70,519.22,519.13,514.72,510.39,506.13,501.94,497.82,493.76,489.78,485.85,481.99,478.19,474.45,470.77,467.15,463.57,460.06,
                                                    456.60,453.19,449.83,446.52,443.26,440.04,436.87,433.75,430.67,427.64,424.64,421.69,418.78,415.91,413.08,410.29,407.53,404.82,402.13,399.48,396.87,394.30,391.75,389.24,386.76,384.31,381.89,379.50,377.15]),2),
                                 np.repeat(np.flip([790.64,780.65,770.89,761.38,752.10,743.04,734.20,725.57,717.13,708.89,700.84,692.97,685.27,677.74,670.38,663.18,656.12,649.22,642.46,635.83,629.35,622.99,616.76,610.66,604.67,598.80,593.05,587.40,
                                                    581.86,576.42,571.08,565.85,560.71,555.65,550.69,545.82,541.03,536.33,531.71,527.16,527.03,522.57,518.18,513.87,509.63,505.45,501.35,497.31,493.33,489.42,485.57,481.78,478.05,474.38,470.76,467.19,
                                                    463.68,460.22,456.82,453.46,450.15,446.89,443.68,440.51,437.39,434.31,431.27,428.28,425.33,422.42,419.54,416.71,413.91,411.15,408.43,405.75,403.10,400.48,397.90,395.35,392.83,390.34,387.89,385.47,383.07]),2)))

    }
    if inst not in edge_wav_ord:stop('ERROR : define edge wavelengths of orders for '+inst)
    return edge_wav_ord[inst]


def return_pix_size(inst): 
    r"""**Spectrograph sampling**

    Returns width of detector pixel in rv space (km/s).
    
    .. math:: 
       \Delta \lambda = \lambda_\mathrm{ref} \Delta v/c 

    Args:
        TBD
    
    Returns:
        TBD
    
    """             
    pix_size = {

        #CARMENES   
        #    optical resolving power = 93400 -> deltav_instru = 3.2 km/s   
        #    - 2.8 pixel / FWHM, so that pixel size = 1.1317 km/s             
        'CARMENES_VIS':1.1317,
        'CARMENES_VIS_CCF':1.1317,
        #    near-infrared resolving power = 80400 -> deltav_instru = 3.72876 km/s   
        #    - 2.3 pixel / FWHM, so that pixel size = 1.62 km/s   
        'CARMENES_NIR':1.1317,
            
        #CORALIE:
        #    ordre 10:  deltaV = 1.7240 km/s
        #    ordre 35:  deltaV = 1.7315 km/s
        #    ordre 60:  deltaV = 1.7326 km/s
        #    resolving power = 55000 -> deltav_instru = 5.45 km/s          
        'CORALIE':1.73,
        
        #ESPRESSO in HR mode
        #    - pixel size = 0.5 km/s
        # 0.01 A at 6000A
        #    - resolving power = 140000 -> deltav_instru = 2.1km/s           
        'ESPRESSO':0.5,

        #ESPRESSO in MR mode
        'ESPRESSO_MR':1.,

        'EXPRES':0.5,

        'GIANO':3.1280284,
        
        #HARPS-N or HARPS: pix_size = 0.016 A ~ 0.8 km/s at 5890 A         
        #    - resolving power = 120000 -> deltav_instru = 2.6km/s         
        'HARPN':0.82,
        'HARPS':0.82,            

        'IRD':2.08442,

        #IGRINS2
        #    blue arm resolving power = 50000 -> deltav_instru = 5.99584 km/s   
        #    - 3 pixel / FWHM, so that pixel size = 1.9986 km/s             
        'IGRINS2_Blue':1.9986,
        #    red arm resolving power = 45000 -> deltav_instru = 6.66204 km/s   
        #    - 3 pixel / FWHM, so that pixel size = 2.22068 km/s             
        'IGRINS2_Red':2.22068,

        #MAROON-X
        #    resolving power = 85000 -> deltav_instru = 3.527 km/s  
        #    - 3.5 pixel / FWHM, so that pixel size = 1.0077 km/s              
        'MAROONX_Blue':1.0077,
        'MAROONX_Red':1.0077,

        #MIKE
        #    blue arm : pix_size ~= 0.02 A -> 1.43 km/s at 4190 A 
        'MIKE_Blue':1.43,
        #    red arm : pix_size ~= 0.05 A -> 2.10 km/s at 7130 A 
        'MIKE_Red':2.10,

        #NIGHT
        'NIGHT':1.1,  

        #NIRPS
        'NIRPS_HA':0.93,
        'NIRPS_HE':0.93,          
        
        #Sophie HE mod: pix_size = 0.0275 A ~ 1.4 km/s at 5890 A 
        'SOPHIE_HE':1.4,  
        
        #STIS E230M
        #    - size varies in wavelength but is roughly constant in velocity:
        # 0.0496 A at 3021A (=4.920 km/s)
        # 0.0375 A at 2274A (=4.944 km/s)   
        #      we take pix_size ~ 4.93 km/s    
        #    - with resolving power = 30000 -> deltav_instru = 9.9 km/s (2 bins)
        'STIS_E230M':4.93,   
        
        #STIS G750L
        #    - size varies in radial velocity and remains roughly constant in wavelength:
        # 195 km/s at 5300 A 
        # 275 km/s at 7500 A
        # ie about 4.87 A 
        #      we take an average pix_size ~ 235 km/s       
        'STIS_G750L':235.,  
           
    } 
    if inst not in pix_size:stop('ERROR : define pixel size for "'+inst+'" in ANTARESS_inst_resp.py > return_pix_size()')
    return pix_size[inst]


def resamp_st_prof_tab(inst,vis,isub,fixed_args,gen_dic,nexp,rv_osamp_line_mod):
    r"""**Resampled spectral profile table**

    Defines resampled spectral grid for line profile calculations.
    Theoretical profiles are directly calculated at the requested resolution, measured profiles are extracted at their native resolution.

    Args:
        inst (str) : Instrument considered.
        vis (str) : Visit considered.
        isub (int) : Index of the exposure considered.
        fixed_args (dict) : Parameters of the profiles considered.
        gen_dic (dict) : General dictionary.
        nexp (int) : Number of exposures in the visit considered.
        rv_osamp_line_mode (float) : RV-space oversampling factor.
    
    Returns:
        TBD
    
    """
    if inst is None:edge_bins = fixed_args['edge_bins']
    else:edge_bins = fixed_args['edge_bins'][inst][vis][isub]

    #Resampled model table
    #    - defined in RV space if relevant for spectral data
    rv_resamp = deepcopy(rv_osamp_line_mod)
    if fixed_args['spec2rv']:
        edge_bins_RV = c_light*((edge_bins/fixed_args['line_trans']) - 1.) 
        if rv_resamp is None:rv_resamp = np.mean(edge_bins_RV[1::]-edge_bins_RV[0:-1]    )
        min_x = edge_bins_RV[0]
        max_x = edge_bins_RV[-1]    
    else:
        min_x = edge_bins[0]
        max_x = edge_bins[-1]    
    delta_x = (max_x-min_x)
    
    #Extend definition range to allow for convolution
    min_x-=0.05*delta_x
    max_x+=0.05*delta_x
    ncen_bins_HR = int(np.ceil(round((max_x-min_x)/rv_resamp)))
    dx_HR=(max_x-min_x)/ncen_bins_HR

    #Define and attribute table for current exposure
    dic_exp = {}
    dic_exp['edge_bins_HR']=min_x + dx_HR*np.arange(ncen_bins_HR+1)
    dic_exp['cen_bins_HR']=0.5*(dic_exp['edge_bins_HR'][1::]+dic_exp['edge_bins_HR'][0:-1])  
    dic_exp['dcen_bins_HR']= dic_exp['edge_bins_HR'][1::]-dic_exp['edge_bins_HR'][0:-1]   
    if inst is None: 
        for key in dic_exp:fixed_args[key] = dic_exp[key]
        fixed_args['ncen_bins_HR']=ncen_bins_HR
        fixed_args['dim_exp_HR']=deepcopy(fixed_args['dim_exp'])
        fixed_args['dim_exp_HR'][1] = ncen_bins_HR
    else:
        for key in ['cen_bins_HR','edge_bins_HR','dcen_bins_HR','dim_exp_HR']:
            if key not in fixed_args:fixed_args[key]={inst:{vis:np.zeros(nexp,dtype=object)}}
        for key in dic_exp:fixed_args[key][inst][vis][isub]  = dic_exp[key]   
        if 'ncen_bins_HR' not in fixed_args:
            fixed_args['ncen_bins_HR']={inst:{vis:ncen_bins_HR}}
        if 'dim_exp_HR' not in fixed_args:
            fixed_args['dim_exp_HR']={inst:{vis:fixed_args['dim_exp'][inst][vis]}} 
            fixed_args['dim_exp_HR'][inst][vis][1] = ncen_bins_HR        

    return None


def def_st_prof_tab(inst,vis,isub,args):
    r"""**Spectral profile table attribution**

    Attributes original or resampled spectral grid for line profile calculations.

    Args:
        TBD
    
    Returns:
        TBD
    
    """  
    args_exp = deepcopy(args)
    if args['resamp']:suff='_HR'
    else:suff=''
    if (inst is None):
        for key in ['edge_bins','cen_bins','dcen_bins','ncen_bins','dim_exp']:args_exp[key] = args[key+suff]
    else:
        for key in ['edge_bins','cen_bins','dcen_bins']:args_exp[key] = args[key+suff][inst][vis][isub]
        for key in ['ncen_bins','dim_exp']:args_exp[key] = args[key+suff][inst][vis]
        if (args['mode']=='ana'):args_exp['func_prof'] = args['func_prof'][inst]
        
    return args_exp



def return_resolv(inst): 
    r"""**Spectral resolving power**

    Returns resolving power of a given spectrograph.

    Args:
        inst (str) : Instrument / spectrograph considered.
    
    Returns:
        inst_res (float) : Resolving power of the spectrograph.
    
    """
    inst_res = {
        'CARMENES_NIR':80400.,
        'CARMENES_VIS':94600.,
        'CORALIE':55000.,
        'ESPRESSO':140000.,
        'ESPRESSO_MR':70000.,
        'EXPRES':137500.,
        'GIANO':50000.,
        'MIKE_Red':83000.,
        'MIKE_Blue':65000.,
        'HARPN':120000.,
        'HARPS':120000.,        
        'IRD':70000.,
        'MAROONX_Red':85000.,
        'MAROONX_Blue':85000.,
        'IGRINS2_Red':50000.,
        'IGRINS2_Blue':45000.,
        'NIGHT':70000.,
        'NIRPS_HE':75000.,
        'NIRPS_HA':88000.,
        'NIRSPEC':25000.,
        'SOPHIE_HR':75000.,  
        'SOPHIE_HE':40000.,  
        'STIS_E230M':30000.,     
        'STIS_G750L':1280.,         
    }
    if inst not in inst_res:stop('ERROR : define spectral resolving power for '+inst)
    return inst_res[inst]

def calc_FWHM_inst(inst,w_c):
    r"""**Spectral resolution**

    Returns FWHM of a Gaussian approximating the LSF for a given resolving power, in rv or wavelength space
    
    .. math:: 
       \Delta v &= c / R   \\
       \Delta \lambda &= \lambda_\mathrm{ref}/R = \lambda_\mathrm{ref} \Delta v/c 
     
    Args:
        TBD
    
    Returns:
        TBD
    
    """
    FWHM_inst = w_c/return_resolv(inst)
    return FWHM_inst
  
    
def get_FWHM_inst(inst,fixed_args,cen_bins):
    r"""**Effective spectral resolution**

    Returns FWHM relevant to convolve the processed data 
    
     - in rv space for analytical profiles
     - in wavelength space for theoretical profiles
     - disabled if measured profiles as used as proxy for intrinsic profiles

    Args:
        TBD
    
    Returns:
        TBD
    
    """     
    #Reference point
    if (fixed_args['mode']=='ana') or fixed_args['spec2rv']:fixed_args['ref_conv'] = c_light
    elif fixed_args['mode']=='theo':fixed_args['ref_conv'] = cen_bins[int(len(cen_bins)/2)]    
    
    #Instrumental response 
    if (fixed_args['mode']=='Intrbin'):FWHM_inst = None
    else:FWHM_inst = calc_FWHM_inst(inst,fixed_args['ref_conv'])      
    
    return FWHM_inst
        

def convol_prof(prof_in,cen_bins,FWHM):
    r"""**Instrumental convolution**

    Convolves input profile with spectrograph LSF.
    Profile must be defined on a uniform spectral grid.

    Args:
        prof_in (array, float) : original spectral profile.
        cen_bins (array, float) : wavelength grid over which `prof_in` is defined.
        FWHM (float) : width of the Gaussian LSF used to convolve `prof_in`.
    
    Returns:
        prof_conv (array, float) : convolved spectral profile.
    
    """  
    
    #Half number of pixels in the kernel table at the resolution of the band spectrum
    #    - a range of 3.15 x FWHM ( = 3.15*2*sqrt(2*ln(2)) sigma = 7.42 sigma ) contains 99.98% of a Gaussian LSF integral
    #      we conservatively use a kernel covering 4.25 x FWHM / dbin pixels, ie 2.125 FWHM or 5 sigma on each side   
    dbins = cen_bins[1]-cen_bins[0]
    hnkern=npint(np.ceil(2.125*FWHM/dbins)+1)
    
    #Centered spectral table with same pixel widths as the band spectrum the kernel is associated to
    cen_bins_kernel=dbins*np.arange(-hnkern,hnkern+1)

    #Discrete Gaussian kernel 
    gauss_psf=np.exp(-np.power(  2.*np.sqrt(np.log(2.))*cen_bins_kernel/FWHM   ,2.))

    #Normalization
    gauss_kernel=gauss_psf/np.sum(gauss_psf)        

    #Convolution by the instrumental LSF   
    #    - bins must have the same size in a given table
    prof_conv=astro_conv(prof_in,gauss_kernel,boundary='extend')

    return prof_conv


def cond_conv_st_prof_tab(rv_osamp_line_mod,fixed_args,data_type):
    r"""**Spectral conversion and resampling**

    Enables/disables operations.

    Args:
        TBD
    
    Returns:
        TBD
    
    """  
    
    #Spectral oversampling
    if (fixed_args['mode']=='ana') and (rv_osamp_line_mod is not None):fixed_args['resamp'] = True
    else:fixed_args['resamp'] = False
    
    #Activate RV mode for analytical models of spectral profiles
    #    - theoretical profiles are processed in wavelength space
    #      measured profiles are processed in their space of origin
    #      analytical profiles are processed in RV space, and needs conversion back to wavelength space if data is in spectral mode  
    #    - since spectral tables will not have constant pixel size (required for model computation) in RV space, we activate the resampling mode so that all models will be calculated on this table and then resampled in spectral space,
    # rather than resampling the exposure in RV space
    if ('spec' in data_type) and (fixed_args['mode']=='ana'):
        fixed_args['spec2rv'] = True
        fixed_args['resamp'] = True 
        if fixed_args['line_trans'] is None:stop('Define "line_trans" to fit spectral data with "mode = ana"')
    else:fixed_args['spec2rv'] = False
    
    return None


def conv_st_prof_tab(inst,vis,isub,args,args_exp,line_mod_in,FWHM_inst):
    r"""**Spectral convolution, conversion, and resampling**

    Applies operations.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    
    #Convolve with instrumental response 
    #    - performed on table with constant bin size
    if FWHM_inst is None:line_mod_out = line_mod_in
    else:line_mod_out =  convol_prof( line_mod_in,args_exp['cen_bins'],FWHM_inst)

    #Convert table from RV to spectral space if relevant
    #    - w = w0*(1+rv/c)
    if args['spec2rv']:
        args_exp['edge_bins'] = args['line_trans']*gen_specdopshift(args_exp['edge_bins'])  
        args_exp['cen_bins'] = args['line_trans']*gen_specdopshift(args_exp['cen_bins'])  

    #Resample model on observed table if oversampling
    if args['resamp']:
        if inst is None:edge_bins_mod_out = args['edge_bins']
        else:edge_bins_mod_out = args['edge_bins'][inst][vis][isub]
        line_mod_out = bind.resampling(edge_bins_mod_out,args_exp['edge_bins'],line_mod_out, kind=args['resamp_mode'])       

    return line_mod_out

