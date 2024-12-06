import cdsapi
import numpy as np
import xarray as xr
import os,sys
from netCDF4 import Dataset
from numpy.polynomial import polynomial as P
from scipy.signal import argrelextrema
import math
import pyproj

c = cdsapi.Client()

def get_height_from_ml_cerra(a_coef,b_coef,sfc_p,t_full_lev,q_full_lev):
    # a_coef and b_coef are the hybrid levels coefficients at half levels
    # sfc_p is the surface pressure
    # t_full_lev is the temperature at full levels
    # q_full_lev is the specific humidity at full levels
    epsilon = 0.622
    epsilon1 = 1.0/epsilon - 1.0
    R = 287.058
    g = 9.80665
    # Calculate pressure at half levels (3d field)
    p_half_lev=a_coef[:,np.newaxis,np.newaxis]+b_coef[:,np.newaxis,np.newaxis]*sfc_p[np.newaxis,:,:]
    # Initialisation of heights at half (2d because of vertical integration) and full levels (3d)
    h_full_lev = np.zeros((np.shape(p_half_lev)[0]-1,np.shape(p_half_lev)[1],np.shape(p_half_lev)[2]))
    h_half_lev = np.zeros((np.shape(p_half_lev)[1],np.shape(p_half_lev)[2]))
    max_k=len(a_coef)-1 # highest level minus 1
    for k in range(max_k-1, -1, -1): # to -1 because I do not have the top level in my data to decrease file size
        virt_temp_k = (1.0 + epsilon1*q_full_lev[k,:,:])*t_full_lev[k,:,:]
        dlnp_k = np.log(p_half_lev[k+1,:,:] / p_half_lev[k,:,:])
        dp_k = p_half_lev[k+1,:,:] - p_half_lev[k,:,:]
        alpha_k = 1.0 - p_half_lev[k,:,:]*dlnp_k/dp_k
        h_full_lev[k,:,:] = h_half_lev + alpha_k*R*virt_temp_k/g
        h_half_lev += R*virt_temp_k*dlnp_k/g
        del virt_temp_k,dlnp_k,dp_k,alpha_k
    # Reverse the array to get increasing heights along the vertical axis
    return h_full_lev[::-1,:,:]


# Subroutine for linear interpolation of a 1d-array
def lin_int(x1,x2,y1,y2,xtarget):
    # x should be the height and y the wind speed
    y=y1+(xtarget-x1)*((y2-y1)/(x2-x1))
    return y

# Subroutine giving the number of days in a specific month of a specific year 
def nb_days_month(year,month):
   if ((month==1) | (month==3) | (month==5) | (month==7) | (month==8) | (month==10) | (month==12)):
      nday=31
   if ((month==4) | (month==6) | (month==9) | (month==11)):
      nday=30
   if month==2:
      if year%4==0:
         nday=29
      else:
         nday=28
   return nday


# Subroutine detecting low-level jets from a wind profile (for one grid point)
# returns a boolean and the model height of max
# Method inspired from Tuononen et al. 2015 (https://rmets.onlinelibrary.wiley.com/doi/10.1002/asl.587)
# Usage: bool = detect_llj(ws_vec,height_vec,hlim)
# No wind 0 at surface, val at 1500 m interpolated, interpolation of max
def detect_llj(ws_vec,height_vec,hlim):
    # ws_vec: wind speed as a function of height (ascending order), 1d
    # height_vec_vec: height as a function of height (ascending order), 1d
    bl=False
    k1=np.nan
    hh=np.where(height_vec<hlim)[0][-1]
    # Calculate wind speed at height 1500 m
    if (int(height_vec[hh+1]*10.0)==15000):
        ws_1500m=ws_vec[hh+1]
        #print('case1')
    elif (int(height_vec[hh]*10.0)==15000):
        ws_1500m=ws_vec[hh]
        #print('case2')
    else:
        ws_1500m=lin_int(height_vec[hh],height_vec[hh+1],ws_vec[hh],ws_vec[hh+1],1500.0)
        t1=((ws_1500m>=ws_vec[hh]) & (ws_1500m<=ws_vec[hh+1]))
        t2=((ws_1500m<=ws_vec[hh]) & (ws_1500m>=ws_vec[hh+1]))
        #print('case3')
        if ~(t1 | t2):
            print(ws_vec[hh],ws_1500m,ws_vec[hh+1])
            print(height_vec[hh],height_vec[hh+1],ws_vec[hh],ws_vec[hh+1])
            print('Problem with the interpolation')
            sys.exit()
    # I concatenate the wind speed at 1500 m to the profile
    ws_vec1=np.hstack((ws_vec[0:hh+1],ws_1500m))
    hgt_vec1=np.hstack((ws_vec[0:hh+1],1500.0))
    # Get all local maxima and minima
    extr_x=argrelextrema(ws_vec1,np.greater,axis=0)[0]
    extr_n=argrelextrema(ws_vec1,np.less,axis=0)[0]
    # If there is no minimum, it means that either there is only one big maximum (or there are identical values which prevents the detection of a minimum)
    # If there is no maximum, it means that either there is no maximum (or there are identical values which prevents the detection of a maximum)
    # Add the bottom and top indices to the minima list
    # even if they are not minima
    extr_n=np.hstack(([0],extr_n,[hh+1]))
    chk=0
    if len(extr_x)>0:
        # Loop over the maxima
        for k in range(len(extr_x)):
            if chk==0:
                a=0
                mn_low=-1
                mn_up=-1
                # Loop over the minima to find surrounding minima to the maximum
                for l in range(len(extr_n)-1):
                    if ((extr_n[l]<extr_x[k]) & (extr_n[l+1]>extr_x[k])):
                        if ((ws_vec1[extr_n[l]]<=ws_vec1[extr_x[k]]) & (ws_vec1[extr_n[l+1]]<=ws_vec1[extr_x[k]])):
                            mn_low=extr_n[l]
                            mn_up=extr_n[l+1]
                            w_max=ws_vec1[extr_x[k]]
                            w_low=ws_vec1[mn_low]
                            w_up=ws_vec1[mn_up]
                            # Condition 1: the max ws must be larger by 2 m/s compared to the surrounding minima
                            # Condition 2: the max must be larger by 25% compared to the surrounding minima
                            cond1=((w_max>=(w_low+2.0)) & (w_max>=(w_up+2.0)))
                            cond2=((w_max>=(w_low*1.25)) & (w_max>=(w_up*1.25)))
                            # I want to keep the lower jet (chk=0 because the loop over k starts from the bottom of the column)
                            if ((cond1 & cond2) & (chk==0)):
                                bl=True
                                k1=extr_x[k]
                                chk=chk+1
                            else:
                                bl=False
    return bl,k1


# Subroutine calculating a 2nd-degree polynomial fit and returning the position and intensity of the maximum
# Note the input xval,yval must be 1d and contain at least 3 points
# Usage: height_max,speed_max=fit2_max(x_input,y_input)
def fit2_max(xval,yval):
    fit = P.Polynomial.fit(xval,yval,2,full=False)
    # Retrieve coefficients
    c=fit.convert().coef
    # Calculating the curve coordinates using h_fit as x
    # h_fit is between the rounded hundreds around the height of max
    h_fit=np.arange(rounddown(xval[0]),roundup(xval[2])+1,1)
    ws_fit=c[0]+c[1]*h_fit+c[2]*h_fit**2
    # Get  position and intensity of new max
    pos=h_fit[np.argmax(ws_fit)]
    spd=ws_fit[np.argmax(ws_fit)]
    return pos,spd



# Subroutine rounding to the nearest integer
# Taken from https://stackoverflow.com/a/59576808
def proper_round(a):
    '''
    given any real number 'a' returns an integer closest to 'a'
    '''
    a_ceil = np.ceil(a)
    a_floor = np.floor(a)
    if np.abs(a_ceil - a) < np.abs(a_floor - a):
        return int(a_ceil)
    else:
        return int(a_floor)


# Round up to next hundred
# Taken from https://stackoverflow.com/questions/8866046/python-round-up-integer-to-next-hundred
def roundup(x):
    return int(math.ceil(x / 100.0)) * 100

# Round down to hundred
def rounddown(x):
    return int(math.floor(x / 100.0)) * 100


args=sys.argv
yr1=int(args[1])

path="./"
patho="LLJ_CERRA/"

# Thresholds
hgt_lim=1500.0 # 1.5 km
diff_speed=2.0 # in m/s


# Arrays with the hybrid levels coefficients taken from
# https://confluence.ecmwf.int/download/attachments/272323672/CERRA_vertical_level_information_update-2.xlsx?version=1&modificationDate=1679055050557&api=v2
a_coeff=np.array([200,400.01814705,600.20531049,800.84865376,1002.322823,1205.07225597,1409.60142123,1616.46831962,1826.27974226,2039.68760852,2245.13907287,2443.60748493,2638.86709028,2832.49979775,3025.54208774,3218.76943582,3412.79508122,3608.11708474,3805.13814403,4004.17960096,4205.48475872,4409.22390201,4615.4922401,4824.31170008,5035.6267318,5249.30485939,5465.13198993,5682.81092589,5901.95961973,6122.10785294,6342.69623499,6563.07174029,6782.49068449,7000.11529682,7215.01512243,7426.16512425,7632.45145456,7832.67141334,8025.53851614,8209.68507814,8383.67219371,8545.99501098,8695.09306301,8829.3592974,8947.15547443,9046.82455727,9126.70755184,9185.16052004,9220.57589837,9231.4030259,9216.17169418,9173.51743274,9102.20775787,9000.26393221,8865.57202986,8697.43902162,8495.99693212,8262.15051152,7997.52200414,7704.36903898,7385.48765158,7044.12542903,6683.87481353,6308.56338352,5922.17017957,5528.71189257,5132.17168475,4736.41228748,4345.09826733,3961.65113906,3589.19226781,3230.4972137,2887.98346849,2563.67370504,2259.20166935,1975.80406433,1714.32229526,1475.22654697,1258.62902667,1064.30484469,891.729936,740.10354692,608.39154257,495.36036972,399.61528282,319.64604924,253.86712551,199.59594044,154.4473517,117.31007387,87.15621236,63.04026119,44.09799931,29.54581996,18.67798233,10.86389822,5.54557311,2.2339457,0.50576257,0,0,0,0,0,0,0])
b_coeff=np.array([0,0,0,0,0,0,0,0,0,0,0.00012087,0.00036024,0.00068848,0.00109795,0.00158673,0.0021558,0.00280805,0.00354784,0.00438071,0.00531335,0.00635345,0.00750973,0.00879187,0.01021056,0.0117775,0.01350539,0.01540798,0.01750007,0.01979753,0.02231733,0.02507755,0.02809738,0.03139715,0.03499832,0.03892349,0.04319635,0.04784172,0.05288546,0.05835451,0.06427671,0.07068086,0.0775966,0.0850543,0.09308493,0.10171999,0.11099134,0.12093108,0.13157121,0.14294366,0.15507994,0.16801097,0.18176666,0.19637594,0.21199045,0.22871468,0.24650008,0.26528952,0.28501868,0.30561583,0.32700299,0.34909716,0.37181059,0.39505208,0.41872849,0.44274471,0.46700574,0.49141652,0.51588321,0.54031428,0.56462024,0.58871454,0.61251444,0.63594022,0.65891657,0.68137161,0.70323743,0.72445035,0.74495,0.76467956,0.78358586,0.80161826,0.81872925,0.83487334,0.85000699,0.86408855,0.87707721,0.88893297,0.89984284,0.91003744,0.91954285,0.9283847,0.93658848,0.94417971,0.95118379,0.95762625,0.96353307,0.96893055,0.97384594,0.97830746,0.98234491,0.9859945,0.98929404,0.99228188,0.99500549,0.99753269,1])
# I selected only the upper levels (from 60 to 106)
a_coeff=a_coeff[60::]
b_coeff=b_coeff[60::]

# Limits of the domain (reduced to cover only Scandinavia)
x_l=220
x_u=900
y_l=500
y_u=1069

for yr in range(yr1,yr1+1):
    for mon in range(1,13):
        nbd=nb_days_month(yr,mon)
        for day in range(1,nbd+1):
            c.retrieve(
                'reanalysis-cerra-model-levels',
                {
                    'format': 'netcdf',
                    'variable': [
                        'specific_humidity', 'temperature', 'u_component_of_wind',
                        'v_component_of_wind',
                    ],
                    'model_level': [
                        '60', '61', '62',
                        '63', '64', '65',
                        '66', '67', '68',
                        '69', '70', '71',
                        '72', '73', '74',
                        '75', '76', '77',
                        '78', '79', '80',
                        '81', '82', '83',
                        '84', '85', '86',
                        '87', '88', '89',
                        '90', '91', '92',
                        '93', '94', '95',
                        '96', '97', '98',
                        '99', '100', '101',
                        '102', '103', '104',
                        '105', '106',
                    ],
                    'data_type': 'reanalysis',
                    'year': str(yr),
                    'month': str(mon).zfill(2),
                    'day': str(day).zfill(2),
                    'time': [
                        '00:00', '03:00', '06:00',
                        '09:00', '12:00', '15:00',
                        '18:00', '21:00',
                    ],
                },
                path+'tmp_'+str(yr)+str(mon).zfill(2)+str(day).zfill(2)+'.nc')
            print(path+'tmp_'+str(yr)+str(mon).zfill(2)+str(day).zfill(2)+'.nc retrieved')
            c.retrieve(
                'reanalysis-cerra-single-levels',
                {
                    'format': 'netcdf',
                    'variable': [ 'surface_pressure' ],
                    'data_type': 'reanalysis',
                    'product_type': 'analysis',
                    'year': str(yr),
                    'month': str(mon).zfill(2),
                    'day': str(day).zfill(2),
                    'time': [
                        '00:00', '03:00', '06:00',
                        '09:00', '12:00', '15:00',
                        '18:00', '21:00',
                    ],
                    'level_type': 'surface_or_atmosphere',
                },
                path+'tmp1_'+str(yr)+str(mon).zfill(2)+str(day).zfill(2)+'.nc')
            print(path+'tmp1_'+str(yr)+str(mon).zfill(2)+str(day).zfill(2)+'.nc retrieved')
            # Read the tmp and tmp1 files
            f1 = xr.open_dataset(path+'tmp1_'+str(yr)+str(mon).zfill(2)+str(day).zfill(2)+'.nc',mask_and_scale=True,decode_times=False)
            timed = f1['time'].values            
            sfp = f1['sp'].values
            f1.close()
            del f1
            f1 = xr.open_dataset(path+'tmp_'+str(yr)+str(mon).zfill(2)+str(day).zfill(2)+'.nc',mask_and_scale=True,decode_times=False)
            temp = f1['t'].values
            qs = f1['q'].values
            f1.close()
            del f1
            nt=len(timed)
            # Calculate the height of the model levels
            height=np.zeros((len(timed),len(a_coeff)-1,np.shape(sfp)[1],np.shape(sfp)[2]))
            for t in range(nt):
                height[t,:,:,:]=get_height_from_ml_cerra(a_coeff,b_coeff,sfp[t,:,:],temp[t,:,:,:],qs[t,:,:,:])
            del sfp,temp,qs
            f1 = xr.open_dataset(path+'tmp_'+str(yr)+str(mon).zfill(2)+str(day).zfill(2)+'.nc',mask_and_scale=True,decode_times=False)
            lon = f1['longitude'].values
            lat = f1['latitude'].values
            wx = f1['u'].values
            wy = f1['v'].values
            f1.close()
            del f1
            ws=np.sqrt(wx*wx+wy*wy)
            U = wx + 1j * wy
            wdir=np.rad2deg(np.pi/2 - np.angle(U)) - 180 % 360
            del wx,wy,U
            wdir=np.where(wdir<0,wdir+360,wdir)
            # select domain            
            lat=lat[y_l:y_u+1,x_l:x_u+1]
            lon=lon[y_l:y_u+1,x_l:x_u+1]
            ny=np.shape(lat)[0]
            nx=np.shape(lat)[1]
            height=height[:,:,y_l:y_u+1,x_l:x_u+1]
            ws=ws[:,::-1,y_l:y_u+1,x_l:x_u+1]
            ws=ws[:,0:-2,:,:]
            wdir=wdir[:,::-1,y_l:y_u+1,x_l:x_u+1]
            wdir=wdir[:,0:-2,:,:]
            direction_field=np.zeros((8,ny,nx))-32767.0
            speed_field=np.zeros((8,ny,nx))-32767.0
            height_field=np.zeros((8,ny,nx))-32767.0
            for y in range(ny):
                for x in range(nx):
                    for t in range(nt):
                        condition,zmax=detect_llj(ws[t,:,y,x],height[t,:,y,x],hgt_lim)
                        if condition:
                            # Fit a curve around the max
                            # Use a 2nd-degree polynom with 3 points only
                            hgt,wspd=fit2_max(height[t,zmax-1:zmax+2,y,x],ws[t,zmax-1:zmax+2,y,x])
                            speed_field[t,y,x]=wspd
                            height_field[t,y,x]=hgt
                            direction_field[t,y,x]=wdir[t,zmax,y,x]
                            del hgt,wspd
                        del condition,zmax
            os.system('rm '+path+'tmp_'+str(yr)+str(mon).zfill(2)+str(day).zfill(2)+'.nc')
            os.system('rm '+path+'tmp1_'+str(yr)+str(mon).zfill(2)+str(day).zfill(2)+'.nc')
            fnout = path+'tmp_nonpacked_'+str(yr)+str(mon).zfill(2)+str(day).zfill(2)+'.nc'
            print(fnout)
            ncfile = Dataset(fnout, 'w')
            # create the time, lat and lon dimensions.
            time = ncfile.createDimension('time', None)
            latitude = ncfile.createDimension('lat', ny)
            longitude = ncfile.createDimension('lon', nx)
            level = ncfile.createDimension('lev', len(a_coeff)-1)
            # Define the coordinate variables.
            time_out = ncfile.createVariable('time', np.float64, ('time'))
            lat_out = ncfile.createVariable('latitude', np.float32, ('lat','lon'))
            lon_out = ncfile.createVariable('longitude', np.float32, ('lat','lon'))
            lev_out = ncfile.createVariable('level', np.int32, ('lev'))
            # Assign units attributes to coordinate variable data.
            time_out.units = 'seconds since 1970-01-01 00:00:00 +00:00'
            #time_out.calendar = 'proleptic_gregorian'
            lat_out.units = 'degrees_north'
            lon_out.units = 'degrees_east'
            lev_out.units = 'no_units'
            time_out.long_name = 'time'
            lat_out.long_name = 'latitude'
            lon_out.long_name = 'longitude'
            lev_out.long_name = 'level'
            # write data to coordinate vars.
            time_out[:] = timed
            lat_out[:,:] = lat
            lon_out[:,:] = lon
            lev_out[:] = np.arange(1,len(a_coeff))
            # create main variables
            g_out = ncfile.createVariable('wind_speed', np.float32, ('time','lev','lat','lon'))
            g1_out = ncfile.createVariable('wind_direction', np.float32, ('time','lev','lat','lon'))
            g2_out = ncfile.createVariable('height', np.float32, ('time','lev','lat','lon'))
            # set the units attribute.
            g_out.units = 'm s**-1'
            g_out.long_name = 'wind speed'
            g_out.fill_value = -32767
            g_out.missing_value = -32767
            g1_out.units = 'degrees from North clockwise'
            g1_out.long_name = 'wind direction'
            g1_out.fill_value = -32767
            g1_out.missing_value = -32767
            g2_out.units = 'm'
            g2_out.long_name = 'height'
            g2_out.fill_value = -32767
            g2_out.missing_value = -32767
            # write data to variables along record (unlimited) dimension.
            for i in range(nt):
                #g_out[i,:,:,:] = np.short(np.floor((ws[i,:,:,:]-ofst)/scf))
                g_out[i,:,:,:] = ws[i,:,:,:]
                g1_out[i,:,:,:] = wdir[i,:,:,:]
                g2_out[i,:,:,:] = height[i,:,:,:]
            # close the file.
            ncfile.close()
            del ncfile,time_out,lev_out,lat_out,lon_out,g_out,g1_out,g2_out
            # Use NCO to pack the data
            fnout1 = path+'cerra_wind_'+str(yr)+str(mon).zfill(2)+str(day).zfill(2)+'.nc'
            os.system('ncpdq -O -P all_new -M flt_sht '+fnout+' '+fnout1+'')
            os.system('rm '+fnout+'')
            del fnout,fnout1
            # Write LLJ file
            # open a new netCDF file for writing.
            mn='{:0>2}'.format(mon)
            ds='{:0>2}'.format(day)
            #FileNameOut = 'tmp_nonpacked1_'+str(mon).zfill(2)+str(day).zfill(2)+'.nc'
            fnout = patho+'LLJ_'+str(yr)+'_'+str(mon).zfill(2)+'_'+str(day).zfill(2)+'.nc'
            print(fnout)
            ncfile = Dataset(fnout, 'w')
            # create the time, lat and lon dimensions.
            time = ncfile.createDimension('time', None)
            latitude = ncfile.createDimension('lat', ny)
            longitude = ncfile.createDimension('lon', nx)
            # Define the coordinate variables.
            time_out = ncfile.createVariable('time', np.float64, ('time'))
            lat_out = ncfile.createVariable('latitude', np.float32, ('lat','lon'))
            lon_out = ncfile.createVariable('longitude', np.float32, ('lat','lon'))
            # Assign units attributes to coordinate variable data.
            time_out.units = 'seconds since 1970-01-01 00:00:00 +00:00'
            #time_out.calendar = 'proleptic_gregorian'
            lat_out.units = 'degrees_north'
            lon_out.units = 'degrees_east'
            time_out.long_name = 'time'
            lat_out.long_name = 'latitude'
            lon_out.long_name = 'longitude'
            # write data to coordinate vars.
            time_out[:] = timed
            lat_out[:,:] = lat
            lon_out[:,:] = lon
            # create main variables
            g_out = ncfile.createVariable('speed_jet', np.float32, ('time','lat','lon'))
            g1_out = ncfile.createVariable('height_jet', np.float32, ('time','lat','lon'))
            g2_out = ncfile.createVariable('direction_jet', np.float32, ('time','lat','lon'))
            # set the units attribute.
            g_out.units = 'm s**-1'
            g_out.long_name = 'Jet speed'
            g_out.fill_value = -32767
            g_out.missing_value = -32767
            g1_out.units = 'm'
            g1_out.long_name = 'Jet height'
            g1_out.fill_value = -32767
            g1_out.missing_value = -32767
            g2_out.units = 'deg'
            g2_out.long_name = 'Jet direction'
            g2_out.fill_value = -32767
            g2_out.missing_value = -32767
            # write data to variables along record (unlimited) dimension.
            for i in range(8):
                g_out[i,:,:] = speed_field[i,:,:]
                g1_out[i,:,:] = height_field[i,:,:]
                g2_out[i,:,:] = direction_field[i,:,:]
            # close the file.
            ncfile.close()
            del ncfile,time_out,lat_out,lon_out,g_out,g1_out,g2_out
            del lat,lon,timed
            del speed_field,height_field,direction_field
            del fnout

