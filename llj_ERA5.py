import os,sys
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import xarray as xr
from numpy.polynomial import polynomial as P
from scipy.signal import argrelextrema
import math
from netCDF4 import Dataset



def get_height_from_ml(a_coef,b_coef,sfc_p,t_level,q_level,sfc_z):
    # Subroutine inspired from https://confluence.ecmwf.int/display/ECC/compute_geopotential_on_ml.py
    R_D = 287.06
    R_G = 9.80665
    # compute moist temperature
    t_level = t_level * (1. + 0.609133 * q_level)
    t_level = t_level * R_D
    z_h=sfc_z # 2d
    z_f=np.zeros_like(t_level) # 3d
    for lev in range(136,-1,-1): # I HAVE TO GO FROM LEVEL 137 (THE SURFACE) TO LEVEL 1 (THE TOP)
        # Compute the pressures on half-levels        
        if lev==0:
            ph_lev = 0.1
        else:
            ph_lev = a_coef[lev - 1] + (b_coef[lev - 1] * sfc_p)
        ph_levplusone = a_coef[lev] + (b_coef[lev] * sfc_p)
        if lev == 0: # lev 0 is the top of the atmosphere
            dlog_p = np.log(ph_levplusone / ph_lev)
            alpha = np.log(2)
        else:
            dlog_p = np.log(ph_levplusone / ph_lev)
            alpha = 1. - ((ph_lev / (ph_levplusone - ph_lev)) * dlog_p)
        # z_f is the geopotential of this full level
        # integrate from previous (lower) half-level z_h to the full level
        z_f[lev,:,:] = z_h + (t_level[lev,:,:] * alpha)
        # z_h is the geopotential of 'half-levels'
        # integrate z_h to next half level
        z_h = z_h + (t_level[lev,:,:] * dlog_p)
    z_f=(z_f[::-1,:,:]-sfc_z)/R_G
    return z_f


# Round up to next hundred
# Taken from https://stackoverflow.com/questions/8866046/python-round-up-integer-to-next-hundred
def roundup(x):
    return int(math.ceil(x / 100.0)) * 100

# Round down to hundred
def rounddown(x):
    return int(math.floor(x / 100.0)) * 100


# Subroutine for linear interpolation of a 1d-array
def lin_int(x1,x2,y1,y2,xtarget):
    # x should be the height and y the wind speed
    y=y1+(xtarget-x1)*((y2-y1)/(x2-x1))
    return y


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
# Method inspired by Tuononen et al. 2015 (https://rmets.onlinelibrary.wiley.com/doi/10.1002/asl.587)
# Usage: bool = detect_llj(ws_vec,height_vec,hlim)
# No wind 0 at surface, val at 1500 m interpolated, interpolation of max
def detect_llj(ws_vec,height_vec,hlim):
    # ws_vec: wind speed as a function of height (ascending order), 1d
    # height_vec_vec: height as a function of height (ascending order), 1d
    bl=False
    k1=np.nan
    hh=np.where(height_vec<hlim)[0][-1]
    # Calculate wind speed at height 1500 m
    if (height_vec[hh+1]==1500.0):
        ws_1500m=ws_vec[hh+1]
    else:
        ws_1500m=lin_int(height_vec[hh],height_vec[hh+1],ws_vec[hh],ws_vec[hh+1],1500.0)
    t1=((ws_1500m>=ws_vec[hh]) & (ws_1500m<=ws_vec[hh+1]))
    t2=((ws_1500m<=ws_vec[hh]) & (ws_1500m>=ws_vec[hh+1]))
    if not (t1 | t2):
        print(ws_vec[hh],ws_1500m,ws_vec[hh+1])
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


args=sys.argv
mon=int(args[1])

path="./"
patho="LLJ_ERA5/"


# Thresholds
hgt_lim=1500.0 # 1.5 km
diff_speed=2.0 # in m/s

year1=2000
year2=2015

# 6-hourly ERA5 data
hour=['00','06','12','18']

# Loop over files
for yr in range(year1,year2+1):
    for mn in range(mon,mon+1):
        nbd=nb_days_month(yr,mn)
        month=str(mn).zfill(2)
        timed=np.zeros((nbd*4))
        cpt=0
        for dd in range(1,nbd+1):
            day=str(dd).zfill(2)
            for t in hour:
                pathi=path+str(yr)+"/"+month+"/"
                filename="era5_"+str(yr)+month+day+t+".nc"
                u=pathi+filename
                print(filename)
                f1 = xr.open_dataset(u,mask_and_scale=True,decode_times=False)
                if cpt==0:
                    lon = f1['longitude1'].values
                    lat = f1['latitude1'].values
                    ny=len(lat)
                    nx=len(lon)
                    # Initialisations
                    direction_field=np.zeros((nbd*4,ny,nx))-32767.0
                    speed_field=np.zeros((nbd*4,ny,nx))-32767.0
                    height_field=np.zeros((nbd*4,ny,nx))-32767.0
                    timed=np.zeros((nbd*4))
                timed[cpt] = f1['time'].values
                ap=f1['ap0'].values
                b=f1['b0'].values
                sp=np.exp(f1['surface_air_pressure_ln'].isel(time=0,hybrid1=0).values) # 2d (nlat x nlon)
                ta=f1['air_temperature_ml'].isel(time=0).values # 3d (137 x nlat x nlon)
                qs=f1['specific_humidity_ml'].isel(time=0).values # 3d (137 x nlat x nlon)
                sg=f1['surface_geopotential'].isel(surface=0).values # 2d (nlat x nlon)
                height=get_height_from_ml(ap,b,sp,ta,qs,sg)
                del ap,b,sp,ta,qs,sg
                if (np.min(height[0,:,:])<0.0):
                    print('negative heights')
                wx = f1['x_wind_ml'].values
                wy = f1['y_wind_ml'].values
                f1.close()
                del u
                # Reverse the vertical axis
                wx=wx[0,::-1,:,:]
                wy=wy[0,::-1,:,:]
                ws=np.sqrt(wx*wx+wy*wy)
                for y in range(ny):
                    for x in range(nx):
                        condition,zmax=detect_llj(ws[:,y,x],height[:,y,x],hgt_lim)
                        if condition:
                            # Fit a curve around the max
                            # Use a 2nd-degree polynom with 3 points only
                            hgt,wspd=fit2_max(height[zmax-1:zmax+2,y,x],ws[zmax-1:zmax+2,y,x])
                            speed_field[cpt,y,x]=wspd
                            height_field[cpt,y,x]=hgt
                            # Get wind direction
                            Uc = wx[zmax,y,x]  + 1j * wy[zmax,y,x]  # vindvektor i forhold til nord # Create a complex object
                            wdir = np.rad2deg(np.pi/2 - np.angle(Uc)) - 180 % 360  # calculate wind direction (1 + 1j * 1 has an angle of 45 deg)
                            del Uc
                            if wdir<0.0:
                                wdir=wdir+360.0
                            direction_field[cpt,y,x]=wdir
                            del hgt,wspd,wdir
                        del condition,zmax
                del wx,wy,ws,height
                cpt=cpt+1
        # Write netcdf monthly
        FileNameOut = 'LLJ_'+str(yr)+'_'+month+'.nc'
        fnout = patho+ FileNameOut
        print(fnout)
        ncfile = Dataset(fnout, 'w')
        # create the time, lat and lon dimensions.
        time = ncfile.createDimension('time', None)
        latitude = ncfile.createDimension('lat', ny)
        longitude = ncfile.createDimension('lon', nx)
        # Define the coordinate variables.
        time_out = ncfile.createVariable('time', np.float64, ('time'))
        lat_out = ncfile.createVariable('latitude', np.float32, ('lat'))
        lon_out = ncfile.createVariable('longitude', np.float32, ('lon'))
        # Assign units attributes to coordinate variable data.
        time_out.units = 'seconds since 2014-03-05 00:00:00 +0000'
        #time_out.units = 'seconds since 1970-01-01 00:00:00 +00:00'
        #time_out.calendar = 'proleptic_gregorian'
        lat_out.units = 'degrees_north'
        lon_out.units = 'degrees_east'
        time_out.long_name = 'time'
        lat_out.long_name = 'latitude'
        lon_out.long_name = 'longitude'
        # write data to coordinate vars.
        time_out[:] = timed
        lat_out[:] = lat
        lon_out[:] = lon
        # create main variables
        g_out = ncfile.createVariable('speed_jet', np.float32, ('time','lat','lon'))
        g1_out = ncfile.createVariable('height_jet', np.float32, ('time','lat','lon'))
        g2_out = ncfile.createVariable('direction_jet', np.float32, ('time','lat','lon'))
        # set the units attribute.
        g_out.units = 'm s**-1'
        g_out.long_name = 'Jet speed'
        g_out.fill_value = -32767.0
        g_out.missing_value = -32767.0
        g1_out.units = 'm'
        g1_out.long_name = 'Jet height'
        g1_out.fill_value = -32767.0
        g1_out.missing_value = -32767.0
        g2_out.units = 'deg'
        g2_out.long_name = 'Jet direction'
        g2_out.fill_value = -32767.0
        g2_out.missing_value = -32767.0
        # write data to variables along record (unlimited) dimension.
        for i in range(nbd*4):
            g_out[i,:,:] = speed_field[i,:,:]
            g1_out[i,:,:] = height_field[i,:,:]
            g2_out[i,:,:] = direction_field[i,:,:]
        # close the file.
        ncfile.close()
        del ncfile,time_out,lat_out,lon_out,g_out,g1_out,g2_out
        del lat,lon,timed
        del speed_field,height_field,direction_field





