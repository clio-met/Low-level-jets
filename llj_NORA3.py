import os
os.environ["OMP_NUM_THREADS"] = "2"
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P
from scipy.signal import argrelextrema
import math
from netCDF4 import Dataset
import pyproj



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


# Subroutine calculating the height from the model levels
# Usage: height=height_from_model_levels(a_coeff,b_coeff,surf_pres,temp)
# Returns the heights in ascending order (index 0 is the lowest model level)
def height_from_model_levels(a_coeff,b_coeff,surf_pres,temp):
    # Surface pressure (surf_pres) is 2d (lat,lon) in Pa
    # Temperature (temp) is 3d (ml,lat,lon) in Kelvin
    # Coefficients a and b (a_coeff, b_coeff) are 1d
    # Constants
    rcst=287.058
    gcst=9.80665
    # Calculation of pressure on the model levels
    pres=a_coeff[:, np.newaxis, np.newaxis]+b_coeff[:, np.newaxis, np.newaxis]*surf_pres[np.newaxis,:,:]
    if ((np.min(temp)==np.nan) | (np.min(temp)<=0.0)):
        print("Min=",np.min(temp))
        print("Problem with reading temperature")
        sys.exit()
    # Calculation of the height
    height=np.zeros((len(a_coeff),np.shape(surf_pres)[0],np.shape(surf_pres)[1]),float)
    # Level index 0 is the top of the atmosphere and highest level index is the bottom of the atmosphere
    height[-1,:,:]=(rcst/gcst)*temp[-1,:,:]*np.log(surf_pres/pres[-1,:,:])
    # Loop over all other levels
    for k in range(len(a_coeff)-2,-1,-1):
        height[k,:,:] = (rcst/gcst)*temp[k,:,:]*np.log(pres[k+1,:,:]/pres[k,:,:])+height[k+1,:,:]
    # Reverse the array to get increasing values along the vertical axis
    height=height[::-1,:,:]
    del pres
    return height


# Calculation of the height from the model levels
# Code given by Ole Nikolai Vignes ()
def get_height_from_ml(path_file):
    ds = xr.open_dataset(path_file).isel(time=0,height0=0)
    # The input is a dataset
    ap = ds.ap
    b = ds.b
    ahalf = xr.zeros_like(ap)
    bhalf = xr.zeros_like(b)
    max_k = len(ahalf.hybrid) -1
    ahalf[max_k] = 2.0*ap[max_k] - 0.0
    bhalf[max_k] = 2.0*b[max_k]  - 1.0
    for k in range(max_k-1, -1, -1):
        ahalf[k] = 2.0*ap[k] - ahalf[k+1]
        bhalf[k] = 2.0*b[k]  - bhalf[k+1]
    surface_pressure = ds.surface_air_pressure
    # Pressure at half levels
    p_at_khalf = ahalf + bhalf*surface_pressure
    del ahalf,bhalf,ap,b
    # Don't forget to include humidity!
    epsilo = 0.622
    epim1 = 1.0/epsilo - 1.0
    spec_hum = ds.specific_humidity_ml
    temperature_at_k = ds.air_temperature_ml
    R = 287.058
    g = 9.81
    # New more accurate height, 3D
    h_at_kfull = xr.full_like(spec_hum, fill_value=0)
    # Compute the height of the next lowest half model level
    virt_temp_k = (1.0 + epim1*spec_hum.isel(hybrid=max_k))*temperature_at_k.isel(hybrid=max_k)
    dlnp_k = np.log(surface_pressure/p_at_khalf.isel(hybrid=max_k))
    dp_k = surface_pressure - p_at_khalf.isel(hybrid=max_k)
    h_at_khalf = R*virt_temp_k*dlnp_k/g
    alpha_k = 1.0 - p_at_khalf.isel(hybrid=max_k)*dlnp_k/dp_k
    h_at_kfull.values[max_k,:,:] = alpha_k*R*virt_temp_k/g
    del virt_temp_k,dlnp_k,dp_k,alpha_k
    del surface_pressure
    # Loop over the rest of the model half levels (except top)
    for k in range(max_k-1, 0, -1):
        virt_temp_k = (1.0 + epim1*spec_hum.isel(hybrid=k))*temperature_at_k.isel(hybrid=k)
        dlnp_k = np.log(p_at_khalf.isel(hybrid=k+1) / p_at_khalf.isel(hybrid=k))
        dp_k = p_at_khalf.isel(hybrid=k+1) - p_at_khalf.isel(hybrid=k)
        alpha_k = 1.0 - p_at_khalf.isel(hybrid=k)*dlnp_k/dp_k
        h_at_kfull.values[k,:,:] = h_at_khalf + alpha_k*R*virt_temp_k/g
        h_at_khalf += R*virt_temp_k*dlnp_k/g
        del virt_temp_k,dlnp_k,dp_k,alpha_k
    # Top level needs special treatment
    virt_temp_k = (1.0 + epim1*spec_hum.isel(hybrid=0))*temperature_at_k.isel(hybrid=0)
    h_at_kfull.values[0,:,:] = h_at_khalf + R*virt_temp_k*np.log(2.0)/g
    del virt_temp_k,h_at_khalf
    # Reverse the array to get increasing heights along the vertical axis
    return h_at_kfull.values[::-1,:,:]



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



# Subroutine detecting low-level jets from a wind profile (for one grid point)
# returns a boolean and the model height of max
# Method inspired by Tuononen et al. 2015
# Usage: bool = detect_llj(ws_vec,height_vec,hlim)
# No wind 0 at surface, val at 1500 m interpolated, interpolation of max
def detect_llj(ws_vec,height_vec,hlim):
    # ws_vec: wind speed as a function of height (ascending order), 1d
    # height_vec_vec: height as a function of height (ascending order), 1d
    bl=False
    k1=np.nan
    hh=np.where(height_vec<hlim)[0][-1]
    # Calculate wind speed at height 1500 m
    ws_1500m=lin_int(height_vec[hh],height_vec[hh+1],ws_vec[hh],ws_vec[hh+1],1500.0)
    t1=((ws_1500m>=ws_vec[hh]) & (ws_1500m<=ws_vec[hh+1]))
    t2=((ws_1500m<=ws_vec[hh]) & (ws_1500m>=ws_vec[hh+1]))
    if ~(t1 | t2):
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


# Subroutine from Birgitte Rugaard Furevik, updated to remove warnings by me
def get_rotate_angle(reader_x,reader_y):
    """Get angle to rotate vectors from one srs to another."""
    proj_to = '+proj=latlong +R=6370997.0 +ellps=WGS84'
    proj_from = '+proj=lcc +lat_0=66.3 +lon_0=-42 +lat_1=66.3 +lat_2=66.3 +no_defs +R=6.371e+06'
    xm, ym = np.meshgrid(reader_x, reader_y)
    if type(proj_from) is str:
        proj_from = pyproj.Proj(proj_from)
    if type(proj_to) is str:
        proj_to = pyproj.Proj(proj_to)
    delta_y = 10  # 10 m along y-axis
    transformer = pyproj.Transformer.from_proj(proj_from, proj_to)  # Projection transformer
    x2, y2 = transformer.transform(xm, ym)    # Transform the regular coordinates in old proj to coordinates in new proj
    x2_delta, y2_delta = transformer.transform(xm, ym + delta_y)  # move a little way along the y-axis
    geod = pyproj.Geod(ellps='WGS84')
    rot_angle_vectors_rad = np.radians(geod.inv(x2, y2, x2_delta, y2_delta)[0])  # azimuth: clockwise from north baseline
    rot_angle_rad = - rot_angle_vectors_rad
    return rot_angle_rad




# Paths
path="https://thredds.met.no/thredds/dodsC/nora3/"
patho="LLJ_NORA3/"


# Thresholds
hgt_lim=1500.0 # 1.5 km
diff_speed=2.0 # in m/s


# To read the files day by day
fcst0=["18","00","06","12"]
ts=["006","009"]



# Read input (year and month)
args=sys.argv
yr=int(args[1])
mon=int(args[2])
print(yr,mon)

# I detect the low-level jets and write 2-d maps of speed, height, and direction
# One output file per day

# Number of days for the month of interest
nd=nb_days_month(yr,mon)

# cpt counts the number of timesteps in the month of interest
cpt=0
# loop over the days within the considered month
for day in range(1,nd+1):
    print(mon,day)
    fcst=fcst0
    cpt1=0
    # loop over the forecast time
    for fc in range(len(fcst)):
        if ((fc==0) & (fcst[fc]=='18')):
            if ((mon>1) & (day==1)):
                year='{:0>4}'.format(yr)
                mn='{:0>2}'.format(mon-1)
                dd='{:0>2}'.format(nb_days_month(yr,mon-1))
            elif ((mon==1) & (day==1)):
                year='{:0>4}'.format(yr-1)
                mn='{:0>2}'.format(12)
                dd='{:0>2}'.format(31)
            else:
                year='{:0>4}'.format(yr)
                mn='{:0>2}'.format(mon)
                dd='{:0>2}'.format(day-1)
        else:
            year='{:0>4}'.format(yr)
            mn='{:0>2}'.format(mon)
            dd='{:0>2}'.format(day)
        for t in range(len(ts)):
            filename=year+"/"+mn+"/"+dd+"/"+fcst[fc]+"/fc"+year+mn+dd+fcst[fc]+"_"+ts[t]+".nc"
            u = path+filename
            print(u)
            f1 = xr.open_dataset(u,mask_and_scale=True,decode_times=False)
            if cpt1==0:
                lon = f1['longitude'].values
                lat = f1['latitude'].values
                xx = f1['x'].values
                yy = f1['y'].values
                ny=np.shape(lon)[0]
                nx=np.shape(lon)[1]
                # Arrays containing the places of occurrence, the speed of the LLJ when it occurs and the height where it occurs
                direction_field=np.zeros((8,ny,nx))-32767.0
                speed_field=np.zeros((8,ny,nx))-32767.0
                height_field=np.zeros((8,ny,nx))-32767.0
                # Array with the time
                timed=np.zeros((8))
                # Get the grid rotation angle
                gridangle = get_rotate_angle(xx,yy)  # grid angle from north
                del xx,yy
            timed[cpt1] = f1['time'].values
            height=get_height_from_ml(u)
            wx = f1['x_wind_ml'].values
            wy = f1['y_wind_ml'].values
            f1.close()
            del u
            ws=np.sqrt(wx*wx+wy*wy)
            wx=wx[0,::-1,:,:]
            wy=wy[0,::-1,:,:]
            # Reverse the vertical axis
            ws=ws[0,::-1,:,:]
            #print(np.min(ws),np.max(ws))
            for y in range(ny):
                for x in range(nx):
                    condition,zmax=detect_llj(ws[:,y,x],height[:,y,x],hgt_lim)
                    if condition:
                        # Fit a curve around the max
                        # Use a 2nd-degree polynom with 3 points only
                        hgt,wspd=fit2_max(height[zmax-1:zmax+2,y,x],ws[zmax-1:zmax+2,y,x])
                        speed_field[cpt1,y,x]=wspd
                        height_field[cpt1,y,x]=hgt
                        # Get wind direction
                        # Rotate the wind compenents
                        u_rot = (wx[zmax,y,x]*np.cos(gridangle[y,x]) - wy[zmax,y,x]*np.sin(gridangle[y,x]))
                        v_rot = (wx[zmax,y,x]*np.sin(gridangle[y,x]) + wy[zmax,y,x]*np.cos(gridangle[y,x]))
                        Uc = u_rot + 1j * v_rot  # vindvektor i forhold til nord # Create a complex object
                        ws1 = abs(Uc)  # calculate wind speed
                        wdir = np.rad2deg(np.pi/2 - np.angle(Uc)) - 180 % 360  # calculate wind direction (1 + 1j * 1 has an angle of 45 deg)
                        del u_rot,v_rot
                        if wdir<0.0:
                            wdir=wdir+360.0
                        direction_field[cpt1,y,x]=wdir
                        del hgt,wspd,wdir,ws1
                    del condition,zmax
            del wx,wy,ws
            cpt=cpt+1
            cpt1=cpt1+1
    # open a new netCDF file for writing.
    mn='{:0>2}'.format(mon)
    ds='{:0>2}'.format(day)
    FileNameOut = 'LLJ_'+str(yr)+'_'+mn+'_'+ds+'.nc'
    fnout = patho+ FileNameOut
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
    for i in range(8):
        g_out[i,:,:] = speed_field[i,:,:]
        g1_out[i,:,:] = height_field[i,:,:]
        g2_out[i,:,:] = direction_field[i,:,:]
    # close the file.
    ncfile.close()
    del ncfile,time_out,lat_out,lon_out,g_out,g1_out,g2_out
    del lat,lon,timed
    del speed_field,height_field,direction_field





