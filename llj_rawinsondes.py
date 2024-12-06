import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
import sys

# Get number of days within a month of a particular year
def get_nb_days(month,year):
    if month in [1,3,5,7,8,10,12]:
        nb=31
    elif month in [4,6,9,11]:
        nb=30
    elif month == 2:
        if year%4 == 0:
            nb=29
        else:
            nb=28
    return nb
    
    
# Subroutine for linear interpolation of a 1d-array
def lin_int(x1,x2,y1,y2,xtarget):
    # x should be the height and y the wind speed
    y=y1+(xtarget-x1)*((y2-y1)/(x2-x1))
    return y


# Smooth profile along the height dimension
# the imput array is 2d: time x height
def smoothing_hgt_Xpoints_2d(arr_in,x):
    # Note: x should an odd number
    x_l=int(x/2)
    ws_smooth=np.zeros_like(arr_in)
    for t in range(np.shape(arr_in)[0]):
        for i in range(x_l,np.shape(arr_in)[1]-x_l):
            ws_smooth[t,i]=(np.sum(arr_in[t,i-x_l:i+x_l+1])/x)
    for j in range(0,x_l):
        #print('start',np.shape(arr_in[:,0:j+x_l+1]))
        ws_smooth[:,j]=np.mean(arr_in[:,0:j+x_l+1],axis=1)
    for j in range(np.shape(arr_in)[1]-x_l,np.shape(arr_in)[1]):
        #print('end',np.shape(arr_in[:,j-x_l::]))
        ws_smooth[:,j]=np.mean(arr_in[:,j-x_l::],axis=1)
    return ws_smooth
    
    
# warning_flag: "Warning: if flagged, there may be issues in the data. A human check is needed. Type of warning: Read bits from left. 0=p>850hPa while RH<6% , 1=p>925hPa while temp>30C, 2=surface wind speed>25m/s, 3=absolute value of surface temp>30C, 4=surface RH<10%, 5=surface p<900hPa"
#Pressure on the ground (t=0) < 900 hPa
#Relative humidity on the ground (t=0) < 10%
#Relative humidity < 6% and pressure > 850 hPa at the same time
#Temperature on the ground-ish < -30 C or > 30 C
#Temperature > 30 C and pressure > 925 hPa  at the same time
#Wind speed at the ground > 25 m/s
def check_data_validity(p,rh,T,ws,wf):
    chk=True
    # Returns True if valid, returns False if not valid
    if p[0]<=900.0:
        chk=False
        print('p[0]<900')
    if rh[0]<10.0:
        chk=False
        print('rh[0]<10')
    a=np.where((rh<6.0) & (p>850.0))[0]
    if len(a)>0:
        chk=False
        print('(rh<6.0) & (p>850.0)')
    del a
    if ((T[0]<-30.0) | (T[0]>30)):
        chk=False
        print('(T[0]<-30.0) | (T[0]>30)')
    a=np.where((T>30) & (p>925))[0]
    if len(a)>0:
        chk=False
        print('(T>30) & (p>925)')
    del a
    if ws[0]>25.0:
        chk=False
        print('ws[0]>25')
    if not(np.isnan(wf)):
        chk=False
        print('wf=','{0:05b}'.format(int(wf)))
    return chk
    
    
# Subroutine to detect the low-level jets
def detect_llj_rs_profile(wind,alt,hgt_lim):
    # New LLJ detection subroutine for radiosondes
    # alt and wind are 1-d arrays and hgt_lim is a scalar (1500 m)
    bl=False
    k1=np.nan
    flag_pb=False
    hh=np.where(alt[:]<=hgt_lim)[0]
    if (len(hh)!=0):
        # Check if there are several times during the profile that we get below 1500 m
        # and pick the first bunch from 0
        diff_hh=hh[1::]-hh[0:-1]
        pos=np.where(diff_hh>1)[0]
        if len(pos)>0:
            hh=pos[0]
        else:
            hh=hh[-1]
        if not((np.any(np.isnan(alt[0:hh+2]))) | np.any(np.isnan(wind[0:hh+2]))):
            # Calculate wind speed at height 1500 m
            ws_1500m=lin_int(alt[hh],alt[hh+1],wind[hh],wind[hh+1],1500.0)
            t1=((ws_1500m>=wind[hh]) & (ws_1500m<=wind[hh+1]))
            t2=((ws_1500m<=wind[hh]) & (ws_1500m>=wind[hh+1]))
            if not(t1 | t2):
                print(wind[hh],ws_1500m,wind[hh+1])
                print('Problem with the interpolation')
                sys.exit()
            # I concatenate the wind speed at 1500 m to the profile
            ws_vec1=np.hstack((wind[0:hh+1],ws_1500m))
            # Looks for peaks and troughs
            peaks, _ = find_peaks(ws_vec1, height=0)
            troughs,_ = find_peaks(np.negative(ws_vec1))
            troughs=np.hstack(([0],troughs,len(ws_vec1)-1))
            if len(peaks)>0:
                chk=0
                # Loop over the maxima
                for k in range(len(peaks)):
                    if chk==0: # If a jet has already been found, I do not look at the other peaks
                        a=0
                        # Loop over the minima to find surrounding minima to the maximum
                        for l in range(len(troughs)-1):
                            if ((troughs[l]<peaks[k]) & (troughs[l+1]>peaks[k])):
                                if ((ws_vec1[troughs[l]]<=ws_vec1[peaks[k]]) & (ws_vec1[troughs[l+1]]<=ws_vec1[peaks[k]])):
                                    mn_low=troughs[l]
                                    mn_up=troughs[l+1]
                                    w_max=ws_vec1[peaks[k]]
                                    w_low=ws_vec1[mn_low]
                                    w_up=ws_vec1[mn_up]
                                    # Condition 1: the max ws must be larger by 2 m/s compared to the surrounding minima
                                    # Condition 2: the max must be larger by 25% compared to the surrounding minima
                                    cond1=((w_max>=(w_low+2.0)) & (w_max>=(w_up+2.0)))
                                    cond2=((w_max>=(w_low*1.25)) & (w_max>=(w_up*1.25)))
                                    # I want to keep the lower jet (chk=0 because the loop over k starts from the bottom of the column)
                                    if ((cond1 & cond2) & (chk==0)):
                                        bl=True
                                        k1=peaks[k]
                                        chk=chk+1
                                    else:
                                        bl=False
        else:
            flag_pb=True
    else:
        flag_pb=True
    # Returns True or False and an index of the height level where the LLJ is
    return bl,k1,flag_pb,hh+1
    

# Calculation of distance along great circle (in km)
# Taken from https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
from math import radians, sin, cos, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    rad_earth=6371.0 # km
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * rad_earth * asin(sqrt(a))



# Read file with the available months for each station
# Note that the months may have missing data
filename='./List_Time_Radiosondes_csv.csv'
df=pd.read_csv(filename,sep=',')
# Get stations names
stations=df.columns.tolist()[2::]
# Add more columns to the dataframe
df['Day'] = [get_nb_days(m,y) for m,y in zip(df['Month'].tolist(),df['Year'].tolist())]
df['YearName1'] = df['Year']
df['MonthName1'] = df['Month']
df['DayName1'] = np.ones(len(df),'int32')*31
# Modify the new columns
df['YearName1'] = np.where(df['Month'] == 1, df['Year']-1, df['Year'])
df['MonthName1'] = np.where(df['Month'] == 1, 12, df['Month']-1)
df['DayName1'] = [get_nb_days(m,y) for m,y in zip(df['MonthName1'].tolist(),df['YearName1'].tolist())]
mmdd=[str(x).zfill(2)+str(y) for x,y in zip(df['Month'].tolist(),df['Day'].tolist())]
yyyymmdd2=[str(x)+y for x,y in zip(df['Year'].tolist(),mmdd)]
del mmdd
mmdd=[str(x).zfill(2)+str(y) for x,y in zip(df['MonthName1'].tolist(),df['DayName1'].tolist())]
yyyymmdd1=[str(x)+y for x,y in zip(df['YearName1'].tolist(),mmdd)]
#print(yyyymmdd1)
del mmdd
sufx=[x+'-'+y for x,y in zip(yyyymmdd1,yyyymmdd2)]
# Add this column to the dataframe
df['FileName']=sufx
del yyyymmdd1,yyyymmdd2,sufx
year1=2000
year2=2015
# fn will contain all files names within the chosen time (years) interval (a list of lists)
fn=[]
for s in range(len(stations)):
    it=np.where((df['Year']>=year1) & (df['Year']<=year2) & (~np.isnan(df[stations[s]])))[0]
    sel_files=df['FileName'].iloc[it].tolist()
    years_files=df['Year'].iloc[it].tolist()
    months_files=df['Month'].iloc[it].tolist()
    # List of the files to use
    fn.append([str(years_files[x])+'/'+str(months_files[x]).zfill(2)+'/'+stations[s]+'_'+sel_files[x]+'.nc' for x in range(len(sel_files))])

perc=[len(l)*100/(12*16) for l in fn] # 12 files per year and I look at 16 years

# I have to remove the first file for Bodø (January 2000)
del fn[10][0]
print(perc)
print(stations)

# Select the stations where there are data for more than 60% of the time during the period chosen
s_sel=np.where(np.array(perc)>=60.0)[0]

# Location of the rawinsondes data
url_rad='https://thredds.met.no/thredds/dodsC/remotesensingradiosonde/'

stations_names=['Sola','Ørland','Jan Mayen','Ekofisk','Bjørnøya','Bodø','Polarfront']
# Get the latitudes and longitudes of the stations from where the rawinsonde was launched 
lons=[]
lats=[]
alts=[]
# loop over the stations with enough data (do not look at Polarfront)
for s in s_sel[0:-1]:
    lons1=[]
    lats1=[]
    alts1=[]
    for file in fn[s]:
        f1=xr.open_dataset(url_rad+file,mask_and_scale=True,decode_times=True)
        lg=f1.attrs['station_longitude_degrees_east']
        if ((np.abs(lg)>=0.0) & (np.abs(lg)<=1.0)):
            lg=lg*100.0
        lt=f1.attrs['station_latitude_degrees_north']
        if ((np.abs(lt)>=0.0) & (np.abs(lt)<=1.0)):
            lt=lt*100.0
        lons1.append(lg)
        lats1.append(lt)
        alts1.append(f1.attrs['station_altitude_meter'])
        f1.close()
        del f1,lg,lt
    lons.append(lons1)
    lats.append(lats1)
    alts.append(alts1)
    del lons1,lats1,alts1


# Polarfront should only be at one location
lons.append([2.0] * len(fn[s_sel[-1]]))
lats.append([66.0] * len(fn[s_sel[-1]]))
alts.append([6.0] * len(fn[s_sel[-1]]))

# Show the various locations of the stations (changed in time)
for i in range(len(stations_names)):
    print(list(set(lons[i])),list(set(lats[i])),list(set(alts[i])))


# Detection of the low-level jets
# Extraction of jets height, speed, direction, date, and wind speed profiles
llj_hgt_all=[]
llj_spd_all=[]
llj_dir_all=[]
llj_date_all=[]
cpt_nan_all=[]
cpt_time_all=[]
count_time_all=[]
count_flg_pb_all=[]
cpt_other_all=[]
dates_detect_all=[]
llj_profiles_all=[]
llj_altitudes_all=[]
month_all=[]
year_all=[]
dist_max_all=[]
istation=0
# loop over the stations with enough data
for s in s_sel:
    llj_hgt=[]
    llj_spd=[]
    llj_dir=[]
    llj_date=[]
    dates_detect=[]
    llj_profiles=[]
    llj_altitudes=[]
    cpt_nan=0
    cpt_len_time=0
    count_time=0
    count_flg_pb=0
    cpt_other=0
    a_cpt=0
    month_list=[]
    year_list=[]
    distm=[]
    # loop over the available months for this station
    for file in fn[s]:
        print(a_cpt,file)
        f1=xr.open_dataset(url_rad+file,mask_and_scale=True,decode_times=True)
        ws=f1['wind_speed'].values
        lon=f1['longitude'].values
        lat=f1['latitude'].values
        time=f1['time']
        tfl=f1['time_from_launch'].values
        altitude=f1['altitude'].values
        wdir=f1['wind_from_direction'].values
        f1.close()
        f1=xr.open_dataset(url_rad+file,mask_and_scale=True,decode_times=False)
        time1=f1['time']
        tfl1=f1['time_from_launch'].values
        f1.close()
        cpt_len_time=cpt_len_time+len(time)
        print(np.shape(ws))
        t2=tfl.astype('timedelta64[s]').astype('int')
        ind_even=np.where(t2%2==0)[0]
        altitude=altitude[:,ind_even]
        ws=ws[:,ind_even]
        lon=lon[:,ind_even]
        lat=lat[:,ind_even]
        wdir=wdir[:,ind_even]
        del t2,ind_even
        ws_smooth=smoothing_hgt_Xpoints_2d(ws,11)
        month_file=int(file[5:7])
        year_file=int(file[0:4])
        # Loop over time
        for t in range(len(time)):
            flag=0
            ws1=ws_smooth[t,:]
            # I need data within the first 1500 m (which is about the first 160 points)
            cond_data=~((np.any(np.isnan(altitude[t,0:160]))) | np.any(np.isnan(ws[t,0:160])) | np.any(np.isnan(wdir[t,0:160])))
            # Looking at the ascent, if it is too slow, there is a problem with the launch
            if cond_data:
                diff_alt=altitude[t,1:160]-altitude[t,0:159]
                if ((np.nanmean(np.array(diff_alt))<=8.0) | (np.nanmean(np.array(diff_alt))>=20.0)):
                    print(np.nanmean(np.array(diff_alt)))
                    cond_ar=False
                else:
                    cond_ar=True
                del diff_alt
            else:
                cond_ar=False
            # I need data within the first 1500 m (slightly redundant with cond_data)
            if np.nanmax(altitude[t,:]>1500.0):
                cond_max=True
            else:
                cond_max=False
            # For the Polarfront, I discard the launches too far from its mooring location
            if stations_names[istation]=='Polarfront':
                dist=haversine(lon[t,0],lat[t,0],lons[istation][a_cpt],lats[istation][a_cpt])
                if dist<30.0:
                    cond_location=True
                else:
                    cond_location=False
                del dist
            else:
                cond_location=True
            # Check if the original wind speed at the lowest point is above 25 m/s
            #and the max wind speed in the first 160 points is below 60 m/s
            if ((ws[t,0]<=25.0) & (np.max(ws[t,0:160])<=60)):
                cond_ws=True
            else:
                cond_ws=False
                if (np.max(ws[t,0:160])>50):
                    print('Check this profile:', file, 'timestep=',t)
            # Check first altitude point if it is too far from the launch altitude
            if ((altitude[t,0]>(alts[istation][a_cpt]+15)) | (altitude[t,0]<(alts[istation][a_cpt]-15))):
                cond_alt=False
            else:
                cond_alt=True
            if (cond_data & cond_max & cond_location & cond_ws & cond_ar & cond_alt):
                flg_pb=True
                # To compare with ERA5, NORA3, and CERRA, I have to get the height above ground, not the altitude
                # So I remove the altitude of the station, the first point is more precise but has to be close to the theoretical altitude of the station
                if (np.abs(altitude[t,0]-alts[istation][a_cpt])<1.0):
                    alt1=altitude[t,:]-altitude[t,0]
                else:
                    alt1=altitude[t,:]-alts[istation][a_cpt]
                if np.nanmin(alt1)!=0:
                    print('Slight problem altitude',t,np.nanmin(alt1),altitude[t,0],alts[istation][a_cpt],np.abs(altitude[t,0]-alts[istation][a_cpt]))
                    #sys.exit()
                if np.nanmin(alt1)<0:
                    print('Negative altitude',t,np.nanmin(alt1),altitude[t,0],alts[istation][a_cpt],np.abs(altitude[t,0]-alts[istation][a_cpt]))
                    sys.exit()
                condition,i_hgt,flg_pb,h_index=detect_llj_rs_profile(ws1,alt1,1500.0)
                if flg_pb:
                    count_flg_pb=count_flg_pb+1
                else:
                    dates_detect.append(np.datetime_as_string(time[t].values, unit ='s'))
                    month_list.append(month_file)
                    year_list.append(year_file)
                    for pt in range(1,len(alt1)):
                        if alt1[pt]<=1500.0:
                            dd=haversine(lon[t,0],lat[t,0],lon[t,pt],lat[t,pt])
                            distm.append(dd)
                    count_time=count_time+1
            else:
                cpt_other=cpt_other+1
                flag=1
                condition=False
            if condition:
                llj_hgt.append(alt1[i_hgt])
                llj_spd.append(ws1[i_hgt])
                llj_dir.append(wdir[t,i_hgt])
                llj_date.append(np.datetime_as_string(time[t].values, unit ='s'))
                # Store the original wind profiles from the lowest level to at least 1500 m
                llj_profiles.append(ws[t,0:h_index+2])
                llj_altitudes.append(alt1[0:h_index+2])
                del h_index,alt1
            del cond_data,cond_max,cond_ws,cond_location,cond_ar,cond_alt
            del ws1
        a_cpt=a_cpt+1
    llj_hgt_all.append(llj_hgt)
    llj_spd_all.append(llj_spd)
    llj_dir_all.append(llj_dir)
    llj_date_all.append(llj_date)
    llj_profiles_all.append(llj_profiles)
    llj_altitudes_all.append(llj_altitudes)
    cpt_time_all.append(cpt_len_time)
    count_time_all.append(count_time)
    count_flg_pb_all.append(count_flg_pb)
    cpt_other_all.append(cpt_other)
    dates_detect_all.append(dates_detect)
    month_all.append(month_list)
    year_all.append(year_list)
    if np.isnan(np.max(np.array(distm))):
        print(distm)
    dist_max_all.append(np.max(np.array(distm)))
    del distm
    istation=istation+1


# Print the number of profiles used to detect jets, the number discarded, and the total number for each station
for s in range(len(s_sel)):
    print(cpt_other_all[s], count_flg_pb_all[s], count_time_all[s], cpt_time_all[s], cpt_other_all[s]+count_flg_pb_all[s]+count_time_all[s])


# Check that the lists length are consisten
# Print the low-level jet frequency
for s in range(len(s_sel)):
    print(s)
    print(len(month_all[s]))
    print(len(dates_detect_all[s]))
    print(len(llj_date_all[s]))
    print(len(llj_altitudes_all[s]))
    print('LLJ frequency=',len(llj_date_all[s])*100/len(month_all[s]))


# Write the output into csv files
for s in range(len(s_sel)):
    print(stations[s_sel[s]])
    dict={'date':llj_date_all[s], 'llj_speed':llj_spd_all[s], 'llj_height':llj_hgt_all[s], 'llj_direction':llj_dir_all[s]}
    df_out = pd.DataFrame(dict)
    #print(df_out)
    df_out.to_csv(stations[s_sel[s]]+'_'+str(year1)+'_'+str(year2)+'.csv')


# Monthly frequency
# Data availability
dates_period=[]
for y in range(2000,2016):
    for m in range(1,13):
        dates_period.append(str(y)+'-'+str(m).zfill(2))

#print(dates_period)
        
nb_oc=np.zeros((len(s_sel),len(dates_period)))
nb_llj_oc=np.zeros((len(s_sel),len(dates_period)))
count_wrongdate=np.zeros((len(s_sel),len(dates_period)))
# Calculate the number of measurements per month over the whole period
for s in range(len(s_sel)):
    for d in range(len(month_all[s])):
        y=year_all[s][d]
        m=month_all[s][d]
        dt=str(int(y))+'-'+str(int(m)).zfill(2)
        if dt in dates_period:
            i=dates_period.index(dt)
            nb_oc[s,i]=nb_oc[s,i]+1
            if len(np.where(np.array(llj_date_all[s])==np.array(dates_detect_all[s][d]))[0])>0:
                if len(np.where(np.array(llj_date_all[s])==np.array(dates_detect_all[s][d]))[0])>1:
                    print('Problem with date',len(np.where(np.array(llj_date[s])==np.array(dates_detect_all[s][d]))[0]))
                nb_llj_oc[s,i]=nb_llj_oc[s,i]+1
        else:
            count_wrongdate[s,i]=count_wrongdate[s,i]+1

            
# Check the counts
for s in range(len(s_sel)):
    print(int(np.sum(nb_oc[s,:])),int(np.sum(nb_llj_oc[s,:])),int(np.sum(count_wrongdate[s,:])))
    
# Upper limit for y-axis depends on the station
ul=[65,85,80,40,60,80,30]
let=['(a)','(b)','(c)','(d)','(e)','(f)','(g)']

for s in range(len(s_sel)):
    # Plot data availability per month
    plt.bar(np.arange(0,len(dates_period),1),nb_oc[s,:],width=1)
    x_pos=np.arange(0,len(dates_period)-1,12)
    bars=dates_period[0::12]
    plt.xticks(x_pos, bars, color='k',rotation=90,fontsize=16)
    plt.yticks(color='k',fontsize=16)
    plt.ylabel('# launches',fontsize=18)
    plt.title(let[s]+' '+stations_names[s],fontsize=20)
    plt.xlim(-1,len(dates_period))
    plt.tight_layout()
    #plt.savefig('DataAvailabilityMonthly_'+stations_names[s]+'_2000-2015.pdf',format='pdf',dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    # Plot the number of low-level jets detected per month
    plt.bar(np.arange(0,len(dates_period),1),nb_llj_oc[s,:],width=1)
    x_pos=np.arange(0,len(dates_period)-1,12)
    bars=dates_period[0::12]
    plt.xticks(x_pos, bars, color='k',rotation=90,fontsize=16)
    plt.yticks(color='k',fontsize=16)
    plt.ylabel('LLJ occurrence',fontsize=18)
    plt.title(stations_names[s],fontsize=20)
    plt.xlim(-1,len(dates_period))
    plt.tight_layout()
    #plt.savefig('MonthlyLLJoccurrence_'+stations_names[s]+'_2000-2015.pdf',format='pdf',dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.show()


# Plot the percentages of low-level jets per month
for s in range(len(s_sel)):
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.bar(np.arange(0,len(dates_period),1),nb_llj_oc[s,:]*100/nb_oc[s,:],width=1)
    x_pos=np.arange(0,len(dates_period)-1,12)
    bars=dates_period[0::12]
    ax.set_ylabel('LLJ freq. (%)',fontsize=18)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bars,fontsize=16,rotation=90)
    ax.tick_params(axis='y', labelsize=16)
    plt.title(stations_names[s],fontsize=20)
    plt.xlim(-1,len(dates_period))
    plt.ylim(0,ul[s])
    plt.tight_layout()
    #plt.savefig('MonthlyPercentageLLJoccurrence_'+stations_names[s]+'_2000-2015.pdf',format='pdf',dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    

for s in range(len(s_sel)):
    print('number of LLJs detected:',int(np.sum(nb_llj_oc[s,:])),'out of',int(np.sum(nb_oc[s,:])),'profiles investigated')



# Plot mean wind speed profiles during low-level jets timesteps
from scipy.interpolate import interp1d

# Check that the jets detected are all correct
for s in range(len(s_sel)):
    print(len(np.where(np.array(llj_hgt_all[s])==0)[0]),len(llj_hgt_all[s]),'if there are values in 1st, problem !')

# For all jets
# Interpolate all profiles on the same heights
heights=np.arange(55,1505,5)
interp_prof_all=np.zeros((len(s_sel),6000,len(heights)))
fig, ax = plt.subplots(figsize=(5.5, 5.5))
for s in range(len(s_sel)):
    interp_prof=[]
    i1=0
    hgt=np.array(llj_hgt_all[s])
    for i in range(len(hgt)):
        f = interp1d(np.array(llj_altitudes_all[s][i]), np.array(llj_profiles_all[s][i]))
        if np.nanmin(np.array(llj_altitudes_all[s][i]))<=heights[0]:
            ws_interp = f(heights)
            interp_prof_all[s,i1,:]=ws_interp
            i1=i1+1
        else:
            print('s=',s,'Min altitude=',np.nanmin(np.array(llj_altitudes_all[s][i])))
    del hgt
    # Get the percentage of jets as a check
    print(i1,len(np.where(np.array(llj_hgt_all[s])>=0.0)[0]),len(llj_altitudes_all[s]),len(llj_hgt_all[s]),(i1/len(llj_hgt_all[s])*100))
    mean_prof=np.sum(interp_prof_all[s,0:i1,:],axis=0)/float(i1)
    ax.plot(mean_prof,heights,label=stations_names[s],linewidth=3)
    del i1
ax.set_xlabel(r"Wind speed (m s$^{-1}$)",fontsize=16)
ax.set_ylabel("Height (m)",fontsize=16)
ax.set_xlim(0,15)
ax.set_ylim(0,1500)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='both', which='major', width=1, length=10)
ax.legend(loc='upper left',fontsize=16,frameon=False)
#plt.savefig("Mean_profile_LLJ_2000_2015_all_stations_HeightAboveGround.pdf",format="pdf",dpi=150, bbox_inches='tight', pad_inches=0.1)
plt.show()
del fig,ax

# For jets below 500 m
# Interpolate all profiles on the same heights
heights=np.arange(55,1505,5)
interp_prof_all=np.zeros((len(s_sel),6000,len(heights)))
fig, ax = plt.subplots(figsize=(5.5, 5.5))
for s in range(len(s_sel)):
    interp_prof=[]
    i1=0
    hgt=np.array(llj_hgt_all[s])
    for i in range(len(hgt)):
        if hgt[i]<500.0:
            f = interp1d(np.array(llj_altitudes_all[s][i]), np.array(llj_profiles_all[s][i]))
            if np.nanmin(np.array(llj_altitudes_all[s][i]))<=heights[0]:
                ws_interp = f(heights)
                #print(np.max(ws_interp[0:90]),np.max(ws_interp),hgt[i])
                interp_prof_all[s,i1,:]=ws_interp
                i1=i1+1
                del ws_interp
            else:
                print('s=',s,'Min altitude=',np.nanmin(np.array(llj_altitudes_all[s][i])))
    del hgt
    # Get the percentage of jets within the first 500 meters
    print(i1,(i1/np.sum(nb_llj_oc[s,:]))*100,(i1/np.sum(nb_oc[s,:]))*100)
    mean_prof=np.sum(interp_prof_all[s,0:i1,:],axis=0)/float(i1)
    ax.plot(mean_prof,heights,label=stations_names[s],linewidth=3)
    del i1,mean_prof
ax.set_xlabel(r"Wind speed (m s$^{-1}$)",fontsize=16)
ax.set_ylabel("Height (m)",fontsize=16)
ax.set_xlim(0,15)
ax.set_ylim(0,1500)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='both', which='major', width=1, length=10)
ax.legend(loc='upper left',fontsize=16,frameon=False)
#plt.savefig("Mean_profile_LLJBelow500m_2000_2015_all_stations_HeightAboveGround.pdf",format="pdf",dpi=150, bbox_inches='tight', pad_inches=0.1)
plt.show()
del fig,ax


# Plots of height/speed distributions
from mpl_toolkits.axes_grid1 import make_axes_locatable
ulim_h=[12,16,12,12,12,31,13]
ulim_s=[10,12,10,10,10,16,10]

    
# With precentage of LLJ
for s in range(len(s_sel)):
    print(stations_names[s])
    spd=np.array(llj_spd_all[s])
    hist_spd,bs=np.histogram(spd,bins=np.arange(-0.5,61.5,1))
    hist_spd=(hist_spd/np.sum(hist_spd))*100
    hgt=np.array(llj_hgt_all[s])
    hist_hgt,bh=np.histogram(hgt,bins=np.arange(-25,1600,50))
    hist_hgt=(hist_hgt/np.sum(hist_hgt))*100
    hist2d_spd_hgt,xe,ye=np.histogram2d(hgt,spd,bins=[np.arange(-25,1600,50),np.arange(-0.5,61.5,1)])
    hist2d_spd_hgt=(hist2d_spd_hgt/np.sum(hist2d_spd_hgt))*100
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    # The central plot
    hist2d_spd_hgt=np.where(hist2d_spd_hgt==0,np.nan,hist2d_spd_hgt)
    p=ax.pcolormesh(bs[0:-1]+0.5,bh[0:-1]+25, hist2d_spd_hgt, cmap='plasma',vmin=0,vmax=2)
    ax.set_xlabel(r"Wind speed (m s$^{-1}$)",fontsize=20)
    ax.set_ylabel("Height (m)",fontsize=20)
    ax.set_xlim(0,60)
    ax.set_ylim(0,1500)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='both', which='major', width=1, length=10)
    # create new axes on the right and on the top of the current axes
    divider = make_axes_locatable(ax)
    # below height and pad are in inches
    ax_histx = divider.append_axes("top", 1.2, pad=0.25, sharex=ax)
    ax_histy = divider.append_axes("right", 1.2, pad=0.25, sharey=ax)
    # make some labels invisible
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    ax_histx.bar(bs[0:-1]+0.5, hist_spd, edgecolor='k',facecolor='None',width=1)
    ax_histy.barh(bh[0:-1]+25, hist_hgt, edgecolor='k',facecolor='None',height=50)
    ax_histy.tick_params(axis='x', labelsize=16)
    ax_histx.tick_params(axis='y', labelsize=16)
    ax_histx.tick_params(axis='both', which='major', width=1, length=10)
    ax_histy.tick_params(axis='both', which='major', width=1, length=10)
    ax_histy.set_xlim(0,ulim_h[s])
    ax_histx.set_ylim(0,ulim_s[s])
    ax_histx.text(42,6,r'$\mu$='+"{:5.2f}".format(np.mean(spd)),fontsize=12)
    ax_histx.text(42,4,'m='+"{:5.2f}".format(np.median(spd)),fontsize=12)
    ax_histx.text(42,2,r'$\sigma$='+"{:5.2f}".format(np.std(spd)),fontsize=12)
    ax_histy.text(5,1350,r'$\mu$='+"{:5.2f}".format(np.mean(hgt)),fontsize=12)
    ax_histy.text(5,1250,'m='+"{:5.2f}".format(np.median(hgt)),fontsize=12)
    ax_histy.text(5,1150,r'$\sigma$='+"{:5.2f}".format(np.std(hgt)),fontsize=12)
    del spd,hgt
    cbar_ax = fig.add_axes([0.92, 0.125, 0.04, 0.493])
    cbar_ax.tick_params(labelsize=16)
    fig.colorbar(p, cax=cbar_ax,extend='max')
    #plt.savefig("Histograms2D_Histograms1D_Percentage_LLJSpeedHeight_2000_2015_"+stations_names[s]+"_HeightAboveGround.pdf",format="pdf",dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.show()


# How many LLJs have a speed above 20 and 30 m/s?
for s in range(len(s_sel)):
    print(stations_names[s])
    spd=np.array(llj_spd_all[s])
    spd20=len(np.where(spd>20.0)[0])
    spd30=len(np.where(spd>30.0)[0])
    print(len(llj_spd_all[s]),spd20,(spd20*100)/len(llj_spd_all[s]),(spd30*100)/len(llj_spd_all[s]))



################## NEW NEW NEW NEW #######################################
hist2d_month_hour_2=np.zeros((len(s_sel),5,12))
for s in range(len(s_sel)):
#for s in range(1):
    hr=np.array([int(dt[11:13]) for dt in llj_date_all[s]])
    mn=np.array([int(dt[14:16]) for dt in llj_date_all[s]])
    mon=np.array([int(dt[5:7]) for dt in llj_date_all[s]])
    dy=np.array([int(dt[8:10]) for dt in llj_date_all[s]])
    yr=np.array([int(dt[0:4]) for dt in llj_date_all[s]])
    llj_dat=np.array(llj_date_all[s])
    # 2d histogram
    for t in range(len(hr)):
        hrmi=int(hr[t]*100)+int(mn[t])
        #print('hrmi',hrmi,'date=',llj_dat[t])
        if ((hrmi>2230) | (hrmi<=330)):
            ha=0
            #if (hrmi>2230):
            #    if ((mon[t]==12) & (dy[t]==31)):
            #        #mon[t]=1
            #        dy[t]=1
            #        yr[t]=yr[t]+1
            #        #print('Case 1',llj_dat[t],hrmi,yr[t],mon[t],dy[t])
            #    elif ((mon[t]==2) & (yr[t]%4==0) & (dy[t]==29)):
            #        #mon[t]=mon[t]+1
            #        dy[t]=1
            #        #print('Case 2',llj_dat[t],hrmi,yr[t],mon[t],dy[t])
            #    elif ((mon[t]==2) & (yr[t]%4!=0) & (dy[t]==28)):
            #        #mon[t]=mon[t]+1
            #        dy[t]=1
            #        #print('Case 3',llj_dat[t],hrmi,yr[t],mon[t],dy[t])
            #    elif ((dy[t]==31) & ((mon[t]==1) | (mon[t]==3) | (mon[t]==5) | (mon[t]==7) | (mon[t]==8) | (mon[t]==10))):
            #        #mon[t]=mon[t]+1
            #        dy[t]=1
            #        #print('Case 4',llj_dat[t],hrmi,yr[t],mon[t],dy[t])
            #    elif ((dy[t]==30) & ((mon[t]==4) | (mon[t]==6) | (mon[t]==9) | (mon[t]==11))):
            #        #mon[t]=mon[t]+1
            #        dy[t]=1
            #        #print('Case 5',llj_dat[t],hrmi,yr[t],mon[t],dy[t])
        elif ((hrmi>330) & (hrmi<=930)):
            ha=1
        elif ((hrmi>930) & (hrmi<=1530)):
            ha=2
        elif ((hrmi>1530) & (hrmi<=2230)):
            ha=3
        hist2d_month_hour_2[s,ha,mon[t]-1]=hist2d_month_hour_2[s,ha,mon[t]-1]+1
        del ha,hrmi
    if np.sum(hist2d_month_hour_2[s,:,:])!=len(hr):
        print('Problem',np.sum(hist2d_month_hour_2[s,:,:]),len(i_ev))
    del hr,mn,mon,dy,yr


hist1d_month_2=np.zeros((len(s_sel),12))
for s in range(len(s_sel)):
#for s in range(1):
    mon=np.array([int(dt[5:7]) for dt in llj_date_all[s]])
    llj_dat=np.array(llj_date_all[s])
    # 1d histogram
    for t in range(len(mon)):
        hist1d_month_2[s,mon[t]-1]=hist1d_month_2[s,mon[t]-1]+1
    if np.sum(hist1d_month_2[s,:])!=len(mon):
        print('Problem',np.sum(hist1d_month_2[s,:]),len(i_ev))
    del mon



# Number of dates over which the detection algorithm is going through
count_ts=np.zeros((len(s_sel),4,12))
for s in range(len(s_sel)):
    yr=np.zeros((len(dates_detect_all[s])),dtype='int32')
    mn=np.zeros((len(dates_detect_all[s])),dtype='int8')
    dy=np.zeros((len(dates_detect_all[s])),dtype='int8')
    hr=np.zeros((len(dates_detect_all[s])),dtype='int8')
    mi=np.zeros((len(dates_detect_all[s])),dtype='int8')
    for t in range(len(dates_detect_all[s])):
        yr[t]=int(year_all[s][t])
        mn[t]=int(month_all[s][t])
        dy[t]=int(dates_detect_all[s][t][8:10])
        hr[t]=int(dates_detect_all[s][t][11:13])
        mi[t]=int(dates_detect_all[s][t][14:16])
    #print(np.min(yr),np.max(yr))
    #print(np.min(mn),np.max(mn))
    #print(np.min(dy),np.max(dy))
    #print(np.min(hr),np.max(hr))
    #print(np.min(mi),np.max(mi))
    for t in range(len(dates_detect_all[s])):
        hrmi=int(hr[t]*100)+int(mi[t])
        if ((hrmi>2230) | (hrmi<=330)):
            ha=0
            if (hrmi>2230):
                if ((mn[t]==12) & (dy[t]==31)):
                    #mn[t]=1
                    dy[t]=1
                    yr[t]=yr[t]+1
                    #print('Case 1',dates_detect_all[s][t],hrmi,yr[t],mn[t],dy[t])
                elif ((mn[t]==2) & (yr[t]%4==0) & (dy[t]==29)):
                    #mn[t]=mn[t]+1
                    dy[t]=1
                    #print('Case 2',dates_detect_all[s][t],hrmi,yr[t],mn[t],dy[t])
                elif ((mn[t]==2) & (yr[t]%4!=0) & (dy[t]==28)):
                    mn[t]=mn[t]+1
                    dy[t]=1
                    #print('Case 3',dates_detect_all[s][t],hrmi,yr[t],mn[t],dy[t])
                elif ((dy[t]==31) & ((mn[t]==1) | (mn[t]==3) | (mn[t]==5) | (mn[t]==7) | (mn[t]==8) | (mn[t]==10))):
                    mn[t]=mn[t]+1
                    dy[t]=1
                    #print('Case 4',dates_detect_all[s][t],hrmi,yr[t],mn[t],dy[t])
                elif ((dy[t]==30) & ((mn[t]==4) | (mn[t]==6) | (mn[t]==9) | (mn[t]==11))):
                    mn[t]=mn[t]+1
                    dy[t]=1
                    #print('Case 5',dates_detect_all[s][t],hrmi,yr[t],mn[t],dy[t])
        elif ((hrmi>330) & (hrmi<=930)):
            ha=1
        elif ((hrmi>930) & (hrmi<=1530)):
            ha=2
        elif ((hrmi>1530) & (hrmi<=2230)):
            ha=3
        count_ts[s,ha,mn[t]-1]=count_ts[s,ha,mn[t]-1]+1
        del ha,hrmi
    del yr,mn,dy,hr,mi
    if np.sum(count_ts[s,:,:])!=len(dates_detect_all[s]):
        print('Problem',np.sum(count_ts[s,:,:]),len(dates_detect_all[s]))



print(hist1d_month_2)
for i in range(len(s_sel)):
    hist2d_month_hour_2[i,4,:]=hist1d_month_2[i,:]/np.sum(hist1d_month_2[i,:])*100

print(hist2d_month_hour_2[:,4,:])
print(np.sum(hist1d_month_2,axis=1))

count_hr=np.sum(hist2d_month_hour_2[:,0:4,:],axis=2)
print(count_hr)
print(np.shape(count_hr))



# Plot annual and diurnal cycles of low-level jets
hist2d_month_hour_2=np.zeros((len(s_sel),5,12))
for s in range(len(s_sel)):
    hr=np.array([int(dt[11:13]) for dt in dates_llj[s]])
    mn=np.array([int(dt[14:16]) for dt in dates_llj[s]])
    mon=np.array([int(dt[5:7]) for dt in dates_llj[s]])
    dy=np.array([int(dt[8:10]) for dt in dates_llj[s]])
    yr=np.array([int(dt[0:4]) for dt in dates_llj[s]])
    llj_dat=np.array(dates_llj[s])
    # 2d histogram
    for t in range(len(hr)):
        hrmi=int(hr[t]*100)+int(mn[t])
        if ((hrmi>2230) | (hrmi<=330)):
            ha=0
        elif ((hrmi>330) & (hrmi<=930)):
            ha=1
        elif ((hrmi>930) & (hrmi<=1530)):
            ha=2
        elif ((hrmi>1530) & (hrmi<=2230)):
            ha=3
        hist2d_month_hour_2[s,ha,mon[t]-1]=hist2d_month_hour_2[s,ha,mon[t]-1]+1
        del ha,hrmi
    if np.sum(hist2d_month_hour_2[s,:,:])!=len(hr):
        print('Problem',np.sum(hist2d_month_hour_2[s,:,:]),len(i_ev))
    del hr,mn,mon,dy,yr


hist1d_month_2=np.zeros((len(s_sel),12))
for s in range(len(s_sel)):
    mon=np.array([int(dt[5:7]) for dt in dates_llj[s]])
    llj_dat=np.array(dates_llj[s])
    # 1d histogram
    for t in range(len(mon)):
        hist1d_month_2[s,mon[t]-1]=hist1d_month_2[s,mon[t]-1]+1
    if np.sum(hist1d_month_2[s,:])!=len(mon):
        print('Problem',np.sum(hist1d_month_2[s,:]),len(i_ev))
    del mon

# Number of dates over which the detection algorithm is going through
count_ts=np.zeros((len(s_sel),4,12))
for s in range(len(s_sel)):
    yr=np.zeros((len(dates_all[s])),dtype='int32')
    mn=np.zeros((len(dates_all[s])),dtype='int8')
    dy=np.zeros((len(dates_all[s])),dtype='int8')
    hr=np.zeros((len(dates_all[s])),dtype='int8')
    mi=np.zeros((len(dates_all[s])),dtype='int8')
    for t in range(len(dates_all[s])):
        yr[t]=int(year_all[s][t])
        mn[t]=int(month_all[s][t])
        dy[t]=int(dates_all[s][t][8:10])
        hr[t]=int(dates_all[s][t][11:13])
    for t in range(len(dates_all[s])):
        hrmi=int(hr[t]*100)+int(mi[t])
        if ((hrmi>2230) | (hrmi<=330)):
            ha=0
            if (hrmi>2230):
                if ((mn[t]==12) & (dy[t]==31)):
                    dy[t]=1
                    yr[t]=yr[t]+1
                    #print('Case 1',dates_detect_all[s][t],hrmi,yr[t],mn[t],dy[t])
                elif ((mn[t]==2) & (yr[t]%4==0) & (dy[t]==29)):
                    dy[t]=1
                    #print('Case 2',dates_detect_all[s][t],hrmi,yr[t],mn[t],dy[t])
                elif ((mn[t]==2) & (yr[t]%4!=0) & (dy[t]==28)):
                    mn[t]=mn[t]+1
                    dy[t]=1
                    #print('Case 3',dates_detect_all[s][t],hrmi,yr[t],mn[t],dy[t])
                elif ((dy[t]==31) & ((mn[t]==1) | (mn[t]==3) | (mn[t]==5) | (mn[t]==7) | (mn[t]==8) | (mn[t]==10))):
                    mn[t]=mn[t]+1
                    dy[t]=1
                    #print('Case 4',dates_detect_all[s][t],hrmi,yr[t],mn[t],dy[t])
                elif ((dy[t]==30) & ((mn[t]==4) | (mn[t]==6) | (mn[t]==9) | (mn[t]==11))):
                    mn[t]=mn[t]+1
                    dy[t]=1
                    #print('Case 5',dates_detect_all[s][t],hrmi,yr[t],mn[t],dy[t])
        elif ((hrmi>330) & (hrmi<=930)):
            ha=1
        elif ((hrmi>930) & (hrmi<=1530)):
            ha=2
        elif ((hrmi>1530) & (hrmi<=2230)):
            ha=3
        count_ts[s,ha,mn[t]-1]=count_ts[s,ha,mn[t]-1]+1
        del ha,hrmi
    del yr,mn,dy,hr,mi
    if np.sum(count_ts[s,:,:])!=len(dates_all[s]):
        print('Problem',np.sum(count_ts[s,:,:]),len(dates_all[s]))

for i in range(len(s_sel)):
    hist2d_month_hour_2[i,4,:]=hist1d_month_2[i,:]/np.sum(hist1d_month_2[i,:])*100

count_hr=np.sum(hist2d_month_hour_2[:,0:4,:],axis=2)

lim_blues=[[10,35],[10,50],[5,40],[5,25],[10,30],[10,50],[4,14]]

for s in range(len(s_sel)):
    hh_new=np.where(count_ts[s,:,:]==0,np.nan,(hist2d_month_hour_2[s,0:4,:]/count_ts[s,:,:])*100)
    fig, ax = plt.subplots(1,1,figsize=(7.5, 3.5))
    p=ax.pcolormesh(np.arange(1,13,1),np.arange(0,4,1),hh_new,cmap='Blues',shading='auto',vmin=lim_blues[s][0],vmax=lim_blues[s][1])
    p1=ax.pcolormesh([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5],[3.5,4.5],[hist2d_month_hour_2[s,4,:]],cmap='Reds',shading='auto',vmin=5,vmax=13)
    count_hr1=np.sum(hist2d_month_hour_2[s,0:4,:],axis=1)/np.sum(count_hr[s,:])*100
    hc=[[i] for i in count_hr1]
    p2=ax.pcolormesh([12.5,13.5],[-0.5,  0.5,  1.5,  2.5,  3.5],hc,cmap='Purples',shading='auto',vmin=10,vmax=50)
    # The black lines
    ax.plot(np.arange(-2,15,1),[3.5]*len(np.arange(-2,15,1)),color='k',linewidth=1)
    ax.plot([12.5]*len(np.arange(-2,6,1)),np.arange(-2,6,1),color='k',linewidth=1)
    # Crosses where there is no launch
    zj=np.where(count_ts[s,:,:]==0)[0].astype('float')
    zi=np.where(count_ts[s,:,:]==0)[1].astype('float')
    zi=zi+1.0 # x-axis starts from 1
    if len(zi)>0:
        for z in range(len(zi)):
            xz=np.arange(zi[z]-0.5,zi[z]+0.6,0.1)
            yz=np.arange(zj[z]-0.5,zj[z]+0.6,0.1)
            yz1=np.arange(zj[z]+0.5,zj[z]-0.6,-0.1)
            ax.plot(xz,yz,color='b')
            ax.plot(xz,yz1,color='b')
    del zi,zj
    ax.set_ylim(-0.5,4.5)
    ax.set_xlim(0.5,13.5)
    ax.set_yticks([0,1,2,3,4])
    ax.set_yticklabels(['00','06','12','18','All'],fontsize=18)
    ax.set_xticks(np.arange(1,14,1))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D','All'],fontsize=18)
    ax.tick_params(axis='both', which='major', width=1, length=10)
    ax.set_ylabel('Hour of the day (UTC)',fontsize=18)
    plt.title(stations_names[s],fontsize=20)
    cbar_ax = fig.add_axes([0.92, 0.125, 0.04, 0.754])
    cbar_ax.tick_params(labelsize=16)
    fig.colorbar(p, cax=cbar_ax,extend='both')
    cbar_ax1 = fig.add_axes([1.03, 0.125, 0.04, 0.754])
    cbar_ax1.tick_params(labelsize=16)
    fig.colorbar(p1, cax=cbar_ax1,extend='both')
    cbar_ax2 = fig.add_axes([1.14, 0.125, 0.04, 0.754])
    cbar_ax2.tick_params(labelsize=16)
    fig.colorbar(p2, cax=cbar_ax2,extend='both')
    #plt.savefig('PercentageLLJ_Month_Hour_2000_2015_'+stations_names[s]+'_crosses.pdf',format='pdf',dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    del fig,ax
    

for s in range(len(s_sel)):
    fig, ax = plt.subplots(1,1,figsize=(7.5, 3.5))      
    p=ax.pcolormesh(np.arange(1,13,1),np.arange(0,4,1),count_ts[s,:,:],cmap='Blues',shading='auto')
    # Crosses where there is no launch
    zj=np.where(count_ts[s,:,:]==0)[0].astype('float')
    zi=np.where(count_ts[s,:,:]==0)[1].astype('float')
    zi=zi+1.0 # x-axis starts from 1
    if len(zi)>0:
        for z in range(len(zi)):
            xz=np.arange(zi[z]-0.5,zi[z]+0.6,0.1)
            yz=np.arange(zj[z]-0.5,zj[z]+0.6,0.1)
            yz1=np.arange(zj[z]+0.5,zj[z]-0.6,-0.1)
            ax.plot(xz,yz,color='b')
            ax.plot(xz,yz1,color='b')
    del zi,zj
    ax.set_ylim(-0.5,3.5)
    ax.set_xlim(0.5,12.5)
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(['00','06','12','18'],fontsize=18)
    ax.set_xticks(np.arange(1,13,1))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],fontsize=18)
    ax.tick_params(axis='both', which='major', width=1, length=10)
    ax.set_ylabel('Hour of the day (UTC)',fontsize=18)
    plt.title(stations_names[s],fontsize=20)
    cbar_ax = fig.add_axes([0.92, 0.125, 0.04, 0.754])
    cbar_ax.tick_params(labelsize=16)
    if s<6:
        fig.colorbar(p, cax=cbar_ax,extend='max')
    else:
        fig.colorbar(p, cax=cbar_ax,extend='both')
    #plt.savefig('Number_Launches_Month_Hour_2000_2015_'+stations_names[s]+'_crosses.pdf',format='pdf',dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    del fig,ax


##################################
# Three subroutines for wind roses
import matplotlib.patches as mp
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
import math
from copy import copy
# Subroutine to plot a 2-d wind rose
# Input values 2D array (speed/height,direction), bins edges (the direction values should be from 0 to 360)
# another input is the list of values for the frequency circles
# filename is the name of the output figure
def wind_rose_2d(val,bins_var2,bins_edges,freq_circles,filename):
    bins_edges=bins_edges[0:-1]
    # Bins center (bin_edges is the input)
    diff=(bins_edges[1]-bins_edges[0])/2.0
    bins_dir_center=bins_edges[0:-1]+diff
    # I have to reverse the arrays because matplotlib plots counter-clockwise
    s1=np.where(bins_dir_center<=90.0)[0][::-1]
    s2=np.where(bins_dir_center>90.0)[0][::-1]
    hist=np.concatenate((val[:,s1],val[:,s2]),axis=1)
    arr=(bins_dir_center[s1]+diff,bins_dir_center[s2[0:]]+diff,[bins_edges[s1[0]+1]])
    angles=bins_edges
    # Get maximum frequency (not used here)
    max_freq=np.max(np.sum(val,axis=0))+1
    cmap = cm.get_cmap('viridis')
    colors=cmap(np.linspace(0,1,np.shape(val)[0]))
    # Open figure
    fig, ax = plt.subplots()
    # Loop over the categories
    for v in range(np.shape(val)[0]):
        patch=[]
        # Loop over the directions
        if v==0:
            for d in range(np.shape(val)[1]):
                patch.append(mp.Wedge((0,0), hist[v,d], angles[d], angles[d+1]))
            p = PatchCollection(patch, alpha=1,edgecolor='k',facecolor=colors[v],linewidth=0.5)
            ax.add_collection(p)
        else:
            for d in range(np.shape(val)[1]):
                patch.append(mp.Wedge((0,0), np.sum(hist[0:v+1,d]), angles[d], angles[d+1], width=hist[v,d]))
            p = PatchCollection(patch, alpha=1,edgecolor='k',facecolor=colors[v],linewidth=0.5)
            ax.add_collection(p)
    # Circles of frequency
    circles=[]
    for r in freq_circles:
        circles.append(mp.Circle((0,0), r, fill=False, alpha=1, color='gray',linewidth=1,linestyle='dotted'))
    for c in circles:
        new_c=copy(c)
        ax.add_patch(new_c)
    ax.axhline(y=0, xmin=0.02, xmax=0.98, alpha=1, color='gray', linewidth=1, linestyle='dotted')
    ax.axvline(x=0, ymin=0.02, ymax=0.98, alpha=1, color='gray', linewidth=1, linestyle='dotted')
    bd=math.cos(math.pi/4.0)*freq_circles[-1]
    ax.plot(np.linspace(0-bd,bd,21),np.linspace(0-bd,bd,21), alpha=1, color='gray', linewidth=1, linestyle='dotted')
    ax.plot(np.linspace(0-bd,bd,21),np.linspace(bd,0-bd,21), alpha=1, color='gray', linewidth=1, linestyle='dotted')
    bbox={'boxstyle':'round','ec':'white','fc':'white'}
    for r in freq_circles:
        xlab=r*math.cos(70.0*math.pi/180.0)
        ylab=r*math.sin(70.0*math.pi/180.0)
        plt.text(xlab, ylab, str(r)+'%', size=10, color='k', ha="center", va="center")
    ofs=freq_circles[-1]*0.1
    plt.text(0, freq_circles[-1]+ofs, 'N', size=15, color='k', ha="center", va="center")
    plt.text(freq_circles[-1]+ofs, 0, 'E', size=15, color='k', ha="center", va="center")
    plt.text(0, 0-freq_circles[-1]-ofs, 'S', size=15, color='k', ha="center", va="center")
    plt.text(0-freq_circles[-1]-ofs, 0, 'W', size=15, color='k', ha="center", va="center")
    ax.axis('off')
    ax.set_aspect('equal')
    plt.ylim(0-freq_circles[-1]-0.5,freq_circles[-1]+0.5)
    plt.xlim(0-freq_circles[-1]-0.5,freq_circles[-1]+0.5)
    #plt.savefig(filename,format='pdf',dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
    
# Subroutine to plot a 2-d wind rose with 4 categories (the seasons here)
# Input values 2D array (speed/height,direction), bins edges (the direction values should be from 0 to 360)
# another input is the list of values for the frequency circles
# filename is the name of the output figure
def wind_rose_2d_seasons(val,bins_var2,bins_edges,freq_circles,filename):
    bins_edges=bins_edges[0:-1]
    # Bins center (bin_edges is the input)
    diff=(bins_edges[1]-bins_edges[0])/2.0
    bins_dir_center=bins_edges[0:-1]+diff
    # I have to reverse the arrays because matplotlib plots counter-clockwise
    s1=np.where(bins_dir_center<=90.0)[0][::-1]
    s2=np.where(bins_dir_center>90.0)[0][::-1]
    hist=np.concatenate((val[:,s1],val[:,s2]),axis=1)
    arr=(bins_dir_center[s1]+diff,bins_dir_center[s2[0:]]+diff,[bins_edges[s1[0]+1]])
    angles=bins_edges
    # Get maximum frequency (not used here)
    max_freq=np.max(np.sum(val,axis=0))+1
    colors=['blue','lightgreen','red','gold']
    # Open figure
    fig, ax = plt.subplots()
    # Loop over the categories
    for v in range(np.shape(val)[0]):
        patch=[]
        # Loop over the directions
        if v==0:
            for d in range(np.shape(val)[1]):
                patch.append(mp.Wedge((0,0), hist[v,d], angles[d], angles[d+1]))
            p = PatchCollection(patch, alpha=1,edgecolor='k',facecolor=colors[v],linewidth=0.5)
            ax.add_collection(p)
        else:
            for d in range(np.shape(val)[1]):
                patch.append(mp.Wedge((0,0), np.sum(hist[0:v+1,d]), angles[d], angles[d+1], width=hist[v,d]))
            p = PatchCollection(patch, alpha=1,edgecolor='k',facecolor=colors[v],linewidth=0.5)
            ax.add_collection(p)
    # Circles of frequency
    circles=[]
    for r in freq_circles:
        circles.append(mp.Circle((0,0), r, fill=False, alpha=1, color='gray',linewidth=1,linestyle='dotted'))
    for c in circles:
        new_c=copy(c)
        ax.add_patch(new_c)
    ax.axhline(y=0, xmin=0.02, xmax=0.98, alpha=1, color='gray', linewidth=1, linestyle='dotted')
    ax.axvline(x=0, ymin=0.02, ymax=0.98, alpha=1, color='gray', linewidth=1, linestyle='dotted')
    bd=math.cos(math.pi/4.0)*freq_circles[-1]
    ax.plot(np.linspace(0-bd,bd,21),np.linspace(0-bd,bd,21), alpha=1, color='gray', linewidth=1, linestyle='dotted')
    ax.plot(np.linspace(0-bd,bd,21),np.linspace(bd,0-bd,21), alpha=1, color='gray', linewidth=1, linestyle='dotted')
    bbox={'boxstyle':'round','ec':'white','fc':'white'}
    for r in freq_circles:
        xlab=r*math.cos(70.0*math.pi/180.0)
        ylab=r*math.sin(70.0*math.pi/180.0)
        plt.text(xlab, ylab, str(r)+'%', size=10, color='k', ha="center", va="center")
    ofs=freq_circles[-1]*0.1
    plt.text(0, freq_circles[-1]+ofs, 'N', size=15, color='k', ha="center", va="center")
    plt.text(freq_circles[-1]+ofs, 0, 'E', size=15, color='k', ha="center", va="center")
    plt.text(0, 0-freq_circles[-1]-ofs, 'S', size=15, color='k', ha="center", va="center")
    plt.text(0-freq_circles[-1]-ofs, 0, 'W', size=15, color='k', ha="center", va="center")
    ax.axis('off')
    ax.set_aspect('equal')
    plt.ylim(0-freq_circles[-1]-0.5,freq_circles[-1]+0.5)
    plt.xlim(0-freq_circles[-1]-0.5,freq_circles[-1]+0.5)
    #plt.savefig(filename,format='pdf',dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.show()


# Subroutine to plot a 1-d wind rose (only one variable)
# Input values 1D array (direction), bins edges (the direction values should be from 0 to 360)
# another input is the list of values for the frequency circles
# filename is the name of the output figure
def wind_rose_1d(val,bins_edges,freq_circles,filename):
    bins_edges=bins_edges[0:-1]
    # Bins center (bin_edges is the input)
    diff=(bins_edges[1]-bins_edges[0])/2.0
    bins_dir_center=bins_edges[0:-1]+diff
    # I have to reverse the arrays because matplotlib plots counter-clockwise
    s1=np.where(bins_dir_center<=90.0)[0][::-1]
    s2=np.where(bins_dir_center>90.0)[0][::-1]
    hist=np.concatenate((val[s1],val[s2]),axis=0)
    arr=(bins_dir_center[s1]+diff,bins_dir_center[s2[0:]]+diff,[bins_edges[s1[0]+1]])
    angles=bins_edges
    # Get maximum frequency (not used here)
    max_freq=np.max(np.sum(val,axis=0))+1
    # Open figure
    fig, ax = plt.subplots()
    patch=[]
    # Loop over the directions
    for d in range(np.shape(val)[0]):
        patch.append(mp.Wedge((0,0), hist[d], angles[d], angles[d+1]))
    p = PatchCollection(patch, alpha=1,edgecolor='k',facecolor='lightseagreen',linewidth=0.5)
    ax.add_collection(p)
    # Circles of frequency
    circles=[]
    for r in freq_circles:
        circles.append(mp.Circle((0,0), r, fill=False, alpha=1, color='gray',linewidth=1,linestyle='dotted'))
    for c in circles:
        new_c=copy(c)
        ax.add_patch(new_c)
    ax.axhline(y=0, xmin=0.02, xmax=0.98, alpha=1, color='gray', linewidth=1, linestyle='dotted')
    ax.axvline(x=0, ymin=0.02, ymax=0.98, alpha=1, color='gray', linewidth=1, linestyle='dotted')
    bd=math.cos(math.pi/4.0)*freq_circles[-1]
    ax.plot(np.linspace(0-bd,bd,21),np.linspace(0-bd,bd,21), alpha=1, color='gray', linewidth=1, linestyle='dotted')
    ax.plot(np.linspace(0-bd,bd,21),np.linspace(bd,0-bd,21), alpha=1, color='gray', linewidth=1, linestyle='dotted')
    bbox={'boxstyle':'round','ec':'white','fc':'white'}
    for r in freq_circles:
        xlab=r*math.cos(70.0*math.pi/180.0)
        ylab=r*math.sin(70.0*math.pi/180.0)
        plt.text(xlab, ylab, str(r)+'%', size=10, color='k', ha="center", va="center")
    ofs=freq_circles[-1]*0.1
    plt.text(0, freq_circles[-1]+ofs, 'N', size=15, color='k', ha="center", va="center")
    plt.text(freq_circles[-1]+ofs, 0, 'E', size=15, color='k', ha="center", va="center")
    plt.text(0, 0-freq_circles[-1]-ofs, 'S', size=15, color='k', ha="center", va="center")
    plt.text(0-freq_circles[-1]-ofs, 0, 'W', size=15, color='k', ha="center", va="center")
    ax.axis('off')
    ax.set_aspect('equal')
    plt.ylim(0-freq_circles[-1]-0.5,freq_circles[-1]+0.5)
    plt.xlim(0-freq_circles[-1]-0.5,freq_circles[-1]+0.5)
    #plt.savefig(filename,format='pdf',dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.show()

##################################

# Plot 1-d wind roses
# Circles limits
circ_lim=[[2,4,6,8,10,12],[5,10,15,20,25],[4,8,12,16],[2,4,6,8],[2,4,6,8],[10,20,30],[2,4,6,8]]
for s in range(len(s_sel)):
    h_dir,bins_edg=np.histogram(np.array(llj_dir_all[s]),bins=np.arange(-5,370,10))
    # Just get from 0 to 350 and sum the last two bins (-5 to 5 and 355 to 365)
    h_dir1=h_dir[0:-1]
    h_dir1[0]=h_dir1[0]+h_dir[-1]
    wind_rose_1d((h_dir1/np.sum(h_dir1))*100,bins_edg,circ_lim[s],'Windrose_LLJ_2000_2015_'+stations_names[s]+'.pdf')
    del bins_edg,h_dir,h_dir1

# Plot 2-d wind roses (as a function of height)
for s in range(len(s_sel)):
    h2d_dir,bins_spd1,bins_dir1=np.histogram2d(np.array(llj_spd_all[s]),np.array(llj_dir_all[s]),bins=[np.array([0,10,20,100]),np.arange(-5,370,10)])
    # Just get from 0 to 350 and sum the last two bins (-5 to 5 and 355 to 365)
    h2d_dir1=h2d_dir[:,0:-1]
    h2d_dir1[:,0]=h2d_dir1[:,0]+h2d_dir[:,-1]
    wind_rose_2d((h2d_dir1/np.sum(h2d_dir1))*100,bins_spd1,bins_dir1,circ_lim[s],'Windrose_DirSpd_LLJ_2000_2015_'+stations_names[s]+'.pdf')
    del h2d_dir,h2d_dir1

# Plot 2-d wind roses (as a function of speed)
for s in range(len(s_sel)):
    h2d_dir,bins_hgt1,bins_dir1=np.histogram2d(np.array(llj_hgt_all[s]),np.array(llj_dir_all[s]),bins=[np.array([0,500,1000,1500]),np.arange(-5,370,10)])
    #print(h2d_dir)
    # Just get from 0 to 350 and sum the last two bins (-5 to 5 and 355 to 365)
    h2d_dir1=h2d_dir[:,0:-1]
    h2d_dir1[:,0]=h2d_dir1[:,0]+h2d_dir[:,-1]
    wind_rose_2d((h2d_dir1/np.sum(h2d_dir1))*100,bins_spd1,bins_dir1,circ_lim[s],'Windrose_DirHgt_LLJ_2000_2015_'+stations_names[s]+'.pdf')
    del h2d_dir,h2d_dir1

# Plot 2-d wind roses (as a function of season)
for s in range(len(s_sel)):
    mon=np.array([int(dt[5:7]) for dt in llj_date_all[s]])
    for i in range(len(mon)):
        if ((mon[i]<=2) | (mon[i]==12)):
            mon[i]=0
        elif ((mon[i]>=3) & (mon[i]<=5)):
            mon[i]=1
        elif ((mon[i]>=6) & (mon[i]<=8)):
            mon[i]=2
        elif ((mon[i]>=9) & (mon[i]<=11)):
            mon[i]=3
    h2d_dir,bins_mon1,bins_dir1=np.histogram2d(mon,np.array(llj_dir_all[s]),bins=[np.arange(-0.5,4.5,1),np.arange(-5,370,10)])
    # Just get from 0 to 350 and sum the last two bins (-5 to 5 and 355 to 365)
    h2d_dir1=h2d_dir[:,0:-1]
    h2d_dir1[:,0]=h2d_dir1[:,0]+h2d_dir[:,-1]
    wind_rose_2d_seasons((h2d_dir1/np.sum(h2d_dir1))*100,bins_mon1,bins_dir1,circ_lim[s],'Windrose_DirSeas_LLJ_2000_2015_'+stations_names[s]+'.pdf')
    del h2d_dir,h2d_dir1,mon



# Plot legends of the wind roses separately
def export_legend(legend, filename):
    fig  = legend.figure
    fig.canvas.draw()
    bbox_leg = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox_leg)


# Legend of wind rose for speed
cmap = cm.get_cmap('viridis')
colors=cmap(np.linspace(0,1,len(bins_spd1)-1))
labels=[str(int(bins_spd1[i]))+"-"+str(int(bins_spd1[i+1]))+" m s$^{-1}$" for i in range(len(bins_spd1[0:-1])-1)]
labels=labels+['>'+str(int(bins_spd1[-2]))+" m s$^{-1}$"]
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
handles = [f("s", colors[i]) for i in range(len(labels))]
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)
export_legend(legend,"legend_windrose_spd_dir.pdf")
plt.show()

# Legend of wind rose for height
cmap = cm.get_cmap('viridis')
colors=cmap(np.linspace(0,1,len(bins_hgt1)-1))
labels=[str(int(bins_hgt1[i]))+"-"+str(int(bins_hgt1[i+1]))+" m" for i in range(len(bins_spd1)-1)]
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
handles = [f("s", colors[i]) for i in range(len(labels))]
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)
export_legend(legend,"legend_windrose_hgt_dir.pdf")
plt.show()

# Legend of wind rose for seasons
colors=['blue','lightgreen','red','gold']
labels=['DJF','MAM','JJA','SON']
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
handles = [f("s", colors[i]) for i in range(len(labels))]
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)
export_legend(legend,"legend_windrose_season_dir.pdf")
plt.show()




