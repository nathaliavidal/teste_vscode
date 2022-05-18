# Autor: Piero Morandim Maduro
# Created at 13/04/2022
# Glider output data 
# READ NOTES OF "Choose folder with binary files and cache file" before use it!!!
import os
import pandas as pd
from io import StringIO
import datetime as dt
import gsw
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import glob
import numpy as np
from geopy import distance

def outOfEnvLims(series, min_envlim, max_envlim):
    '''
    Check for values out of environmental limits.
    Input:
        series (pandas.core.series.Series): Parameter's series.
        min_envlim (float): Minimum acceptable value according to local conditions.
        max_envlim (float): Maximum acceptable value according to local conditions.
    Output:
        outoflimsFlag (pandas.core.series.Series): Flagged series.
    '''
    inadequate = (series > max_envlim) | (series < min_envlim)
    outoflimsFlag = series.mask(inadequate==True, other=-9999)
    outoflimsFlag = outoflimsFlag.mask(outoflimsFlag!=-9999, other=1)
    outoflimsFlag = outoflimsFlag.mask(outoflimsFlag==-9999, other=4)
    return outoflimsFlag

def latlon(x):
    '''
        Converts type of latitude and longitude coordenates pattern. 
        e.g: longitude '-4210.6420' to '-42.177367'
    '''
    try:
        coord = (int(x[1:3]) + float(x[3:])/60.)*-1
    except:
        coord = np.nan
    return coord

def distanceDispl(df):
    '''
        This function calculates glider displacement distance between 
        recording times. 
    '''
    lon = pd.DataFrame(df['m_gps_lon'].dropna().values,np.arange(
                                        len(df['m_gps_lon'].dropna())))
    lat = pd.DataFrame(df['m_gps_lat'].dropna().values,np.arange(
                                        len(df['m_gps_lat'].dropna())))
    #Calculate distance between first gps data communication and last.
    dist_glider = distance.distance((lon.loc[lon.index[0],:].values, 
                                    lat.loc[lat.index[0],:].values),
                                    (lon.loc[lon.index[-1],:].values, 
                                    lat.loc[lat.index[-1],:].values)).km
    return dist_glider

##################### Choose folder with binary files and cache file ##########################
# Directory must have those items: dbd2asc.exe, folder named "cache" with cache files and 
# *.tbd and *.sbd files. DON'T USE IT WITH SERVER DIRECTORY !!!!!!!!
# Copy those files what you want to run for your machine before use it.
os.chdir(r'C:\Users\nathalia.vidal\OneDrive - OceanPact Serviços Marítimos S.A\Documentos\glider\REM\dados')
################################## Import Data ################################################
list_extension_files = ['sbd','tbd']
print('\t\t Welcome to glider slocum project by Ocean Pact!')
print('\nThis project was developed by Piero Morandim Maduro of Oceanografia Management - OceanPact GEO')
for file_extension in list_extension_files:
    print('\n\t\t It\'s running *.%s file estension.' %file_extension)
    #Select file extention - .tbd or .sbd
    binary_files = glob.glob('*.'+file_extension)
    df_final, dist_glider_final, amar_on_final,speed = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),pd.DataFrame()
    df_data_per_mission = pd.DataFrame(columns=['first day','last day'])
    #speed =  pd.DataFrame(columns=['start of the dive','end of the dive','average speeds'])
    for bin_file in binary_files:
    #filetbd = 'unit_975-2022-102-2-0.sbd'
        extension_file = bin_file.split('.')[-1]
        file_condition = bin_file.split('.')[0]
        # Only file names finished with 0 are accepetables data measured. 
        ## Another ones are "baby files"
        if file_condition.split('-')[-1] == '0':
            file = extension_file+'_'+bin_file.split('.')[0]+'.txt'
            #Convert binary into ASCII file
            #os.system('dbd2asc '+bin_file+' > '+file)
            #Get ASCII file 
            folder = os.getcwd()
            datfile = os.path.join(folder,file)
            #get metadata of ASCII file
            (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = \
                                                            os.stat(datfile)
            if size>=2000:
                #Read ASCII file
                with open(datfile,'r') as f:
                    aux = f.readlines()
                #Carrying out dataframe
                data_aux = aux[17:]
                data_aux = "".join(map(lambda x: x.strip() + '\n',data_aux)) 
                df = pd.read_csv(StringIO(data_aux), sep=" ", header=None)
                #Rename columns 
                cols = aux[14].strip().split(' ')
                df.columns = cols
                #Get units of variables
                var_units = aux[15].strip().split(' ')
                #Set index with timestamp
                if extension_file == 'tbd':
                    sci = 'sci_'
                    #Carrying out density - RULES: converts cond -> s/m to  
                    # ms/cm & pressure -> bar to dbar & temp °C
                    df[sci+'m_water_salt'] = gsw.conversions.SP_from_C(
                                                        df[sci+'water_cond']*10,
                                                        df[sci+'water_temp'], 
                                                        df[sci+'water_pressure']*10)
                else:
                    sci = ''
                #Set index
                df.index = df[sci+'m_present_time'].apply(dt.datetime.fromtimestamp)
                #Saving first and last datetime of mission
                d = {'first day': [df.index[0]], 'last day': [df.index[-1]]}
                df_data_per_mission = df_data_per_mission.append(pd.DataFrame(data=d),ignore_index=True)
                # Concatenating data of all files
                df_final = pd.concat([df_final,df], axis=0)
                if extension_file == 'sbd':
                    # Converting latitude & longitude unit
                    df['m_gps_lat'] = df['m_gps_lat'].astype(str)
                    df['m_gps_lon'] = df['m_gps_lon'].astype(str)
                    df['m_gps_lat'] = df['m_gps_lat'].apply(latlon)
                    df['m_gps_lon'] = df['m_gps_lon'].apply(latlon)
                    # Calculating distance between mission
                    dist_glider = pd.Series(data=distanceDispl(df),
                                        index=[df['m_gps_lat'].dropna().index[-1]])
                    dist_glider_final = pd.concat([dist_glider_final,dist_glider],axis=0)
                    # Checking if the Amar was collecting 
                    if any(df['c_amar_on']==0):
                        # getting the speed only on dives(positive values) when the amar it on
                        vel = df['m_depth_rate_avg_final'][(df['m_depth_rate_avg_final']>0)]
                        #create a df with the average of speeds for each dive
                        d_vel = {'start of the dive': [vel.index[0]], 'end of the dive': [vel.index[-1]],
                        'average speeds':[vel.mean()],'speed min':[vel.min()],'speed max':[vel.max()]}
                        speed = speed.append(pd.DataFrame(data=d_vel),ignore_index=True)
                        speed_above = speed['average speeds'][(speed['average speeds']>0.109)]
                        print(speed)
                        amar_on = pd.Series(data=0,
                                            index=[df['c_amar_on'].dropna().index[-1]])
                        
                    else:
                        amar_on = pd.Series(data=-1,
                                        index=[df['c_amar_on'].dropna().index[-1]])
                    amar_on_final = pd.concat([amar_on_final,amar_on],axis=0)

    #########################################################################################
    #df_final = df_final.loc[df_final.index[0]:xxxxx,:] # trocar xxxxx pela data final
    if extension_file == 'sbd':
        # Calculating number of missions the Amar was collecting
        mask = (amar_on_final[0]==0).values
        num_mission_amar = amar_on_final[0].where(mask).dropna()
        print('\n\n#######################################################################')
        print('#######################################################################')
        print('\nDistancia percorrida no intervalo selecionado (km): %s' %str(
                np.sum(dist_glider_final).values[0]))
        print('Numero de missões com amar ligado: %s de %s' %(
                str(len(num_mission_amar)),
                str(len(amar_on_final))))
        print('\nNumero de missões com velocidade média acima de 0.1 m/s: %s' %str(
                len(speed_above)))
        print('\n#######################################################################')
        print('#######################################################################')
        ################################ Ploting Map ########################################
        llcrnrlon, llcrnrlat = -49, -28
        urcrnrlon, urcrnrlat = -39, -22.5
        linewidth, fontsize = 0.5, 8
        plt.figure(figsize=(10,12))
        gliderMap = Basemap(projection='merc', llcrnrlon=llcrnrlon,
                            llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
                            urcrnrlat=urcrnrlat, resolution='h') #resolution='h'
        gliderMap.drawparallels(np.arange(int(llcrnrlat), int(urcrnrlat), 1),
                                labels=[1,0,0,0], linewidth=linewidth,
                                fontsize=fontsize, fontweight='normal')
        gliderMap.drawmeridians(np.arange(int(llcrnrlon), int(urcrnrlon), 1),
                                labels=[0,0,0,1], linewidth=linewidth,
                                fontsize=fontsize, fontweight='normal')
        gliderMap.fillcontinents(color='oldlace',lake_color='aliceblue')
        gliderMap.drawcoastlines(linewidth=linewidth)
        gliderMap.drawmapboundary(fill_color='aliceblue')
        # Position of scale
        gliderMap.drawmapscale(-39.8, -27.5, -39.8, -27.5, 100,
                            fontsize=6.5, barstyle='fancy')
        # Plot Polygon of Bacia de Santos
        santos = pd.read_csv('bacia_santos.csv')
        lat_santos_utm, lon_santos_utm = gliderMap(santos['lon'].tolist(), santos['lat'].tolist())
        gliderMap.plot(lat_santos_utm, lon_santos_utm, color="dimgrey", latlon=False, 
                    linewidth=1, label='Bacia de Santos')
        # Plot Polygon of AGBS
        agbs = pd.read_csv('agbs.csv')
        lon_agbs_utm, lat_AGBS_utm = gliderMap(agbs['lon'].tolist(), agbs['lat'].tolist())
        gliderMap.plot(lon_agbs_utm, lat_AGBS_utm, color='C0',linewidth=0.5)
        plt.text(lon_agbs_utm[0]*0.95, lat_AGBS_utm[0]*1.025, 'AGBS',fontsize=6.5,
                fontweight='bold', ha='left', va='center', color='C0')
        # Plot polygon selected (P03)
        polygon = 'PO3'
        df_polygon = pd.read_csv('coordenadas_poligonos.csv')
        df_polygon_selected = df_polygon.where(df_polygon['POLIGONO']== polygon).dropna()
        lon_utm, lat_utm = gliderMap(df_polygon_selected['LON'].tolist(), 
                                    df_polygon_selected['LAT'].tolist())
        gliderMap.plot(lon_utm, lat_utm, marker='o', markersize=2.5, color="yellowgreen",
                    latlon=False, linewidth=1, label=polygon)
        plt.text(lon_utm[0]*0.95, lat_utm[0]*1.025, polygon, fontsize=6.5,
                fontweight='bold', ha='left', va='center', color="yellowgreen")
        # Plot glider
        # Converting latitude & longitude unit
        df_final['m_gps_lat'] = df_final['m_gps_lat'].astype(str)
        df_final['m_gps_lon'] = df_final['m_gps_lon'].astype(str)
        df_final['m_gps_lat'] = df_final['m_gps_lat'].apply(latlon)
        df_final['m_gps_lon'] = df_final['m_gps_lon'].apply(latlon)
        lon_glider_utm, lat_glider_utm = gliderMap(df_final['m_gps_lon'].dropna().tolist(), 
                                                df_final['m_gps_lat'].dropna().tolist())
        gliderMap.plot(lon_glider_utm, lat_glider_utm, marker='o', markersize=1.5, color="C1",
                    latlon=False, linewidth=1.2, label='Glider')
        gliderMap.plot(lon_glider_utm[0], lat_glider_utm[0], marker='o', markersize=3.5, color="C2",
                    latlon=False, linewidth=1.2, label='Glider')
        gliderMap.plot(lon_glider_utm[-1], lat_glider_utm[-1], marker='o', markersize=3.5, color="C3",
                    latlon=False, linewidth=1.2, label='Glider')
        plt.savefig('Mapa_deslocamento_Glider_975.png', bbox_inches='tight')
        plt.close('all')
    if extension_file == 'tbd':
        ######################################### QC #########################################################
        temp_lims = (1,30); salt_lims = (33,38)
        df_final['temp_outOfLimsFlag'] = outOfEnvLims(df_final[sci+'water_temp'], temp_lims[0], temp_lims[1])
        df_final['salt_outOfLimsFlag'] = outOfEnvLims(df_final[sci+'m_water_salt'], salt_lims[0], salt_lims[1])                                                                                                                                                      
        df_final = df_final.mask(df_final['temp_outOfLimsFlag']==4, other=np.nan)
        df_final = df_final.mask(df_final['salt_outOfLimsFlag']==4, other=np.nan)
        #####################################################################################################
        ################################ Ploting Glider CTD profiles ########################################
        fontsize = 12
        color_temp = 'r'
        color_salt = 'b'
        # Plot temperature:
        fig, ax1 = plt.subplots(figsize=(9.2,12))
        # Define color to plot temperature per step time
        tempinterval = np.linspace(.2, .6, len(df_data_per_mission.index))
        tempcolors = [plt.cm.Reds(i) for i in tempinterval]
        for time_mission, color in enumerate(tempcolors):
            ax1.plot(df_final.loc[
                        df_data_per_mission.iloc[time_mission]['first day']:
                        df_data_per_mission.iloc[time_mission]['last day'], sci+'water_temp'],
                    -df_final.loc[
                            df_data_per_mission.iloc[time_mission]['first day']:
                            df_data_per_mission.iloc[time_mission]['last day'], sci+'water_pressure']*10, 
                    color=color,alpha=.3, lw=.5)
        ax1.set_xlabel('Temperatura [°C]', color=color_temp, fontsize=fontsize)
        ax1.set_ylabel('Profundidade [m]', fontsize=fontsize)
        ax1.set_title('Glider n/s 975')
        ax1.set_xlim(temp_lims)
        ax1.tick_params(axis='x', labelcolor=color_temp)
        ax1.grid(linestyle='dotted', zorder=0)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig('Perfil_Temperatura_Glider_975.png', bbox_inches='tight')
        plt.close('all')
        # Plot salinity:
        fig2, ax2 = plt.subplots(figsize=(9.2,12))
        saltinterval = np.linspace(.2, .6, len(df_data_per_mission.index))
        saltcolors = [plt.cm.Blues(i) for i in saltinterval]
        for time_mission, color in enumerate(saltcolors):
            ax2.plot(df_final.loc[
                        df_data_per_mission.iloc[time_mission]['first day']:
                        df_data_per_mission.iloc[time_mission]['last day'], sci+'m_water_salt'],
                    -df_final.loc[
                            df_data_per_mission.iloc[time_mission]['first day']:
                            df_data_per_mission.iloc[time_mission]['last day'], sci+'water_pressure']*10, 
                    color=color,alpha=.3, lw=.5)
        ax2.set_xlabel(u'Salinidade [PSU]', color=color_salt, fontsize=fontsize)
        ax2.set_title('Glider n/s 975')
        ax2.set_xlim(salt_lims)
        ax2.tick_params(axis='x', labelcolor=color_salt)
        ax2.grid(linestyle='dotted', zorder=0)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        #plt.ylim([-max(df_final[sci+'water_pressure']*10), 0])
        plt.savefig('Perfil_Salinidade_Glider_975.png', bbox_inches='tight')
        plt.close('all')
    print('\n\t\t The tasks with *.%s file estension was finished!' %file_extension)

