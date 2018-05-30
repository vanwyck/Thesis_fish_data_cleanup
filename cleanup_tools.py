# -*- coding: utf-8 -*-
"""
Created on Wed May 30 13:29:10 2018

Contains functions:
    - angle_filter
    - calculate_max_time
    - clean_toa_data
    - CTCRW
    - find_delay
    - iterate_over_passings
    - KalmanFilter
    - KFImplementationCTCRW
    - max_swimming_speed
    - plot_fish_tracks
    - prepare_tag_data
    - RTSSImplementationCTCRW
    - RTSSmoother
    - speed_filter
    - split_passings
    - to_reference_coord
    - two_stage_speed_filter


@author: Thoma
"""

import pandas as pd
import numpy as np
from tools_analysis import calc_soundspeed
from pyproj import Proj, transform
import matplotlib.pyplot as plt

def angle_filter(data, max_angle = 165,xcolname = 'X',ycolname = 'Y'):
    '''
    This function will remove positions with unrealisticly high speeds as a form of validation gating before the Kalman Filter.
    The method used is analogous to the one defined by McConnel in 1992
    
    Inputs
    ------
    data : dataframe 
        The dataset that requires cleaning. Must contain at least the columns 'X','Y' and 'DATETIME'
    max_angle : int, optional
        Maximum turning angle expected to come from actual fish positions, expressed in degrees
    
    Returns
    -------
    cleaned_data : dataframe
    '''
    #calculate distance between steps
    dist=np.sqrt((data[xcolname]-data[xcolname].shift(1))**2+(data[ycolname]-data[ycolname].shift(1))**2)
    double_dist = np.sqrt((data[xcolname].shift(-1)-data[xcolname].shift(1))**2+(data[ycolname].shift(-1)-data[ycolname].shift(1))**2)
    # use Cosine rule to determine angle formed by points befor and after observation 
    angle = 180 - np.abs(180/np.pi*np.arccos((dist**2 + dist.shift(-1)**2 - (double_dist)**2)/(2*dist*dist.shift(-1))))
    # remove speeds higher than maximum    
    cleaned_data = data[(angle < max_angle)  | angle.isna() ].copy()
    return cleaned_data


def calculate_max_time(input_data,hydro,all_temp_data,tollerance =0.01,S = 0.4,z = 1):
    '''
    This function calculates the time an acoustic signal should take to travel between receivers.
    
    Parameters
    ----------
    input_data: DataFrame
        The raw synchronised data of one selected tag
    hydro : DataFrame
        The positions of the receivers
    all_temp_data : DataFrame
        The temperature data of the whole deployment time. 
        
    Returns
    -------
    tag_data : DataFrame
        Reorganised dataframe, sorted by TOA. 
        Contains two new colums: 'max_time' and 'soundspeed', 
        containing the maximumt travel time from the previous receiver and the speed of sound respectively
    '''
    tag_data = input_data.copy()
    # reorganise the data and sort by TOA
    tag_data = tag_data.reset_index(level = 0,drop = False).reset_index(level = 0, drop = True)
    tag_data = tag_data.loc[:,['synced_time','level_0']]
    tag_data = tag_data.sort_values(by='synced_time')
    #read in receiver positions
    tag_data = tag_data.join(hydro[['X','Y']],on = tag_data.level_0) 
    #calculate the distance between the receiving tags
    dist=np.sqrt((tag_data.X-tag_data.X.shift(1))**2+(tag_data.Y-tag_data.Y.shift(1))**2) 
    #interpolate the soundspeed data
    temp = all_temp_data.append(pd.DataFrame(None, index = tag_data.synced_time)).sort_index().interpolate()
    temp = temp[temp.index.isin(tag_data.synced_time)].values
    #calculate soundspeed
    tag_data['soundspeed']= calc_soundspeed(T=temp,S = S,z= z)
    #calculate the maximal time a signal should travel
    tag_data['max_time'] = pd.to_timedelta(dist/tag_data.soundspeed + tollerance, unit = 's' )
    tag_data= tag_data.drop(columns = ['X','Y'])
    return tag_data

def clean_toa_data(end_data,min_delay):
    '''
    Cleans up the TOA matrix, created with prepare_tag_data.
    When observations follow eachother faster than the minimum known time delay,
    the obeservation with the least receivers is removed.
    
    Parameters
    ----------
    end_data : DataFrame
        The TOA matrix. Output of the prepare_tag_data.
    min_delay : int or float
        The minimum time delay for the used transmitter, defined by the manufacturer
    
    Returns
    -------
    end_data_cleaned : DataFrame
        Cleaned dataframe, with unreliable observations removed
    '''
    #remove soundspeed and synced_time column for analysis
    end_data_cleaned = end_data.iloc[:,:-2].copy()
    end_data_cleaned['t_diff'] = end_data_cleaned.mean(axis = 1).diff().values
    end_data_cleaned['receiver_amount'] = end_data_cleaned.count(axis = 1).values            
    #check if second point of impossible interval is the wrong one
    #define second point of impossible interval
    #see if last point was picked up by more receivers
    end_data_cleaned['second_wrong'] = ((end_data_cleaned['receiver_amount'].shift(1)>end_data_cleaned.receiver_amount)
                                        & (end_data_cleaned['t_diff']<min_delay)) 
    #check if first point of impossible interval is the wrong one
    #define first point of impossible interval
    #see if next point was picked up by more receivers
    end_data_cleaned['first_wrong'] = ((end_data_cleaned['receiver_amount'].shift(-1)>end_data_cleaned.receiver_amount)
                                        & (end_data_cleaned['t_diff'].shift(-1)<min_delay)) 
    end_data_cleaned['true_error'] = end_data_cleaned.first_wrong| end_data_cleaned.second_wrong
    end_data_cleaned = end_data[~end_data_cleaned.true_error].copy()
    return end_data_cleaned

def CTCRW(m_k_old, P_k_old,delta_t,beta, sigma):
    A_k = np.eye(4)
    A_k[(0,1),(2,3)] = (1-np.exp(-beta*delta_t))/beta
    A_k[(2,3),(2,3)] = np.exp(-beta*delta_t)
    V_xi = sigma**2 /beta**2*(delta_t-2/beta*(1-np.exp(-beta*delta_t))+1/(2*beta)*(1-np.exp(-2*beta*delta_t)))
    V_zeta = sigma**2*(1-np.exp(-2*beta*delta_t))/(2*beta) 
    C = sigma**2/ (2*beta**2)*(1-2*np.exp(-beta*delta_t)+np.exp(-2*beta*delta_t))
    Q = np.array([[V_xi, 0,C,0],
                  [0,V_xi,0,C],
                  [C,0,V_zeta,0],
                  [0,C,0,V_zeta]],np.float)
    m_k_hat = np.dot(A_k,m_k_old)
    P_k_hat = np.dot(np.dot(A_k,P_k_old), A_k.transpose() ) + Q #+ np.eye(4)/8 
    return m_k_hat,P_k_hat

def find_delay(ID):
    ID = np.int(ID)
    if ((ID >= 100) & (ID <= 109)| (ID == 255)):
        min_delay = 17 + 3.6
        max_delay = 33 + 3.6
    elif ((ID >= 38725) & (ID <=38740)):
        min_delay = 17 + 3.2
        max_delay = 33 + 3.2
    elif ((ID >= 34429) & (ID <= 34453)):
        min_delay = 40 + 3.2
        max_delay = 80 + 3.2
    elif ((ID >= 34517 & ID <=34536)|(ID >= 64485 & ID <=64513)|(ID == 53429)):
        min_delay = 15 + 3.2
        max_delay = 30 + 3.2
    elif (ID == 16200):
        min_delay = 10+3.2
        max_delay = 10+3.2
    else:
        print('Invalid ID')    
    return (min_delay,max_delay)

def iterate_over_passings(VPS_data_fish,beta,sigma,xcolname = 'X', ycolname = 'Y', tcolname = 'DATETIME',errorcol = 'HPE'):
    data_out = pd.DataFrame()
    for i in VPS_data_fish.pas_nr.unique():
        track = VPS_data_fish[VPS_data_fish.pas_nr == i].reset_index(drop = True)
        if len(track)>3:
            #run filter
            track,m_k,P_k = KFImplementationCTCRW(track,beta,sigma ,errorcol= errorcol)
            track = RTSSImplementationCTCRW(track,m_k,P_k,beta,sigma)
            data_out  = pd.concat([data_out,track])
    return data_out

def KalmanFilter(m_k_hat, P_k_hat, H, y_k, R_k):
    v_k = y_k - np.dot(H ,m_k_hat)
    S_k = np.dot(np.dot(H,P_k_hat), H.transpose() ) + R_k
    K_k = np.dot(np.dot(P_k_hat,H.transpose()),np.linalg.inv(S_k))
    m_k = m_k_hat + np.dot(K_k,v_k)
    P_k = P_k_hat - np.dot(np.dot(K_k,S_k),K_k.transpose())
    return m_k , P_k

def KFImplementationCTCRW(data,beta,sigma, xcolname = 'X', ycolname = 'Y', tcolname = 'DATETIME',errorcol = 'HPE'):
    ''' Optional inputs 'colnam' contain strings with the names of the colums where the x position, 
    y position and transmission time can be found.
    '''
    data_out = data.copy()
    # first point can't be filtered: copy from original
    data_out.loc[0,'X_filtered']= data.loc[0,xcolname]
    data_out.loc[0,'Y_filtered']= data.loc[0,ycolname]
    data_out.loc[0,'filtered_error']= data.loc[0,errorcol]
    H = np.hstack((np.eye(2) , np.zeros((2,2))))
    R_k = np.zeros((2,2))
    P_k = np.zeros((4,4))
    #first 2 points are omitted here as it is needed to calculate the initial speed
    P_k[(0,1),(0,1)] = data.loc[0,errorcol]**2
    #initialize m_k based on first 2 points
    delta_t = data[tcolname].diff()/pd.Timedelta(seconds = 1)
    v_X =  (data[xcolname][1] - data[xcolname][0])/delta_t[1]
    v_Y =  (data[ycolname][1] - data[ycolname][0])/delta_t[1]
    m_k = np.array([data[xcolname][1],data[ycolname][1],v_X,v_Y]) # = x_(k-1)
    m_k_vector = m_k
    P_k_vector = P_k
    #run the filtering algorithm
    for i in range(1,len(data)):
        y_k = data.loc[i,[xcolname,ycolname]].values
        if len(errorcol) ==2:
            R_k[0,0] = data.loc[i,errorcol[0]]**2
            R_k[1,1] = data.loc[i,errorcol[1]]**2
        else:
            R_k[(0,1),(0,1)] = data.loc[i,errorcol]**2
        m_k_hat, P_k_hat = CTCRW(m_k,P_k,delta_t[i],beta,sigma) 
        m_k, P_k = KalmanFilter(m_k_hat, P_k_hat, H, y_k, R_k)
        data_out.loc[i,'X_filtered'] = m_k[0]
        data_out.loc[i,'Y_filtered'] = m_k[1]
        data_out.loc[i,'filtered_error'] = np.sqrt(P_k[0,0])
        m_k_vector = np.vstack((m_k_vector,m_k))
        P_k_vector = np.dstack((P_k_vector,P_k))
    return data_out, m_k_vector,P_k_vector

def max_swimming_speed(ID,length_data):
    ID = np.int(ID)
    body_length = length_data.Length_mm[length_data.ID == ID].item()/1000   #mm to m
    if (ID >= 100 & ID <= 109)|(ID >= 38725 & ID <=38740):
        max_speed = body_length * 7.5  
    else:     
        max_speed = body_length *8
    return max_speed

def plot_fish_tracks(ID,path,hydro,wall_data, pas_nr = 'All'): 
    ID = str(ID)
    plt.rcParams.update({'font.size': 14})
    fish_data = pd.read_csv(''.join((path,'Example_data/VPS_track_cleaned_',ID,'.csv')))
    YAPS_data = pd.read_csv(''.join((path,"Example_data/YAPS_track_cleaned_",str(ID),".csv")))
    passings = pd.concat((fish_data.pas_nr,YAPS_data.pas_nr)).unique()
    if (pas_nr != 'All'):
        passings = passings[passings == pas_nr]
    for i,pas in enumerate(passings):
        filtered_track = fish_data[fish_data.pas_nr == pas]
        YAPS_track = YAPS_data[YAPS_data.pas_nr == pas]
        YAPS_track = YAPS_data[YAPS_data.pas_nr == pas]
        fig = plt.figure(figsize=(16,4))
        filtered_track_temp = filtered_track[filtered_track.X_filtered.notna()]
        ax1 = fig.add_subplot(1,3,1)
        ax1.plot(wall_data.X,wall_data.Y,color = 'black',linewidth=1,label = 'Canal wall')
        ax1.scatter(hydro['X'],hydro['Y'],color = 'red',label = 'Hydrophones')
        ax1.plot(filtered_track['X'],filtered_track['Y'],linestyle = '--',linewidth = 2, marker = 'o',
                         markersize = 4,color = 'mediumseagreen',label = 'Raw VPS track')
        ax1.plot(filtered_track_temp['X_filtered'],filtered_track_temp['Y_filtered'],linestyle = '-',linewidth = 1.5, marker = 'o',
                         markersize = 4,color = 'royalblue',label = 'Cleaned track')
        ax1.set_title('VPS + Three-stage + KF')
        ax1.axis('equal')
        ax1.set_ylim([-60,200])
        ax1.set_xlim([-20,300])
        ax1.set_xlabel( 'Horizontal postion (m)')
        ax1.set_ylabel('Vertical position (m)')
        ax2 = fig.add_subplot(1,3,2)
        ax2.plot(wall_data.X,wall_data.Y,color = 'black',linewidth=1)
        ax2.scatter(hydro['X'],hydro['Y'],color = 'red')
        ax2.plot(filtered_track['X'],filtered_track['Y'],linestyle = '--',linewidth = 2, marker = 'o',
                         markersize = 4,color = 'mediumseagreen')
        ax2.plot(YAPS_track['X'],YAPS_track['Y'],linestyle = '-',linewidth = 1.5, marker = 'o',
                         markersize = 4,color = 'royalblue')
        ax2.set_title('YAPS-BI')
        ax2.axis('equal')
        ax2.set_ylim([-60,200])
        ax2.set_xlim([-20,300])
        ax2.set_yticklabels([],[])
        ax2.set_xlabel( 'Horizontal postion (m)')
        ax3 = fig.add_subplot(1,3,3)
        ax3.plot(wall_data.X,wall_data.Y,color = 'black',linewidth=1)
        ax3.scatter(hydro['X'],hydro['Y'],color = 'red')
        ax3.plot(filtered_track['X'],filtered_track['Y'],linestyle = '--',linewidth = 2, marker = 'o',
                         markersize = 4,color = 'mediumseagreen')
        ax3.plot(YAPS_track['X_smoothed'],YAPS_track['Y_smoothed'],linestyle = '-',linewidth = 1.5, marker = 'o',
                         markersize = 4,color = 'royalblue')
        ax3.set_title('YAPS-BI + KF + RTSS')
        ax3.axis('equal')
        ax3.set_ylim([-60,200])
        ax3.set_xlim([-20,300])
        ax3.set_yticklabels([],[])
        ax3.set_xlabel( 'Horizontal postion (m)')
        handles,labels = ax1.get_legend_handles_labels()
        fig.legend(handles,labels, loc=8,ncol=4,bbox_to_anchor=(0.45, 0.97))
        fig.subplots_adjust(wspace=0.05) 
        plt.show()

def prepare_tag_data(input_data,pas_tol = 2):
    """
    Function to rearrange detections to dataframe with 1 observation per row.
    Observations are split up in different passings if time between observations is too long
    
    Inputs
    ------
    tag_data : DataFrame 
        The output of the 'calculate_max_time' function
        Contains at least the receivers at index level zero and columns 'synced_time' and 'max time'
    pas_tol : Float
        Max minutes between observations before a new passing is started and the track is split up
    
    Returns
    -------
    end_data : DataFrame
         TOA matrix usable by YAPS and TDOA positioning
         Columns are the receivers, rows are the observations, split up into passings
         Also contains a soundspeed and synced_time column
    """
    tag_data = input_data.copy()
    tag_data['time_diff'] = tag_data.synced_time.diff()/pd.Timedelta(seconds=1)
    tag_data['sec_since_start'] = tag_data['time_diff'].cumsum().fillna(0)
    # make gaps when time since previous is > max_time
    gaps = tag_data.synced_time.diff() >  tag_data.max_time
    # cumsum of falses and trues creates groups
    tag_data['groups_obs'] = gaps.cumsum() 
    # idem for tracks
    gaps2 = tag_data.synced_time.diff() > pd.Timedelta(minutes = pas_tol)
    tag_data['groups_pas'] = gaps2.cumsum()
    # save soundspeed for YAPS model and synced time for splitting in tracks
    soundspeed = tag_data.set_index(['groups_obs'])['soundspeed'].groupby('groups_obs').mean()
    synced_time = tag_data.set_index(['groups_obs'])['synced_time'].groupby('groups_obs').first()
    # reshape the resulting dataframe
    end_data = tag_data.set_index(['groups_pas','groups_obs','level_0'])['sec_since_start'].unstack()
    # put back soundspeed and synced time
    end_data['soundspeed'] = soundspeed.values
    end_data['synced_time']= synced_time.values
    return end_data

def RTSSImplementationCTCRW(data,m_k_vector,P_k_vector,beta,sigma, tcolname = 'DATETIME'):
    '''Requires a run of the regular KF implementation algortihm first. 
    The output of that function can then serve as the input of this one.
    
    '''    
    data_out = data.copy()
    #last filtered point can't be smoothed: copy
    data_out.loc[len(data)-1, 'X_smoothed'] = data.loc[len(data)-1, 'X_filtered']
    data_out.loc[len(data)-1, 'Y_smoothed'] = data.loc[len(data)-1, 'Y_filtered']
    data_out.loc[len(data)-1, 'smoothed_error'] = data.loc[len(data)-1, 'filtered_error']
    delta_t = data_out[tcolname].diff()/pd.Timedelta(seconds = 1)
    A_k = np.eye(4)
    P_k_s = P_k_vector[:,:,-1]
    m_k_s = m_k_vector[-1]
    for i in reversed(range(0,(len(data)-1))):
        #rebuild the Pk as created by the KF algorithm
        P_k = P_k_vector[:,:,i]
        m_k = m_k_vector[i]
        # predict m_(k+1) and P_(k+1) 
        m_k_hat_next, P_k_hat_next = CTCRW(m_k,P_k,delta_t[i+1],beta,sigma) 
        A_k[(0,1),(2,3)] = (1-np.exp(-beta*delta_t[i+1]))/beta
        A_k[(2,3),(2,3)] = np.exp(-beta*delta_t[i+1])
        m_k_s, P_k_s = RTSSmoother(m_k,m_k_hat_next, m_k_s, P_k, P_k_hat_next, P_k_s, A_k)
        data_out.loc[i,'X_smoothed'] = m_k_s[0]
        data_out.loc[i,'Y_smoothed'] = m_k_s[1]
        data_out.loc[i,'smoothed_error'] = np.sqrt(P_k_s[0,0])
    return data_out

def RTSSmoother(m_k, m_k_hat_next, m_k_s_next, P_k, P_k_hat_next, P_k_s_next, A_k):
    G_k = np.dot(np.dot(P_k, A_k.transpose()),np.linalg.inv(P_k_hat_next))
    m_k_s = m_k + np.dot(G_k,(m_k_s_next-m_k_hat_next))
    P_k_s = P_k + np.dot(np.dot(G_k,(P_k_s_next-P_k_hat_next)),G_k.transpose())
    return m_k_s, P_k_s

def speed_filter(data, max_speed = 'infer', points_used = 2,xcolname = 'X',ycolname = 'Y' ):
    '''
    This function will remove positions with unrealisticly high speeds as a form of validation gating before the Kalman Filter.
    The method used is analogous to the one defined by McConnel in 1992
    
    Inputs
    ------
    data : dataframe 
        The dataset that requires cleaning. Must contain at least the columns 'X','Y' and 'DATETIME'
    max_speed : int of str
        If the maximum swimmingspeed for the species is known, give as input. 
        If not, 'infer' should be given. Then the top 10% speeds are removed
    
    Returns
    -------
    cleaned_data : dataframe
    '''
    if points_used == 2:
        #calculate swimming speed between steps
        dist=np.sqrt((data[xcolname]-data[xcolname].shift(1))**2+(data[ycolname]-data[ycolname].shift(1))**2)
        delta_t = data.DATETIME.diff()/pd.Timedelta(seconds = 1)
        speed = dist/delta_t
        # calculate the local average speed as defined by McConnell (1992) 
        V = np.sqrt(.5*(speed.shift(1)**2+speed.shift(-1)**2))
    elif points_used == 4:
        dist=np.sqrt((data[xcolname]-data[xcolname].shift(1))**2+(data[ycolname]-data[ycolname].shift(1))**2)
        delta_t = data.DATETIME.diff()/pd.Timedelta(seconds = 1)
        speed = dist/delta_t
        double_dist = np.sqrt((data.X-data.X.shift(2))**2+(data.Y-data.Y.shift(2))**2)
        double_speed = double_dist/(delta_t+ delta_t.shift(1))
        V = np.sqrt(.25*(speed.shift(1)**2+speed.shift(-1)**2+double_speed**2 + double_speed.shift(-2)**2))
    if max_speed == 'infer':
        max_speed = speed.quantile(.95)
    # remove speeds higher than maximum    
    cleaned_data = data[(V < max_speed)  | V.isna() ].copy()
    return cleaned_data

def split_passings(end_data, min_obs):
    '''
    Function splits the cleaned TOA data from a fish track into passings of uninterrupted datapoints, usable by YAPS.
    
    Inputs
    ------
    end_data_cleaned: dataframe containing cleaned dataset; interferences removed, but empty rows not yet introduced
    min_obs: int,minimum observations needed to create a usable track for YAPS
    
    Returns
    -------
    end_data_split: dataframe, containing the different passings split up
    timestamps: dataframe containing time of first and last observation of each passing. 
                Used to link the YAPS tracks to VPS and filtered tracks
    '''
    # synced time is used to easiliy access the number of observations per passage
    passings = end_data.synced_time.groupby(level = 0).agg(['count','first','last'])
    #keep only long tracks
    passings = passings[passings['count'] >= min_obs]
    end_data_split = end_data[end_data.index.get_level_values(0).isin(passings.index.values)]
    #save start and end time of every passing to ling with VPS results
    timestamps = passings.loc[:,('first','last')]
    return end_data_split, timestamps

def to_reference_coord(input_data, input_columns, output_columns,x_0=647669.5941349089,y_0=5662691.951341231):
    """
    This function transforms locations expressed as latitude and longitude to our reference coordinates.
    This reference system expresses distances from the orgin, expressed in meters.
    The orgin is located at the bottom left of the hydrophone array (x_0 = x pos of S15, y0 is y pos of S12).
    
    Parameters
    ----------
    input_data : DataFrame
        The dataframe containing two columns containing latitudes and longitudes.
        These are transformed to the reference coordinates.
    input_columns : list 
        Names of the columns with longitude and latitude coordinates, in that order.
    output_columns : list 
        Names of the columns where the new coordinates should come
    x_0 : float, default the value for receiver S15
        Value used to redefine the origin to one close to the hydrphone array
    y_0 : float, default the value for receiver S12
        Value used to redefine the origin to one close to the hydrphone array
    
    Returns
    -------
    output_data : Dataframe
        The same as the input dataframe, with two aditional columns (names as defined by output_columns), containing the new coordinates.
    """
    coord = input_data.as_matrix(columns = input_columns)
    inProj = Proj(init='epsg:4326')   #epsg for gps data
    outProj = Proj(init='epsg:32631')   #epsg projection 32631 - wgs 84 / utm zone 31n 
    x1,y1 = coord[:,0], coord[:,1]
    x2,y2 = transform(inProj,outProj,x1,y1)
    x2 -= x_0 
    y2 -= y_0
    output_data = input_data.copy()
    output_data[output_columns[0]]= x2
    output_data[output_columns[1]] =y2
    return output_data

def two_stage_speed_filter(data, max_speed = 'infer',points_used = 2,xcolname = 'X',ycolname = 'Y'):
    '''
    This function will remove positions with unrealisticly high speeds as a form of validation gating before the Kalman Filter.
    The method used is analogous to the one defined by Austin (2003)
    
    Inputs
    ------
    data : dataframe 
        The dataset that requires cleaning. Must contain at least the columns 'X','Y' and 'DATETIME'
    max_speed : int of str
        If the maximum swimmingspeed for the species is known, give as input. 
        If not, 'infer' should be given. Then the top 10% speeds are removed
    
    Returns
    -------
    cleaned_data : dataframe
    '''
    # stage one: remove point if all speeds are too high
    if points_used == 2:
        dist=np.sqrt((data[xcolname]-data[xcolname].shift(1))**2+(data[ycolname]-data[ycolname].shift(1))**2)
        delta_t = data.DATETIME.diff()/pd.Timedelta(seconds = 1)
        speed = dist/delta_t
        if max_speed == 'infer':
            max_speed = speed.quantile(.95)
        cleaned_data = data[((speed.shift(1) < max_speed)| (speed.shift(-1) < max_speed)
                             | speed.isna()) ].copy()
    elif points_used == 4:
        dist=np.sqrt((data[xcolname]-data[xcolname].shift(1))**2+(data[ycolname]-data[ycolname].shift(1))**2)
        delta_t = data.DATETIME.diff()/pd.Timedelta(seconds = 1)
        speed = dist/delta_t
        if max_speed == 'infer':
            max_speed = speed.quantile(.95)
        double_dist = np.sqrt((data.X-data.X.shift(2))**2+(data.Y-data.Y.shift(2))**2)
        double_speed = double_dist/(delta_t+ delta_t.shift(1))
        cleaned_data = data[((speed.shift(1) < max_speed)| (speed.shift(-1) < max_speed) |(double_speed < max_speed)
                             | (double_speed.shift(-2) < max_speed)  | double_speed.isna()) ].copy()
    #stage 2: McConnel speed filter
    cleaned_data = speed_filter(cleaned_data,max_speed, points_used,xcolname = xcolname,ycolname= ycolname )
    return cleaned_data