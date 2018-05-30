# -*- coding: utf-8 -*-
"""
Created on Thu Dec 1 2016
Contains functions:
- refSyncCheck

@author: jennavergeynst
"""
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
from datetime import timedelta
from math import ceil, floor
from pyproj import Proj, transform
import os
import sys
import pprint
from geopy.distance import vincenty # more accurate than great_circle!
import pickle
from shapely.geometry import Polygon, Point
from scipy.interpolate import griddata


#sys.path.insert(0, "/Users/jennavergeynst/Documents/Ham/functions/")
#from paths import *

def add_gps_coo(vps_data, gps_pos):
    """
    Function to add the gps positions of fixed transmitters to the dataframe of vps positions of those transmitters
    
    Parameters
    ---------
    vps_data : dataframe with vps positions of fixed transmitters containing the column TRANSMITTER
    gps_pos : dataframe with gps positions of fixed transmitters containing the columns Longitude, Latitude, Station Name
    
    Returns
    -------
    vps_data : original dataframe with 2 extra columns LON_GPS and LAT_GPS
    """
    vps_data['LON_GPS'] = np.nan
    vps_data['LAT_GPS'] = np.nan
    vps_data = vps_data.reset_index()
    for fixed in gps_pos['Station Name']:
        vps_data['LON_GPS'][vps_data.TRANSMITTER==fixed] = gps_pos.Longitude[gps_pos['Station Name']==fixed].item()
        vps_data['LAT_GPS'][vps_data.TRANSMITTER==fixed] = gps_pos.Latitude[gps_pos['Station Name']==fixed].item()
        
    return vps_data

def add_perc_size(track_pos, fixed_pos, acc_goal):
    """

    Add for each position
    perc = the percentage of fixed tag positions with error <= accuracy goal
    size = the nb of fixed tags positions calculated with this URX group
    of the corresponding URX group.
    
    perc is to be used as color code
    size has to be >10 to use the position's classification
    
    Parameters
    ---------
    track_pos : positions of fish/test with at least column URX
    fixed_pos : positions of fixed tags with at least columns HPEm and URX
    acc_goal : maximum allowed error
        
    """
    
    URX_groups, good_performers, bad_performers = classify_URX_groups(
        fixed_tags=fixed_pos, acc_goal=acc_goal)
    track_pos['class_perc'] = list(
        track_pos.reset_index().apply(lambda row: find_perc_size(row, URX_groups)[0], axis=1))
    track_pos['class_groupsize'] = list(
        track_pos.reset_index().apply(lambda row: find_perc_size(row, URX_groups)[1], axis=1))

        
    return track_pos


def calcDisDetU (allData, shortID, cutoff = 1000000):
    """
    Function used to calculate variables on individual fish (eel) positioning data.
    Input is the short ID of the transmitter ('shortID'), the data and the HPE cutoff (default no cutoff)
    csv file is read by script
    Output is 
    - dataframe with 3 extra columns: DISTANCE, DETINT, U
    - list of period that each eel was detected
    - list of duration of each eel's period
    """

    tagData = allData[allData["TRANSMITTER"]==shortID].reset_index(drop=True)
    # read datetime column in datetime format
    tagData['DATETIME'] = pd.to_datetime(tagData['DATETIME'])
    tagData = tagData[tagData['HPE']<cutoff].reset_index(drop=True)
    
    #Calculate distance from last detection
    tagData['DISTANCE'] = distance(tagData['X'],tagData['Y'])
    
    #Calculate time since last detection
    tagData['DETINT'] = timeInterval(tagData['DATETIME'])

    #Calculate swim velocity since last detection
    tagData['U'] = tagData['DISTANCE'] / tagData['DETINT']
    
    #Calculate calibrated depth
    
    with open(ABIOTICS_PATH+'depth_cal_coeff.pkl', 'rb') as handle:
        depth_cal_coeff = pickle.load(handle)
    
    if shortID.startswith('10'):
        tagData['calibrated_depth'] = tagData['DEPTH']*depth_cal_coeff['slope']+depth_cal_coeff['intercept']
        
    else:
        tagData['calibrated_depth'] = np.nan
    
    #Calculate duration of stay
    length = len(tagData['DATETIME'])
    durStay = tagData['DATETIME'][length-1]-tagData['DATETIME'][0]
    
    #give first and last moment of detection
    period = [min(tagData['DATETIME']),max(tagData['DATETIME'])]
    
    # save the last point of detection
    lastline = tagData.iloc[-1, ]

    return tagData, durStay, period, lastline



def calc_RMS2D(vps_data, binby):
    """
    calculate the 2-dimensional RMS on vps-data
    
    Parameters
    ----------
    vps-data : dataframe with columns LON, LAT, LON_GPS, LAT_GPS, HPE
    binby : 'HPE' or 'HPE_log'
    
    Returns
    -------
    bins : dataframe with for each bin the average and number of HPE/HPE_log
    tobin : copy of original dataframe with extra column containing the bin for each datapoint
    """
    
    tobin = vps_data.copy()
    tobin['HPE_log'] = tobin['HPE'].apply(np.log)
    tobin['bin'] = pd.cut(tobin[binby], bins=np.arange(floor(min(tobin[binby])),ceil(max(tobin[binby])+1)))
    
    UTM_vps = tobin.apply(transform_wgs_to_utm, lon_col = 'LON', lat_col = 'LAT', axis = 1)
    UTM_vps.columns = ['x', 'y']
    UTM_gps = tobin.apply(transform_wgs_to_utm, lon_col = 'LON_GPS', lat_col = 'LAT_GPS', axis = 1)
    UTM_gps.columns = ['x', 'y']
    
    tobin['xe'] = abs(UTM_gps.x - UTM_vps.x)
    tobin['ye'] = abs(UTM_gps.y - UTM_vps.y)
    
    bins = tobin.groupby(by='bin')[binby].agg(['mean', 'count'])
    
    bins['xeSd'] = tobin.groupby(by='bin')['xe'].std()
    bins['yeSd'] = tobin.groupby(by='bin')['ye'].std()
    bins['RMS2D'] = 2*(bins['xeSd']**2+bins['yeSd']**2)**(0.5)
    
    bins = bins[bins['count']>3]
    
    return bins, tobin

def calc_soundspeed(T, S, z):
    """
    Equation of Mackenzie: http://asa.scitation.org/doi/10.1121/1.386919
    parameters
    ----------
    T = temperatuur in °C
    S = salinity in parts per thousand
    z = depth in meters
    
    returns
    ------
    soundspeed in m/s
    """

    a1 = 1448.96
    a2 = 4.591
    a3 = -5.304e-2
    a4 = 2.374e-4
    a5 = 1.34
    a6 = 1.63e-2
    a7 = 1.675e-7
    a8 = -1.025e-2
    a9 = -7.139e-13
    
    SS = a1 + a2*T + a3*T**2 + a4*T**3 + a5*(S-35) + a6*z + a7*z**2 + a8*T*(S-35) + a9*T*z**3
    return SS

def calc_velocity_in_dict(fish_positions, ID_list):
    
    """
    Function to calculate the swimming velocities after position filtering.
    
    Parameters
    ----------
    fish_positions = dataframe containing all fish positions for each transmitter
    ID_list = list with transmitter names
    
    Returns
    -------
    fish_dict = dictionnary with for each transmitter (key) a dataframe with swimming velocities
    The function also prints a summary of all velocities
    
    """
    fish_dict = {ID: fish_positions[fish_positions.TRANSMITTER==ID].reset_index() for ID in ID_list}
    for ID in ID_list:
        if len(fish_dict[ID])>1:
            fish_dict[ID]['distance'] = distance(fish_dict[ID]['X'],fish_dict[ID]['Y'])
            fish_dict[ID]['interval'] = timeInterval(fish_dict[ID]['DATETIME'])
            fish_dict[ID]['U'] = fish_dict[ID]['distance'] / fish_dict[ID]['interval']
    return pd.concat(fish_dict)


def check_passage(fishID_list, location, non_vps_detections):
    """
    Check if the fish in 'fishID_list' were detected by the receiver at 'location' ('DV', 'MS', 'NS', 'WKC', 'SA')
    
    Parameters
    ---------
    fishID_list : list of fish IDs
    location : 'DV', 'MS', 'NS', 'WKC' or 'SA'
    non_vps_detections : dataframe with vps detections at different non-vps-receivers, with columns ID and Date/Time
    
    Returns
    -------
    detected : set of detected IDs from 'fishID_list'
    non_detected : set of nondetected IDs from 'fishID_list'
    """
    # first set numeric keys to names
    non_vps_receivers = {'122321': 'DV', '122319': 'MS', '122331': 'NS', '122326': 'WKC', '122360': 'SA'}
    non_vps_detections = dict((non_vps_receivers[key], value) for (key, value) in non_vps_detections.items())
    
    all_detections = pd.DataFrame(non_vps_detections[location].ID.unique())
    fish_detections = detections[detections.isin(fishID_list)].dropna()
    fd = set([str(int(x)) for x in list(fish_detections[0])])
    detected = fd
    non_detected = set(fishID_list)-fd
    
    return detected, non_detected

def check_passage_time(fishID_list, location, non_vps_detections):
    """
    Check first detection, last detection and number of detections
    
    Parameters
    ---------
    fishID_list : list of fish IDs
    location : 'DV', 'MS', 'NS', 'WKC' or 'SA'
    non_vps_detections : dict with dataframes with vps detections at different non-vps-receivers, 
    with columns ID and Date/Time
    
    Returns
    -------
    summary : dataframe containing for each fish first, last detection time and nb of detections
    """
    
    
    sub = non_vps_detections[location][non_vps_detections[location].ID.isin(fishID_list)]
    first_d = sub.drop_duplicates(subset='ID', keep='first').set_index('ID').loc[:,['Date/Time']].rename(columns={'Date/Time':'First '+location})
    last_d = sub.drop_duplicates(subset='ID', keep='last').set_index('ID').loc[:,['Date/Time']].rename(columns={'Date/Time':'Last '+location})
    count = pd.DataFrame(sub.groupby('ID')['Date/Time'].count()).rename(columns={'Date/Time':'count '+location})
    summary = pd.concat([first_d, last_d, count], axis=1)
    
    return summary


def classify_fish_pos(fishdata, good_performers, bad_performers):
    """
    Classify the fish positions as good, bad or unclassified, according to the classification 
    of the receiver clusters.
    The function also writes out the results.
    
    Parameters
    ---------
    fishdata = dataframe with fish positions and at least column URX
    good_performers = list with good performing receiver clusters
    bad_performers = list with bad performing receiver clusters
    """
    
    fish_good = fishdata[fishdata.URX.isin(good_performers)]
    fish_bad = fishdata[fishdata.URX.isin(bad_performers)]
    fish_rest = fishdata[fishdata.URX.isin(good_performers+bad_performers)==False]
    
    print('Good positions: {:.2f}%'.format(len(fish_good)/len(fishdata)*100))
    print('Bad positions: {:.2f}%'.format(len(fish_bad)/len(fishdata)*100))
    print('Unclassified positions: {:.2f}%'.format(len(fish_rest)/len(fishdata)*100))

    
    return fish_good, fish_bad, fish_rest

    

def classify_URX_groups(fixed_tags, acc_goal, acc_limit=0.95, min_group_size=10):
    """
    Function to classify URX-groups (receiver clusters). 
    All fixed positions are grouped per receiver cluster used for calculation of the position. 
    For each receiver cluster, the percentage of positions with error <= accuracy_goal is calculated. 
    Clusters with a groupsize >= min_group_size are classified 
        - as good performers if this percentage is >= accuracy limit; 
        - as bad performers if this percentags is < accuracy limit.
    
    Parameters:
    -----------
    fixed_tags = dataframe with positions of the fixed transmitters, with at least columns HPEm and URX (datetime not in index!)
        URX = list of receivers used to calculate the position
    acc_goal = maximum allowed error
    acc_limit = minimum proportion of the group that has to meet the acc_goal
    group_size = minimum number of group members to accept a group as bad or good performing, default 10
    
    Returns:
    --------
    URX_groups = dataframe with for each URX-group the percentage of positions with error <= accuracy goal and the groupsize
    good_performers = list with good performing receiver clusters
    bad_performers = list with bad performing receiver clusters
    
    """
    
    import warnings
    warnings.filterwarnings("ignore")
                            
    fixed_tags.loc[:,'acc_check'] = [error <= acc_goal for error in fixed_tags['HPEm']]
    URX_groups = fixed_tags.groupby(by=['URX'])['acc_check'].agg(['mean', 'count'])
    URX_groups = URX_groups.reset_index().rename(columns = {'mean': 'percentage', 'count': 'groupsize'})
    URX_subset = URX_groups[URX_groups['groupsize']>=min_group_size].reset_index(drop = True)
    
    good_performers = list(pd.DataFrame(URX_subset[URX_subset['percentage'] >= acc_limit])['URX'])
    bad_performers = list(pd.DataFrame(URX_subset[URX_subset['percentage'] < acc_limit])['URX'])
    
    return URX_groups, good_performers, bad_performers

def compare_cluster_lists(URX_col, list_performers, limit):
    """
    list1 : data.URX
    list2 : list of clusters
    limit : only show differences with length <= limit
    
    Returns: dataframe with clusters of performerlist in index
    To get corresponding cluster of position data: data.URX[row]
    """
    receiver_differences = {}
    for cluster in list_performers:
        receiver_differences[cluster] = get_symm_diff_list(URX_col, cluster, limit)
        
        
    return pd.concat(receiver_differences)

def create_HPE_filter_stats(all_positions, lower_HPE, higher_HPE, steps, accepted_error):
    """
    Check the influence of different HPE limits on average error, median error, nb of calculated positions, ...
    
    Parameters
    ----------
    all_positions = dataframe of vps-positions containing at least columns HPE and HPEm
    lower_HPE = lowest HPE limit
    higher_HPE = highest HPE limit
    steps = steps in HPE
    accepted_error = accuracy goal
    
    Returns
    -------
    HPE_filter_stats = dataframe with a statistic per column calculated for different HPE-limits in the index
    """

    data_nb = len(all_positions)
    
    keys = ['nb_points', 'data_loss','avg HPEm', 'med HPEm','quant_75', 'quant_95', 'quant_99', 
            'false_pos','false_neg', 'included', 'excluded','f_pos_tot', 'f_neg_tot']

    stats = {key: {} for key in keys}

    
    for lim in np.arange(lower_HPE, higher_HPE, steps):
        stats['avg HPEm'][lim] = all_positions[all_positions.HPE <= lim]['HPEm'].mean()
        stats['med HPEm'][lim] = all_positions[all_positions.HPE <= lim]['HPEm'].median()
        stats['quant_75'][lim] = all_positions[all_positions.HPE <= lim]['HPEm'].quantile(0.75)
        stats['quant_95'][lim] = all_positions[all_positions.HPE <= lim]['HPEm'].quantile(0.95)
        stats['quant_99'][lim] = all_positions[all_positions.HPE <= lim]['HPEm'].quantile(0.99)
        stats['nb_points'][lim] = all_positions[all_positions.HPE <= lim]['HPEm'].count()
        stats['data_loss'][lim] = data_nb - all_positions[all_positions.HPE <= lim]['HPEm'].count()
        stats['false_pos'][lim] = all_positions[(all_positions.HPE <= lim)&(all_positions.HPEm>accepted_error)]['HPEm'].count()/stats['nb_points'][lim]*100
        stats['false_neg'][lim] = all_positions[(all_positions.HPE > lim)&(all_positions.HPEm<=accepted_error)]['HPEm'].count()/stats['data_loss'][lim]*100
        stats['included'][lim] = stats['nb_points'][lim]/data_nb*100
        stats['excluded'][lim] = stats['data_loss'][lim]/data_nb*100
        stats['f_pos_tot'][lim] = all_positions[(all_positions.HPE <= lim)&(all_positions.HPEm>accepted_error)]['HPEm'].count()/data_nb*100
        stats['f_neg_tot'][lim] = all_positions[(all_positions.HPE > lim)&(all_positions.HPEm<=accepted_error)]['HPEm'].count()/data_nb*100

   
    return pd.DataFrame.from_dict(stats)

def create_interpolated_grid_df(method, points, values, gridx, gridy):
    """
    function to create a dataframe of a gridframe
    
    Parameters
    ---------
    method: 'nearest', 'linear', 'cubic'
    points: df[['x','y']].as_matrix
    values: df.values_to_interpolate.as_matrix
    gridx, gridy = np.meshgrid(np.linspace(min,max,nb_of_gridcells_in_x), np.linspace(min,max,nb_of_gridcells_in_y))
    
    Returns
    -------
    df with columns x, y and value, containing the interpolated value on each gridpoint
    
    """
    grid_z0 = griddata(points, values, (gridx, gridy), method=method)
    gridframe = pd.DataFrame()

    gridframe['x'] = np.concatenate(gridx)
    gridframe['y'] = np.concatenate(gridy)
    gridframe['value'] = np.concatenate(grid_z0)
    return gridframe

def distance(X,Y):
    """
    Function to calculate the distance between sequential points in a series. 
    The distance from the previous point is calculated, so the first element is NAN.
    X is the series of x-coordinates, Y is the series of Y-coordinates (in m)
    """
    X_1 = X.shift(1)
    Y_1 = Y.shift(1)
    dist = ((X-X_1)**2+(Y-Y_1)**2)**0.5
    return dist

def error_on_fish(fishData, estimator, predictors, to_predict = 'HPEm'):
    """
    function to predict the error (HPEm) on fishdata, based on a random forest estimator.
    Inputs:
    - fishdata
    - estimator (from function randFor)
    - predictors (as used to construct the estimator)
    - to_predict: HPEm by default
    
    Output: prints the statistics on the predicted error and returns dataframe with error column added
    """
    pred_fishData = fishData
    X_fish = pred_fishData.loc[:,predictors].values
    Y_fish = estimator.predict(X_fish)
    pred_fishData[to_predict] = Y_fish
    print('\nStatistics on predicted HPEm and on given HPE')
    print(pred_fishData.loc[:,['HPEm', 'HPE']].describe([.25, .5, .75, .90, .95, .99]))
    return(pred_fishData)


def error_on_vps(vps_track, gps_track):
    """
    Function to calculate the error of each vps point (Vincenty distance to corresponding gps point).
    For those vps positions that have no gps point on exact datetime: closest gps point in time is taken
    
    Parameters
    ----------
    vps_track : vps data for a given period, with datetime as index
    gps_track: gps data (or known positions) for a given period, with datetime as index
    
    Returns
    -------
    distances : tuple object with for each datetime (corresponding to a vps position) the error
    vps_results: vps data complemented with HPEm-values (errors), LAT_GPS and LON_GPS
    """
    
    gps_selection = gps_track[gps_track.index.isin(vps_track.index)] # only gps points corresponding with a vps point
    
    vps_in_gps = vps_track[vps_track.index.isin(gps_selection.index)==True]
    vps_in_gps = vps_in_gps.reset_index()
    vps_in_gps['LON_GPS'] = gps_selection.reset_index()['LON']
    vps_in_gps['LAT_GPS'] = gps_selection.reset_index()['LAT']
    
    vps_not_in_gps = vps_track[vps_track.index.isin(gps_selection.index)==False] #if length of trimble selection is not equal to length vps
    lon=[]
    lat=[]
    # this loop needs datetime in index of both dataframes!
    for i in range(len(vps_not_in_gps)):
        lon.append(gps_track.iloc[gps_track.index.get_loc(vps_not_in_gps.index[i], method='nearest')].LON)
        lat.append(gps_track.iloc[gps_track.index.get_loc(vps_not_in_gps.index[i], method='nearest')].LAT)

    vps_not_in_gps = vps_not_in_gps.reset_index() # to make it possible adding columns
    vps_not_in_gps['LAT_GPS'] = lat
    vps_not_in_gps['LON_GPS'] = lon

    vps_result = pd.concat([vps_in_gps, vps_not_in_gps], ignore_index=True).set_index('DATETIME').sort_index()
    
    positions = vps_result.loc[:,['LAT_GPS', 'LON_GPS']].rename(columns = {'LAT_GPS': 'LAT', 'LON_GPS': 'LON'})
    
    distances = pd.concat([pd.Series(vps_track.index), pd.Series(vincent_distance(vps_track, positions))], axis = 1).rename(columns = {0: 'Vinc_dist'})
    vps_result['HPEm'] = vincent_distance(vps_track, positions)
    
    return distances, vps_result


def filter_performance(unfiltered_track, filtered_track, accuracy_goal):
    """
    Function to calculate performance statistics of the URX filtering method
    
    Paramaters
    ----------
    unfiltered_track : dataframe  with original vps positions, datetime as index
    filtered_track : dataframe with filtered vps positions, datetime as index and HPEm included
    
    Returns
    -------
    performance : dictionnary with average HPEm; HPEm quantiles 70, 80, 90, 95, 99;
                    nb of included/excluded positions; nb and percentages of false pos/negs
    """
    
    performance = {}
    performance['avg HPEm'] = filtered_track['HPEm'].mean()
    performance['70-Q HPEm'] = filtered_track['HPEm'].quantile(0.7)
    performance['80-Q HPEm'] = filtered_track['HPEm'].quantile(0.8)
    performance['90-Q HPEm'] = filtered_track['HPEm'].quantile(0.9)
    performance['95-Q HPEm'] = filtered_track['HPEm'].quantile(0.95)
    performance['99-Q HPEm'] = filtered_track['HPEm'].quantile(0.99)
    performance['acc_goal_satisfiers'] = stats.percentileofscore(filtered_track['HPEm'],accuracy_goal)
    performance['included'] = len(filtered_track)
    performance['false_pos'] = len(filtered_track[filtered_track.HPEm>accuracy_goal])
    performance['false_pos_perc'] = performance['false_pos']/len(filtered_track)*100
    
    bl = unfiltered_track.index.isin(filtered_track.index)
    excluded = unfiltered_track[bl==False]
    performance['false_neg'] = len(excluded[excluded.HPEm<accuracy_goal])
    performance['false_neg_perc'] = performance['false_neg']/len(excluded)*100
    performance['excluded'] = len(excluded)
    
    return performance

def find_tagname(full_ID, transmitter_df, name='Name', fullID = 'Pinger Full ID'):
    """
    transmitter_df: df containing fullIDs and tagnames
    """
    if full_ID in list(transmitter_df[fullID]):
        tagname = transmitter_df[transmitter_df[fullID]==full_ID][name].item()
    else:
        tagname = 'NaN'
    return tagname

def find_perc_size(row, URX_groups):
    """
    Goal
    ----
    
    find for each position, of the corresponding URX group:
    perc = the percentage of fixed tag positions with error <= accuracy goal
    size = the nb of fixed tags positions calculated with this URX group
    
    perc is to be used as color code
    size has to be >10 to use the position's classification
    
    Use
    ---
    
    URX_groups, good_performers, bad_performers = classify_URX_groups(...)
    positions['class_perc'] = list(positions.reset_index().apply(lambda row: find_perc_size(row, URX_groups)[0], axis=1))
    positions['class_groupsize'] = list(positions.reset_index().apply(lambda row: find_perc_size(row, URX_groups)[1], axis=1))
    
    """
    if row.URX in list(URX_groups.URX):
        perc = URX_groups[URX_groups.URX==row.URX].percentage.item()
        size = URX_groups[URX_groups.URX==row.URX].groupsize.item()
    else:
        perc=None
        size=None
        
    return perc,size

def get_bad_performers(fixed_tags, acc_goal, acc_limit):
    """
    Function that returns the list of well-performing URX-groups.
    Takes as data the positions of the fixed transmitters
    acc_goal = maximum allowed error
    acc_limit = minimum proportion of the group that has to meet the acc_goal
    """
    fixed_tags.loc[:,'acc_check'] = [error <= acc_goal for error in fixed_tags['HPEm']]
    URX_groups = fixed_tags.groupby(by=['URX'])['acc_check'].mean()
    
    return list(pd.DataFrame(URX_groups[URX_groups < acc_limit]).reset_index()['URX'])


def get_good_performers(fixed_tags, acc_goal, acc_limit):
    """
    Function that returns the list of well-performing URX-groups.
    Takes as data the positions of the fixed transmitters
    acc_goal = maximum allowed error
    acc_limit = minimum proportion of the group that has to meet the acc_goal
    """
    fixed_tags.loc[:,'acc_check'] = [error <= acc_goal for error in fixed_tags['HPEm']]
    URX_groups = fixed_tags.groupby(by=['URX'])['acc_check'].mean()

    return list(pd.DataFrame(URX_groups[URX_groups >= acc_limit]).reset_index()['URX'])


def get_difference_between_cluster_strings(cluster1, cluster2):
    """
    cluster1, cluster2 : string of receiver clusters
    """
    return set(cluster1.split(' ')).difference(set(cluster2.split(' ')))

def get_symm_diff_between_cluster_strings(cluster1, cluster2):
    """
    cluster1, cluster2 : string of receiver clusters
    
    Returns : set of receivers differing between both clusters
    """
    return set(cluster1.split(' ')).symmetric_difference(set(cluster2.split(' ')))

def get_symm_diff_list(cluster_list, compare_cluster, limit):
    """
    cluster_list : data.URX
    compare_cluster : 1 cluster (string containing receiver names)
    
    Returns: series with symmetric difference sets
    """
    symdif = pd.DataFrame(cluster_list.apply(lambda x: get_symm_diff_between_cluster_strings(x, compare_cluster))) 
    return symdif[symdif.URX.apply(lambda x: len(x)<=limit)]

def infotable(rec_no_vps):
    table=rec_no_vps.groupby(by=['location', 'ID'])['Date/Time'].agg(['min', 'max', 'size'])
    table.columns=['begin_time', 'end_time', 'nb_of_detections']
    table[['begin_time', 'end_time']] = table[['begin_time', 'end_time']].astype(str)
    table=table.reset_index()
    return(table)

def is_pos_in_x_closest_well_perf(row, fixed_tag_gps_pos, fixed_tag_data, acc_goal, conf_level, x=4):
    """
    Check if the position is calculated by a receiver cluster that belongs to 
    the well performing receiver clusters of the x closest stationary tags (default 4 = optimum for tests).
    If x = 13 (all stationary tags) the result is the same as taking all fixed tags for classification (with unlimited nb_in_group)
    Usage: list = [is_pos_in_x_closest_well_perf(position_data.iloc[p,:], ref_syncs_only, fixed_tag_data, 2.5, 0.95, x=3) for p in np.arange(len(position_data))]
    
    Parameters
    ----------
    row = row of the dataframe corresponding to one position
    x = number of closest receivers to take into account
    fixed_tag_gps_pos = dataframe with gps positions (Latitude, Longitude) and Station Name of fixed tags
    fixed_tag_data = dataframe with vps positions
    acc_goal
    conf_level
    
    Return
    ------
    True if the receiver cluster is in the list of well performing clusters of the x closest receivers, 
    False if not
    
    """
    
    point = row.loc[['LAT', 'LON']]
    coordinates = fixed_tag_gps_pos.loc[:,['Latitude', 'Longitude']]
    coo_list = [tuple(x) for x in coordinates.to_records(index=False)]
    dist_to_fixed = pd.DataFrame({'fixed_tag': fixed_tag_gps_pos['Station Name'], 'distance': [vincenty(x, point).m for x in coo_list]})
    x_closest = dist_to_fixed.sort_values(by='distance').reset_index(drop=True)[0:x]
    subset = fixed_tag_data[fixed_tag_data.TRANSMITTER.isin(x_closest.fixed_tag)]
    URX_groups, good_performers, bad_performers = classify_URX_groups(subset, acc_goal, conf_level, min_group_size=10)
    
    return row.URX in good_performers

def lookuptable(dates_to_match, lookup_dates, lookup_values, method='nearest'):
    """
    NOTE: USE lookuptable.reindex(dates_to_match, method='nearest') INSTEAD, 100 TIMES FASTER!
    
    Function that works like excel's lookup table: finds for each date in 'dates_to_match'
    the closest date in the 'lookup_dates', and returns the corresponding 'lookup_values'
    
    Parameters
    ----------
    dates_to_match : 1D-array or listlike, datetimes to match
    lookup_dates : 1D-array or listlike, datetimes to lookup
    lookup_values : 1D-array or listlike, values to lookup, 
                    must have the same length as lookup_dates
    method : matching method, default 'nearest'

    Returns
    -------
    matching_values = list with corresponding lookupvalues
    """
    
    # check if lookup_dates and lookup_values are of equal length, if not return error message
    if len(lookup_dates)==len(lookup_values):
        df = pd.DataFrame({'datetime':dates_to_match, 'matches':None})
        lookuptable = pd.DataFrame({'datetime':lookup_dates, 'values':lookup_values}).set_index('datetime')

        df.matches = df.apply(lambda row: lookuptable.iloc[
            lookuptable.index.get_loc(row['datetime'], method=method)]['values'], axis=1)
        res = list(df.matches)
    else:
        print('Error: lookup_dates and lookup_values must have equal length!')
        res = None
    return res

def make_geo_poly(df, x_name, y_name):
    """
    Makes a GeoPanda Polygon out of dataframe containing the coordinates of the polygon
    df : DataFrame with x columns and y column that contain the x,y coordinates of the polygon
    x : name of x-col
    y : name of y-col
    """
    
    return Polygon([(x,y) for x,y in zip(df[x_name],  df[y_name])])

def make_poly_list(ref_syncs, cluster):
    
    """
    Converts URX of this cluster to a list of (longitude,latitude) coordinates of the cluster receivers.
    Use on a dataframe with URX column: list(map(lambda x: make_poly_list(ref_syncs, x), df.URX))
    df['inside'] = list(map(lambda i: point_in_poly(df[0].LON.iloc[i], df.LAT.iloc[i], df.poly_lists.iloc[i]), df.index))
    This is needed to calculate point_in_poly on an entire dataframe:
    """
    
    stations_cluster = ref_syncs[ref_syncs['Station Name'].isin(cluster.split(sep=' '))]
    poly_list = list(zip(stations_cluster.Longitude, stations_cluster.Latitude))
    
    return poly_list


def performance_table(track_with_filters, accuracy_goal, limits):
    """
    Function to calculate the URX filter performance for different accurarcy limits
    
    Parameters
    ----------
    track_with_filters : dictionnary with keys 0, 0.7, 0.8, ..., 0.99 where each element contains 
                        the filtered dataframe corresponding with the limit indicated in key (0 = unfiltered)
    accuracy_goal : max acceptable error on a position (e.g. 2.5m, 5m, ...)
    limits : list of accuracy limits (keys of track_with_filters), excluding 0
    
    Returns
    -------
    perf : table with limits in the index and performance statistics in the columns
    """
    perf = []
    for i in limits:
        temp = filter_performance(track_with_filters[0], track_with_filters[i], accuracy_goal)
        temp = pd.DataFrame.from_dict(temp, orient='index')
        if isinstance(perf, pd.DataFrame):
            perf = pd.concat([perf, temp], axis = 1)
        else:
            perf = temp
    perf.columns = limits
    
    return perf.transpose()



def point_in_poly(x,y,poly):
    
    """
    !! OPGELET !! FUNCTIE WERKT NIET ZOALS VERWACHT, GEEF FOUTEN!!
    x = lon
    y = lat
    poly = list of (x,y) pairs
    returns True if point is in polygon, otherwise returns False
    """

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


def randFor(fixedData, predictors, to_predict, estimator_crit='mse', val_scoring='r2', nb_valgroups=4):
    """
    function that gathers al utilities for fitting and validating the random forest estimator
    Inputs:
    - predictors: columns used in the prediction
    - to_predict: HPEm, the error on the positions
    - estimator_crit: criterion used in fitting, default mse
    - val_scoring: score given in validation, default r2
    - nb_valgroups: data are divided in groups (default 4) to do cross validation
    
    Output: 
    - estimator (.feature_importances_, .predict)
    - scores cross validation
    - scores timeseries split validation
    - importances (attribution of each predictor)
    """
    

    X_fix = fixedData.loc[:,predictors].values
    # to predict:
    Y_fix = fixedData[to_predict].values
    estimator = RandomForestRegressor(n_estimators=10, criterion = estimator_crit)
    estimator.fit(X_fix, Y_fix)
    
    score_cross = cross_val_score(estimator, X_fix, Y_fix, cv = nb_valgroups, scoring = val_scoring)

    tscv = TimeSeriesSplit(n_splits=nb_valgroups)
    for train_index, test_index in tscv.split(X_fix, y = Y_fix):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_fix[train_index], X_fix[test_index]
        Y_train, Y_test = Y_fix[train_index], Y_fix[test_index]
        estimator.fit(X_train, Y_train)
        score_ts_split = estimator.score(X_test, Y_test)
    
    importances = {}
    i = 0
    for pred in predictors:
        importances[pred] = estimator.feature_importances_[i]
        i = i+1
    
    print('Importance of each variable: \n')
    pprint.pprint(importances)
    
    print('\nK-fold cross validation scores: \n')
    print(score_cross)
    
    print('\nTimeseries cross validation scores: \n')
    print(score_ts_split)
    
    return(estimator, score_cross, score_ts_split, importances)


def refSyncCheck(path, IDlist, shifter, transRateStat):
    """
    Function returning:
    - concatenated dataframe with positions of each fixed trasmitter
    - for HPE, HPEm and position yield:
        - concatenated dataframe with statistics per hour-of-day for each fixed transmitter
        - concatenated dataframe with statistics per day-of-week for each fixed transmitter
    Calculated statistisc are min, mean, max, median, number
    Parameters needed:
    - path: VPS_RESULTS_...
    - IDlist: REFS+SYNCS
    - shifter: half of the timeframe over which position yield will be calculated
    - transRateStat: average random delay + transmittion length of the refs and syncs
    
    """

    # load positioning data of reference and synctags
    dataRefSync = {}
    hour_summ_HPE = {}
    weekday_summ_HPE = {}
    hour_summ_HPEm = {}
    weekday_summ_HPEm = {}
    hour_summ_posYield = {}
    weekday_summ_posYield = {}


    for tagID in IDlist:
        filename = ''.join([path,'Positions/TRANSMITTER-', tagID, '-CALC-POSITIONS.csv'])
        temp = pd.read_csv(os.path.join(filename))
        temp['DATETIME'] = pd.to_datetime(temp['DATETIME'])
        temp['hour'] = temp['DATETIME'].apply(lambda x: x.hour)
        hour_bins = pd.cut(temp['hour'], bins=np.arange(0,25,1), right = False)
        temp['hour_bins'] = hour_bins
        temp['weekday'] = temp['DATETIME'].apply(lambda x: x.weekday())
        weekday_bins = pd.cut(temp['weekday'], bins=np.arange(0,8,1), right = False)
        temp['weekday_bins'] = weekday_bins

        #calculations for position yield evolution
        after = temp['DATETIME'].shift(-shifter)
        before = temp['DATETIME'].shift(shifter)
        timediff = after - before
        expected = timediff.astype('timedelta64[s]')/transRateStat # expected nb of pings within given time interval
        temp['posYieldInTime'] = (2*shifter-1)/expected*100

        dataRefSync[tagID] = temp
        hour_summ_HPE[tagID] = temp.groupby('hour_bins')['HPE'].agg([min,np.mean,np.std,max, np.median, np.count_nonzero])
        weekday_summ_HPE[tagID] = temp.groupby('weekday_bins')['HPE'].agg([min,np.mean,np.std,max, np.median, np.count_nonzero])
        hour_summ_HPEm[tagID] = temp.groupby('hour_bins')['HPEm'].agg([min,np.mean,np.std,max, np.median, np.count_nonzero])
        weekday_summ_HPEm[tagID] = temp.groupby('weekday_bins')['HPEm'].agg([min,np.mean,np.std,max, np.median, np.count_nonzero])
        hour_summ_posYield[tagID] = temp.groupby('hour_bins')['posYieldInTime'].agg([min,np.mean,np.std,max, np.median, np.count_nonzero])
        weekday_summ_posYield[tagID] = temp.groupby('weekday_bins')['posYieldInTime'].agg([min,np.mean,np.std,max, np.median, np.count_nonzero])

    dataRefSync_kols = pd.concat(dataRefSync, axis=0)
    dataRefSync = pd.concat(dataRefSync, axis = 1)
    return(dataRefSync, dataRefSync_kols, hour_summ_HPE, hour_summ_HPEm, hour_summ_posYield, weekday_summ_HPE, weekday_summ_HPEm, weekday_summ_posYield)

def timeInterval(datetime):
    """
    Function to calculate the interval between sequential times in a series.
    """
    
    detInterval = np.zeros((datetime.shape[0]))
    for t in range (1,datetime.shape[0]):
        currentInt = (datetime[t] - datetime[t-1])
        detInterval[t] = currentInt.seconds
    detInterval[0] = 'NaN'
    return detInterval

def three_fold_cross(dataset):
    """
    Split dataset randomly in 3 parts.
    
    Inputs:
    -------
    dataset = pandas dataframe
    
    Returns:
    --------
    3 new dataframes which are parts of the old dataframe
    """
    nb_list = np.arange(len(dataset))
    np.random.shuffle(nb_list)
    
    part1 = dataset.iloc[nb_list[:round(len(dataset)/3)],:]
    part2 = dataset.iloc[nb_list[round(len(dataset)/3)+1:round(len(dataset)/3*2)],:]
    part3 = dataset.iloc[nb_list[round(len(dataset)/3*2)+1:],:]
    
    
    return part1, part2, part3

def transform_lambert72_to_latlon(row):
    """
    Transform Lambert 72 coordinates in latitude, longitude. Can be called on a dataframe with df.apply(lambda row: transform(...) axis = 1)
    
    Parameters
    ---------
    row = row of dataframe containing columns Easting and Northing
    
    Returns
    -------
    dataframe with longitude in first column, latitude in second column
    """
    lamb72 = Proj(init='EPSG:31370')
    lonlat = Proj(init='epsg:4326')
    return pd.Series(transform(lamb72,lonlat,row.Easting,row.Northing))

def transform_latlon_to_lambert72(row, lon_col, lat_col):
    """
    Transform latlon coordinates to lambert 72. Can be called on a dataframe with df.apply(lambda row: transform(...), axis = 1)
    
    Parameters
    ---------
    row = row of dataframe containing columns with longitude and latitude
    
    Returns
    -------
    dataframe with East meters in first column, North meters in second column
    """
    lamb72 = Proj(init='EPSG:31370')
    lonlat = Proj(init='epsg:4326')
    return pd.Series(transform(lonlat,lamb72,row[lon_col],row[lat_col]))

def transform_point(x, y, trans_matrix):
    """
    coordinate transformation of a point
    
    Parameters
    ----------
    x = x-coordinate of the point
    y = y-coordinate of the point
    trans_matrix = transformation matrix
    
    Returns
    -------
    a, b = transformed coordinates
    
    Usage
    -----
    Create the transformation matrix based on 3 known points in both coordinate systems:
    transformation matrix = AB * XY.I with:
        AB = np.matrix([[a1, a2, a3], [b1, b2, b3], [1, 1, 1]])
        XY = np.matrix([[x1 , x2, x3, [y1, y2, y3], [1, 1, 1]])
        
    Apply the function on a list:
    [transform_point(x,y, trans_matrix) for (x,y) in zip(xlist, ylist)]
    
    """
    
    ab = trans_matrix*np.matrix([x, y, 1]).T
    a = ab[0].item()
    b = ab[1].item()
    return a, b

def transform_wgs_to_utm(row, lon_col, lat_col):
    """
    Converts the lon and lat coordinates of this row into a Series of x and y in UTM (m).
    Usage: df.apply(lambda row: transform(row, lon_col, lat_col), axis = 1)
    
    """
    utm12n = Proj("+init=EPSG:32612")
    wgs84 = Proj("+init=EPSG:4326")
    return pd.Series(transform(wgs84, utm12n, row[lon_col], row[lat_col]))



def URX_filtering(FISH, tagData_F0, URX_bad_performing):
    """
    function for performing filtering based on receiver group performance
    """
    
    tD = {}
    stayDurations_F1 = {}
    periods_F1 = {}
    lastLines_F1 = {}
    vertical_F1 = {}

    for shortID in FISH:
        if len(tagData_F0[shortID])>1: # if there is only one position for the fish, no reason to discard it or for calculating velocities
            boolean_to_discard = tagData_F0[shortID]['URX'].apply(lambda x: x in URX_bad_performing)
            # recalculate distance, time and velocity between detections.
            tempTD, tempDS, tempPer, tempLast = calcDisDetU(tagData_F0[shortID][boolean_to_discard==False], str(shortID)) #extra argument: cutoff
            stayDurations_F1[shortID] = tempDS
            periods_F1[shortID] = tempPer
            tD[shortID] = tempTD
            lastLines_F1[shortID] = tempLast
        else:
            stayDurations_F1[shortID]=0
            periods_F1[shortID]=None
            tD[shortID]=tagData_F0[shortID]
            lastLines_F1[shortID] = None

    #tagData_F1 = pd.concat(tD, axis=1)
    tagData_F1 = tD
    vertical_F1 = pd.concat(tD, axis=0)
    lastLines_F1 = pd.concat(lastLines_F1, axis = 1)
    lastLines_F1 = lastLines_F1.transpose()
    
    return tagData_F1, vertical_F1, lastLines_F1


def URX_filtering_stats(pos_to_filter, known_pos, limits, accuracy_goal, min_nb_in_group):
    """
    Calculate percentages of excluded, included and unclassified positions for data to be filtered, and also false neg/pos/neutrals.
    The filtering is based on data with known positions, of which the URX groups are classified for a given accuracy limit.
    
    Parameters
    ----------
    pos_to_filter = dataframe with positioning data needing to be filtered
    known_pos = dataframe with positions with known HPEm (error)
    limits = list of accuracy limits (e.g. [0.7, 0.8, 0.9])
    accuracy_goal = maximum acceptable error (usually 2.5)
    min_nb_in_group = minimum number of positions calculated by a URX-cluster to allow it to be classified (usually 5)
    
    Returns
    -------
    dataframe with limits in index and percentages in columns
    """
    keys = ['excluded', 'false_neg', 'f_neg_tot', 'included', 'false_pos', 'f_pos_tot', 'unclassified', 'false_neutral', 'f_neutr_tot', 'avg_HPEm', 'med_HPEm', 'quant_95']
    stats = {key: {} for key in keys}

    for lim in limits:
        URX_groups, good_perf, bad_perf = classify_URX_groups(known_pos, accuracy_goal, lim, min_group_size = min_nb_in_group)
        
        excluded_pos = pos_to_filter[pos_to_filter['URX'].isin(bad_perf)]
        stats['excluded'][lim] = len(excluded_pos)/len(pos_to_filter)*100
        if len(excluded_pos) > 0:
            stats['false_neg'][lim] = len(excluded_pos[excluded_pos.HPEm<=2.5])/len(excluded_pos)*100
        else:
            stats['false_neg'][lim] = None
        
        stats['f_neg_tot'][lim] = len(excluded_pos[excluded_pos.HPEm<=2.5])/len(pos_to_filter)*100

        included_pos = pos_to_filter[pos_to_filter['URX'].isin(good_perf)]
        stats['included'][lim] = len(included_pos)/len(pos_to_filter)*100
        if len(included_pos) > 0:
            stats['false_pos'][lim] = len(included_pos[included_pos.HPEm>2.5])/len(included_pos)*100
        else:
            stats['false_pos'][lim] = None
        
        stats['f_pos_tot'][lim] = len(included_pos[included_pos.HPEm>2.5])/len(pos_to_filter)*100
            
        unclassified_pos = pos_to_filter[np.logical_not((pos_to_filter['URX'].isin(bad_perf)|(pos_to_filter['URX'].isin(good_perf))))]
        stats['unclassified'][lim] = len(unclassified_pos)/len(pos_to_filter)*100
        if len(unclassified_pos) > 0:
            stats['false_neutral'][lim] = len(unclassified_pos[unclassified_pos.HPEm>2.5])/len(unclassified_pos)*100
        else:
            stats['false_neutral'][lim] = None
            
        stats['f_neutr_tot'][lim] = len(unclassified_pos[unclassified_pos.HPEm>2.5])/len(pos_to_filter)*100
        
        stats['avg_HPEm'][lim] = included_pos.HPEm.mean()
        stats['med_HPEm'][lim] = included_pos.HPEm.median()
        stats['quant_95'][lim] = included_pos.HPEm.quantile(0.95)

            
    return pd.DataFrame.from_dict(stats)

def URX_to_binary(allData, wishlist, RECS):
    
    """
    Function to select the needed IDs and convert the URX receiver list in binary columns.
    Inputs:
    - allData (can be dataframe or dictionary)
    - wishlist: eg FISH, EEL, fixed transmitters, ...
    - RECS: all receivers (including stations without sync tags)
    """
    
    if type(allData) == dict:
        allData = pd.concat(allData, axis = 0).reset_index(drop = True)

    data_with_binaries = allData[allData['TRANSMITTER'].isin(wishlist)].reset_index(drop = True)
    data_with_binaries = data_with_binaries.loc[:,['TRANSMITTER', 'LAT', 'LON', 'HPEm', 'HPE', 'URX']]
    
    for receiver in RECS:
        data_with_binaries[receiver] = data_with_binaries['URX'].apply(lambda x: receiver in x)*1
        
    # keep only predictors and HPEm
    data_with_binaries = data_with_binaries.loc[:,['LAT', 'LON', 'HPE', 'HPEm']+RECS]
    
    return(data_with_binaries)




def vincent_distance(series1, series2, latcol='LAT', loncol='LON'):
    """
    Calculates vincenty distance between 2 points in m
    Input are 2 series, containing columns with LAT and LON coordinates (LAT,LON! in geopy)
    Error if both series are not equal in length!
    """
    vincent_series = []

    if len(series1)==len(series2):
        for i in range(len(series1)):
            p1 = series1.loc[:,[latcol, loncol]].iloc[i,:]
            p2 = series2.loc[:,[latcol, loncol]].iloc[i,:]
            vincent_series.append(vincenty(p1, p2).m)
    else:
        print('Series are not of equal length!')
    
    return vincent_series





