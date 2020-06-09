# IMPORTS

import numpy as np
import pandas as pd
import sklearn.metrics
import math
import matplotlib.pyplot as plt

# Auxiliar functions for error checking and list/array size

def aux_error_checking(state_gt, state_pred):
    '''
    Checks for list/array size incompatibility
    '''
    
    if isinstance(state_gt, list):
        if len(state_gt) != len(state_pred):
            print('Ground truth and predicted arrays must be of the same size')
            return True
        else:
            return False
    elif state_gt.shape[0] != state_pred.shape[0]:
        print('Ground truth and predicted arrays must be of the same size')
        return True
    else:
        return False
    

def aux_get_size(state_gt):
    '''
    Gets the size of list/array for iteration
    '''
    
    temp_length = 0
    if isinstance(state_gt, list):
        temp_length = len(state_gt)
    else:
        temp_length = state_gt.shape[0]
        
    return temp_length


# Definition of metrics

def metrics_re(state_gt, state_pred):
    '''
    Calculates the relative error of the energy estimation: 
    RE = (sum(energy_gt) - sum(energy_pred)) / sum(energy_gt)
    '''
    if aux_error_checking(state_gt, state_pred):
        return
    
    temp_numerator = (np.sum(state_gt) - np.sum(state_pred))
    temp_denominator = np.sum(state_gt)
    
    if temp_denominator != 0:
        return temp_numerator / temp_denominator
    else:
        return 0


def metrics_cv_rmsd(state_gt, state_pred):
    '''
    Calculates the 1 - Covariance of the RMSD
    inv_cv = 1 - (RMSE / mean(ground_truth))
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    if np.mean(state_gt) == 0:
        return 0
    
    temp_cv_rmsd = 1 - (np.sqrt(sklearn.metrics.mean_squared_error(state_gt, state_pred)) / np.mean(state_gt))
    
    return temp_cv_rmsd


def metrics_rmse(state_gt, state_pred):
    '''
    Calculates the root mean squared values
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    temp_rmse = np.sqrt(sklearn.metrics.mean_squared_error(state_gt, state_pred))
    
    return temp_rmse
    

def metrics_abse(state_gt, state_pred):
    '''
    Calculate the Average Error of the ground truth vs predicted values
    ae = sum()
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    temp_ae = 0
    temp_length = aux_get_size(state_gt)
    for i in np.arange(0, temp_length, 1):
        temp_ae += np.abs(state_gt[i] - state_pred[i])
    
    return temp_ae / temp_length


def metrics_mae(state_gt, state_pred):
    '''
    Calculates the Mean Absolute Error of the ground truth vs predicted values
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    return sklearn.metrics.mean_absolute_error(state_gt, state_pred)


def metrics_ae(state_gt, state_pred):
    '''
    Calculate the Average Error of the ground truth vs predicted values
    ae = sum()
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    temp_ae = 0
    temp_length = aux_get_size(state_gt)
    for i in np.arange(0, temp_length, 1):
        temp_ae += (state_pred[i] - state_gt[i])
    
    return temp_ae / temp_length
    
    
def metrics_sde(state_gt, state_pred):
    '''
    Calculate the standard deviation error
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    temp_length = aux_get_size(state_gt)
        
    temp_placeholder = 0    
    temp_delta_mean = metrics_ae(state_gt, state_pred)
    
    for i in np.arange(0, temp_length, 1):
        temp_placeholder += ((np.abs(state_pred[i] - state_gt[i]) - temp_delta_mean) ** 2)
        
    temp_sde = np.sqrt(temp_placeholder / temp_length)
    
    return temp_sde


def metrics_rsquared(state_gt, state_pred):
    '''
    Calculates the r-squared value
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    temp_rsquared = sklearn.metrics.r2_score(state_gt, state_pred)
    
    return temp_rsquared


def metrics_psde(state_gt, state_pred):
    '''
    Percent Standard Deviation Explained: 1 - sqrt(1 - r-squared)
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    temp_rsquared = metrics_rsquared(state_gt, state_pred)
    temp_psde = 1 - np.sqrt(1 - temp_rsquared)
    
    return temp_psde


def metrics_ee(state_gt, state_pred):
    '''
    Calculates the energy error of the predictions versus the ground truth values
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    temp_length = aux_get_size(state_gt)
    temp_numerator = 0
    for i in np.arange(0, temp_length, 1):
        temp_numerator += np.abs(state_pred[i] - state_gt[i])
    
    temp_denominator = np.sum(state_gt)
    
    if temp_denominator == 0:
        return 0
    
    temp_ee = temp_numerator / temp_denominator
    
    return temp_ee


def metrics_eav1(state_gt, state_pred, alpha_param=1.4):
    '''
    Calculate energy accuracy
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    temp_ee = metrics_ee(state_gt, state_pred)
    temp_eav1 = math.exp(-alpha_param * temp_ee)
    
    return temp_eav1


def metrics_mr_micro(state_gt, state_pred):
    '''
    Micro Match rate of energy
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    temp_numerator = 0
    temp_denominator = 0
    
    temp_length = aux_get_size(state_gt)
    for i in np.arange(0, temp_length, 1):
        temp_numerator += min(state_gt[i], state_pred[i])
        temp_denominator += max(state_gt[i], state_pred[i])
        
        
    if temp_denominator == 0:
        return 0
    
    temp_mr = temp_numerator / temp_denominator
    
    return temp_mr


def metrics_mr_feel(state_gt, state_pred):
    '''
    Macro Match rate FEEL of energy. Outputs Macro MR, UnderMR and OverMR
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    temp_mr = 0
    temp_mr_count = 0
    
    temp_undermatch_rate = 0
    temp_overmatch_rate = 0
    
    temp_numerator = 0
    temp_denominator = 0
    
    temp_length = aux_get_size(state_gt)
    for i in np.arange(0, temp_length, 1):
        
        if state_gt[i] == 0 and state_pred[i] > 0:
            temp_overmatch_rate += 1
        elif state_gt[i] > 0 and state_pred[i] == 0:
            temp_undermatch_rate += 1
        elif state_gt[i] == 0 and state_pred[i] == 0:
            temp_mr += 1
            temp_mr_count += 1
        else:
            temp_numerator = min(state_gt[i], state_pred[i])
            temp_denominator = max(state_gt[i], state_pred[i])
            
            temp_mr += (temp_numerator / temp_denominator)
            temp_mr_count += 1
    
    temp_undermatch_rate /= temp_length
    temp_overmatch_rate /= temp_length
    
    if temp_mr_count > 0:
        temp_mr = temp_mr / temp_mr_count
    else:
        temp_mr = 0
    
    return temp_mr, temp_undermatch_rate, temp_overmatch_rate


def metrics_fece(state_gt, state_pred):
    '''
    Custom metric to return the correctly explained under, over and correct estimations
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    total = 0
    
    cu = 0
    co = 0
    c = 0
    
    ozero = 0
    o = 0
    u = 0
    
    temp_length = aux_get_size(state_gt)
    for i in np.arange(0, temp_length, 1):
        total += state_gt[i]
        
        # Correctly assigned
        if state_gt[i] == state_pred[i]:
            c += state_gt[i]
            
        # Underestimation
        elif state_gt[i] > state_pred[i]:
            u += (state_gt[i] - state_pred[i])
            cu += state_pred[i]
            
        # Overestimation
        elif state_gt[i] < state_pred[i]:
            if state_gt[i] > 0:
                o += (state_pred[i] - state_gt[i])
            else:
                ozero += (state_pred[i] - state_gt[i])
            co += state_gt[i]
            
    return c, co, cu, o, ozero, u, total


def metrics_MRuMR_macro(state_gt, state_pred):
    '''
    Macro Match and Un-Match rate of energy
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    temp_mr = 0
    temp_mr_count = 0
    
    temp_unmatch_rate = 0
    temp_unmatch_rate_zero_counts = 0
    
    temp_length = aux_get_size(state_gt)
    for i in np.arange(0, temp_length, 1):
        
        if state_gt[i] > 0:
            temp_numerator = min(state_gt[i], state_pred[i])
            temp_denominator = max(state_gt[i], state_pred[i])
            
            temp_mr += (temp_numerator / temp_denominator)
            temp_mr_count += 1
        else:
            temp_unmatch_rate_zero_counts += 1
            
            if state_pred[i] > 0:
                temp_unmatch_rate +=1
            
    
    if temp_unmatch_rate_zero_counts > 0:
        temp_unmatch_rate /= temp_unmatch_rate_zero_counts
    else:
        temp_unamtch_rate = 0
    
    if temp_mr_count > 0:
        temp_mr = temp_mr / temp_mr_count
    else:
        temp_mr = 0
    
    return temp_mr, temp_unmatch_rate


def metrics_sem(state_gt):
    '''
    Standard error of the mean
    '''
    
    temp_length = aux_get_size(state_gt)
    temp_std = np.std(state_gt)
    
    temp_sem = temp_std / np.sqrt(temp_length)
    
    return temp_sem
    
    
def metrics_fee(state_gt, state_pred):
    '''
    Calculates the fraction of energy explained
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    temp_numerator = np.sum(state_pred)
    temp_denominator = np.sum(state_gt)
    
    if temp_denominator <= 0:
        return np.nan
    
    return temp_numerator / temp_denominator


def metrics_teca(state_gt, state_pred):
    '''
    Total Energy Correctly Assigned
    '''
    
    if state_gt.shape != state_pred.shape:
        print('Error! Array dimension must match!')
        return
    
    temp_placeholder = 0
    for i in np.arange(0, state_gt.shape[0], 1):
        
        temp_appliance = 0
        for j in np.arange(0, state_gt.shape[1], 1):
            temp_appliance += np.abs(state_pred[i, j] - state_gt[i, j])
            
        temp_placeholder += temp_appliance
        temp_totalEnergy += np.sum()
        
    temp_numerator = 1 - temp_placeholder
    temp_denominator = 2 * (np.sum(state_gt))
    
    if temp_denominator == 0:
        return 0
    
    return temp_numerator / temp_denominator


def metrics_etea(state_gt, state_pred):
    '''
    Error in total energy assigned
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    temp_sum_gt = np.sum(state_gt)
    temp_sum_pred = np.sum(state_pred)
    
    return np.abs(temp_sum_pred - temp_sum_gt)


def metrics_dev(state_gt, state_pred):
    '''
    Ratio between ETEA and the actual energy consumed
    '''
    
    if aux_error_checking(state_gt, state_pred):
        return
    
    temp_sum_gt = np.sum(state_gt)
    temp_etea = metrics_etea(state_gt, state_pred)
    
    if temp_sum_gt == 0:
        return 0
    
    return (temp_etea / temp_sum_gt)


def metrics_fteac(state_gt, state_pred):
    return