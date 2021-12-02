#!/usr/bin/env python

"""
Read wafer probing data and dump CSV files containing
k-factor information for each chip
"""

import typer

import pandas as pd
from pandas import HDFStore
import hdf5plugin
import h5py

import numpy as np
#import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from scipy import optimize
from typing import List, Dict, Tuple

from pathlib import Path


def gather_and_clean_data(input_file : Path) -> Dict[str, pd.DataFrame]:
    """Get the data that we want and clean it up a bit"""

    vin_a_tti_cur = pd.read_hdf(input_file, "shunt_curves/VIN_A_tti_cur")
    vin_d_tti_cur = pd.read_hdf(input_file, "shunt_curves/VIN_D_tti_cur")
    vin_a_int = pd.read_hdf(input_file, "shunt_curves/VIN_A_int")
    vin_d_int = pd.read_hdf(input_file, "shunt_curves/VIN_D_int")
    vin_a = pd.read_hdf(input_file, "shunt_curves/VIN_A")
    vin_d = pd.read_hdf(input_file, "shunt_curves/VIN_D")
    vofs = pd.read_hdf(input_file, "shunt_curves/VOFS")
    gnd_d = pd.read_hdf(input_file, "shunt_curves/GNDD")
    gnd_a = pd.read_hdf(input_file, "shunt_curves/GNDA")
    data = {
        "vin_a_tti_cur": vin_a_tti_cur,
        "vin_d_tti_cur": vin_d_tti_cur,
        "vin_a_int": vin_a_int,
        "vin_d_int": vin_d_int,
        "vin_a": vin_a,
        "vin_d": vin_d,
        "vofs": vofs,
    }

    # properly decode the chip_sn column data as string from bytes
    def decode_chip_sn(df: pd.DataFrame) -> None:
        if "chip_sn" in df:
            df["chip_sn"] = df["chip_sn"].apply(lambda sn: str(sn.replace("\x00", "")))
    for k, v in data.items() :
        decode_chip_sn(v)

    # keep hold of the list of chip sn
    chip_sn_list = list(data["vofs"]["chip_sn"])

    # drop non-data columns
    def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop("chip_sn", axis = 1).drop("name", axis = 1)

    for k, v in data.items() :
        data[k] = drop_columns(v)

    return data, chip_sn_list

def compute_k_factors(data: Dict[str, pd.DataFrame], chip_sn_list : List[str], use_internal : bool = False) -> pd.DataFrame:
    """Compute k-factors from a single point"""
    
    # get the relevant data
    vin_a_tti_cur = data["vin_a_tti_cur"]
    vin_d_tti_cur = data["vin_d_tti_cur"]
    vin_a_int = data["vin_a_int"]
    vin_d_int = data["vin_d_int"]
    vin_a = data["vin_a"]
    vin_d = data["vin_d"]
    vofs = data["vofs"]
        
    # constants
    rext = 600 # ohm
    shunt_interval = 0.2

    # for internal (VMUX) measurements, VIN is 1/4
    vin_d_select = vin_d_int*4 if use_internal else vin_d
    vin_a_select = vin_a_int*4 if use_internal else vin_a
        
    # compute digital k-factor
    idx_sel_digital = abs(vin_d_select - 1.6) < shunt_interval
    dig_numerator = rext * vin_d_tti_cur[idx_sel_digital]
    dig_denominator = vin_d_select[idx_sel_digital] - vofs[idx_sel_digital]*2
    k_dig = dig_numerator / dig_denominator
    k_dig["k_dig"] = k_dig.mean(axis = 1) # compute the row average
    
    # compute analog k-factor
    idx_sel_analog = abs(vin_a_select - 1.6) < shunt_interval
    ana_numerator = rext * vin_a_tti_cur[idx_sel_analog]
    ana_denominator = vin_a_select[idx_sel_analog] - vofs[idx_sel_analog]*2
    k_ana = ana_numerator / ana_denominator
    k_ana["k_ana"] = k_ana.mean(axis = 1) # compute the row average
    
    # add back the chip SN columns
    k_dig["chip_sn"] = chip_sn_list
    k_ana["chip_sn"] = chip_sn_list
    
    # drop data columns, keep only k-factor info
    k_dig = k_dig[["chip_sn", "k_dig"]]
    k_ana = k_ana[["chip_sn", "k_ana"]]

    k = k_dig
    k["k_ana"] = k_ana["k_ana"]
    
    return k

def compute_k_factors_from_fit(data: Dict[str, pd.DataFrame], chip_sn_list: List[str], use_internal: bool = False) -> pd.DataFrame:
    """Compute k-factors from a linear fit of the VI curves"""
    
    # linear fit function
    def linear_func(x, m, b) :
        return m * x + b
    
    def inverse_at(y, m, b) :
        return (y - b) / m
    
    # get the relevant data
    vin_a_tti_cur = data["vin_a_tti_cur"]
    vin_d_tti_cur = data["vin_d_tti_cur"]
    vin_a_int = data["vin_a_int"]
    vin_d_int = data["vin_d_int"]
    vin_a = data["vin_a"]
    vin_d = data["vin_d"]
    vofs = data["vofs"]
    
    # constants
    rext = 600 # ohm

    # for internal (VMUX) measurements, VIN is 1/4
    vin_d_select = vin_d_int*4 if use_internal else vin_d
    vin_a_select = vin_a_int*4 if use_internal else vin_a
        
    # compute k-factors from linear fit
    k_anas = []
    k_digs = []
    for ichip, chip_sn in enumerate(chip_sn_list) :
        
        # digital k-factor fit
        current = vin_d_tti_cur.iloc[ichip]
        vin = vin_d_select.iloc[ichip]
        (dig_slope, dig_intercept), _ = curve_fit(linear_func, current, vin)        
        current_at_1p6v = inverse_at(1.6, dig_slope, dig_intercept)
        k_dig = rext * current_at_1p6v / (1.6 - vofs.iloc[ichip].mean() * 2)
        k_digs.append(k_dig)
        
        # analog k-factor fit
        current = vin_a_tti_cur.iloc[ichip]
        vin = vin_a_select.iloc[ichip]
        (ana_slope, ana_intercept), _ = curve_fit(linear_func, current, vin)
        current_at_1p6v = inverse_at(1.6, ana_slope, ana_intercept)
        k_ana = rext * current_at_1p6v / (1.6 - vofs.iloc[ichip].mean() * 2)
        k_anas.append(k_ana)
        
    k_dig = pd.DataFrame({"chip_sn": chip_sn_list, "k_dig": k_digs})
    k_ana = pd.DataFrame({"chip_sn": chip_sn_list, "k_ana": k_anas})

    k = k_dig
    k["k_ana"] = k_ana["k_ana"]
    
    return k


def main(
        input_file: str = typer.Argument(..., help = "Input wafer probing HDF5 file")
        ):

    input_file = Path(input_file)
    if not input_file.exists() or not input_file.is_file() :
        print(f"ERROR: Could not find provided input \"{input_file}\"")
        sys.exit(1)

    data, chip_sn_list = gather_and_clean_data(input_file)

    # get the k-factors as computed by BDAQ/Bonn (using VIN_D, VIN_A)
    k_bonn = compute_k_factors(data, chip_sn_list)
    k_bonn.to_csv("k_factors_bonn.csv")

    # get the k-factors as computed by BDAQ/Bonn BUT using internal VIN measurements (VIN_D_int, VIN_A_int)
    k_int = compute_k_factors(data, chip_sn_list, use_internal = True)
    k_int.to_csv("k_factors_int.csv")

    # get the k-factors from performing linear fit (using VIN_D, VIN_A)
    k_bonn_fit = compute_k_factors_from_fit(data, chip_sn_list)
    k_bonn_fit.to_csv("k_factors_bonn_fit.csv")


    # get the k-factors from performing linear fit BUT using internal VIN measurementts (VIN_D_int, VIN_A_int)
    k_int_fit = compute_k_factors_from_fit(data, chip_sn_list, use_internal = True)
    k_int_fit.to_csv("k_factors_int_fit.csv")

if __name__ == "__main__" :
    typer.run(main)
