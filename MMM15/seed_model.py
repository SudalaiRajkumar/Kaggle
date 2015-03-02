# -*- coding: utf-8 -*-
"""
Created on Tue Mar 02 10:17:40 2015

@author: Sudalai Rajkumar S

Module to produce the seed based bench mark given in the competition
"""

import csv
import numpy as np
import pandas as pd

def getSeedStats(seeds_file):
        seeds_file_handle = open(seeds_file, 'r')
        reader = csv.DictReader(seeds_file_handle)
        out_dict = {}

        for row in reader:
                season_dict = out_dict.get(row['season'], {})
                season_dict[row['team']] = int(row['seed'][1:3])
                out_dict[row['season']] = season_dict

        return out_dict

if __name__ == "__main__":
        data_path = "/home/sudalai/Others/Kaggle/MMM15/Data/"
        test_fixture_file = data_path + "sample_submission.csv"
        tourney_seeds_file= data_path + "tourney_seeds.csv"
        sub_file = open("sub_seedmodel.csv","w")
        sub_file.write('id,pred\n')

        seeds_dict = getSeedStats(tourney_seeds_file)

        test_file_handle = open(test_fixture_file,'r')
        reader = csv.DictReader(test_file_handle)
        for row in reader:
                id_val = row['id']
                season = id_val.split("_")[0]
                fteam = id_val.split("_")[1]
                steam = id_val.split("_")[2]

                fteam_seed = seeds_dict[season][fteam]
                steam_seed = seeds_dict[season][steam]

                pred_val = 0.5 + ((steam_seed - fteam_seed)*0.03)

                sub_file.write(str(id_val) + "," + str(pred_val) + "\n")
        sub_file.close()
