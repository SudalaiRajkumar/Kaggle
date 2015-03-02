# -*- coding: utf-8 -*-
"""
Created on Tue Mar 02 12:12:45 2015

@author: Sudalai Rajkumar S
"""

import sys
import csv
import numpy as np
import pandas as pd

def getSeasonStats(season_file):
	season_file_handle = open(season_file, 'r')
	reader = csv.DictReader(season_file_handle)
	out_dict = {}

	for row in reader:
		season_dict = out_dict.get(row['season'], {})
		wteam_dict = season_dict.get(row['wteam'], {})
		lteam_dict = season_dict.get(row['lteam'], {})
		
		wteam_dict['NoOfWins'] = wteam_dict.get('NoOfWins',0) + 1
		wteam_dict['NoOfGames'] = wteam_dict.get('NoOfGames',0) + 1
		wteam_dict['TotalScore'] = wteam_dict.get('TotalScore',0) + int(row['wscore'])
		wteam_dict['NumOT'] = wteam_dict.get('NumOT',0) + int(row['numot'])
		if row['wloc'] == "H":
			wteam_dict['NoOfHomeWins'] = wteam_dict.get('NoOfHomeWins',0) + 1
		elif row['wloc'] == 'N':
			wteam_dict['NoOfNeutralWins'] = wteam_dict.get('NoOfNeutralWins',0) + 1
		elif row['wloc'] == 'A':
                        wteam_dict['NoOfAwayWins'] = wteam_dict.get('NoOfAwayWins',0) + 1
		else:
			print row['wloc']
			sys.exit()
		wteam_dict['fgm'] = wteam_dict.get('fgm',0) + int(row['wfgm'])
		wteam_dict['fga'] = wteam_dict.get('fga',0) + int(row['wfga'])
		wteam_dict['fgm3'] = wteam_dict.get('fgm3',0) + int(row['wfgm3'])
                wteam_dict['fga3'] = wteam_dict.get('fga3',0) + int(row['wfga3'])
		wteam_dict['ftm'] = wteam_dict.get('ftm',0) + int(row['wftm'])
                wteam_dict['fta'] = wteam_dict.get('fta',0) + int(row['wfta'])
		wteam_dict['or'] = wteam_dict.get('or',0) + int(row['wor'])
                wteam_dict['dr'] = wteam_dict.get('dr',0) + int(row['wdr'])
		wteam_dict['ast'] = wteam_dict.get('ast',0) + int(row['wast'])
                wteam_dict['to'] = wteam_dict.get('to',0) + int(row['wto'])
                wteam_dict['pf'] = wteam_dict.get('pf',0) + int(row['wpf'])
		wteam_dict['stl'] = wteam_dict.get('stl',0) + int(row['wstl'])
                wteam_dict['blk'] = wteam_dict.get('blk',0) + int(row['wblk'])
		
		


		lteam_dict['NoOfLoss'] = lteam_dict.get('NoOfLoss',0) + 1
                lteam_dict['NoOfGames'] = lteam_dict.get('NoOfGames',0) + 1
                lteam_dict['TotalScore'] = lteam_dict.get('TotalScore',0) + int(row['lscore'])
		lteam_dict['NumOT'] = lteam_dict.get('NumOT',0) + int(row['numot'])
		if row['wloc'] == "H":
                        lteam_dict['NoOfAwayLoss'] = lteam_dict.get('NoOfAwayLoss',0) + 1
                elif row['wloc'] == 'N':
                        lteam_dict['NoOfNeutralLoss'] = lteam_dict.get('NoOfNeutralLoss',0) + 1
                elif row['wloc'] == 'A':
                        lteam_dict['NoOfHomeLoss'] = lteam_dict.get('NoOfHomeLoss',0) + 1
                else:
                        print row['wloc']
                        sys.exit()
		lteam_dict['fgm'] = lteam_dict.get('fgm',0) + int(row['lfgm'])
                lteam_dict['fga'] = lteam_dict.get('fga',0) + int(row['lfga'])
                lteam_dict['fgm3'] = lteam_dict.get('fgm3',0) + int(row['lfgm3'])
                lteam_dict['fga3'] = lteam_dict.get('fga3',0) + int(row['lfga3'])
                lteam_dict['ftm'] = lteam_dict.get('ftm',0) + int(row['lftm'])
                lteam_dict['fta'] = lteam_dict.get('fta',0) + int(row['lfta'])
                lteam_dict['or'] = lteam_dict.get('or',0) + int(row['lor'])
                lteam_dict['dr'] = lteam_dict.get('dr',0) + int(row['ldr'])
                lteam_dict['ast'] = lteam_dict.get('ast',0) + int(row['last'])
                lteam_dict['to'] = lteam_dict.get('to',0) + int(row['lto'])
                lteam_dict['pf'] = lteam_dict.get('pf',0) + int(row['lpf'])
                lteam_dict['stl'] = lteam_dict.get('stl',0) + int(row['lstl'])
		lteam_dict['blk'] = lteam_dict.get('blk',0) + int(row['lblk'])
                lteam_dict['blk'] = lteam_dict.get('blk',0) + int(row['lblk'])


		season_dict[row['wteam']] = wteam_dict
		season_dict[row['lteam']] = lteam_dict
		out_dict[row['season']] = season_dict

	#print out_dict['1985']['1228']
	#print out_dict['1985']['1328']
	#print out_dict['1985'].keys()
	#print len(out_dict['1985'].keys())
	return out_dict

def prepareTrainData(tourney_results_file, season_dict, seeds_dict):
	tourney_file_handle = open(tourney_results_file, 'r')
	reader = csv.DictReader(tourney_file_handle)
	header_list = ['WinPercentage','LossPercentage','AverageScore','HomeWinPerc','AwayWinPerc','NeutralWinPerc','HomeLossPerc','AwayLossPerc','NeutralLossPerc','NumOTPerc', 'NoOfGames',    'OppWinPercentage','OppLossPercentage','OppAverageScore','OppHomeWinPerc','OppAwayWinPerc','OppNeutralWinPerc','OppHomeLossPerc','OppAwayLossPerc','OppNeutralLossPerc', 'OppNumOTPerc', 'OppNoOfGames', 'fgm','fga','fgm3','fga3','ftm','fta','or','dr','to','pf','ast','stl','blk', 'ofgm','ofga','ofgm3','ofga3','oftm','ofta','oor','odr','oto','opf','oast','ostl','oblk',  'DV']

	out_list = []
	for row in reader:
		season = row['season']
		if season == '2011':
			break

		# Get the winning and losing team #
		wteam = row['wteam']
		lteam = row['lteam']
		# Get the stats of both teams #
		season_wteam_dict = season_dict[season][wteam]
		season_lteam_dict = season_dict[season][lteam]

		# Get the seeds of both teams #
		#seed_wteam = seeds_dict[season][wteam]
		#seed_lteam = seeds_dict[season][lteam]
		# Get win percentage for both teams #
		win_perc_wteam = season_wteam_dict.get('NoOfWins',0) / float(season_wteam_dict['NoOfGames'])
		win_perc_lteam = season_lteam_dict.get('NoOfWins',0) / float(season_lteam_dict['NoOfGames'])
		# Get Loss percentage for both teams #
		loss_perc_wteam = season_wteam_dict.get('NoOfLoss',0) / float(season_wteam_dict['NoOfGames'])
		loss_perc_lteam = season_lteam_dict.get('NoOfLoss',0) / float(season_lteam_dict['NoOfGames'])
		# Get the average score for both teams #
		avg_score_wteam = season_wteam_dict.get('TotalScore',0) / float(season_wteam_dict['NoOfGames'])
		avg_score_lteam = season_lteam_dict.get('TotalScore',0) / float(season_lteam_dict['NoOfGames'])
		# Get the home win percentage for both teams #
		home_win_perc_wteam = season_wteam_dict.get('NoOfHomeWins',0) / float(season_wteam_dict['NoOfGames'])
		home_win_perc_lteam = season_lteam_dict.get('NoOfHomeWins',0) / float(season_lteam_dict['NoOfGames'])
		# Get the Away win percentage for both teams #
		away_win_perc_wteam = season_wteam_dict.get('NoOfAwayWins',0) / float(season_wteam_dict['NoOfGames'])
		away_win_perc_lteam = season_lteam_dict.get('NoOfAwayWins',0) / float(season_lteam_dict['NoOfGames'])
		# Get the neutral win percentage #
		neutral_win_perc_wteam = season_wteam_dict.get('NoOfNeutralWins',0) / float(season_wteam_dict['NoOfGames'])
		neutral_win_perc_lteam = season_lteam_dict.get('NoOfNeutralWins',0) / float(season_lteam_dict['NoOfGames'])
		# Get the home loss percentage for both teams #
                home_loss_perc_wteam = season_wteam_dict.get('NoOfHomeLoss',0) / float(season_wteam_dict['NoOfGames'])
                home_loss_perc_lteam = season_lteam_dict.get('NoOfHomeLoss',0) / float(season_lteam_dict['NoOfGames'])
                # Get the Away loss percentage for both teams #
                away_loss_perc_wteam = season_wteam_dict.get('NoOfAwayLoss',0) / float(season_wteam_dict['NoOfGames'])
                away_loss_perc_lteam = season_lteam_dict.get('NoOfAwayLoss',0) / float(season_lteam_dict['NoOfGames'])
                # Get the neutral loss percentage #
                neutral_loss_perc_wteam = season_wteam_dict.get('NoOfNeutralLoss',0) / float(season_wteam_dict['NoOfGames'])
                neutral_loss_perc_lteam = season_lteam_dict.get('NoOfNeutralLoss',0) / float(season_lteam_dict['NoOfGames'])
		# Get the number of overtime matches for both teams #
		ot_wteam = season_wteam_dict['NumOT'] 
                ot_lteam = season_lteam_dict['NumOT'] 
		# Get the number of matches for both teams #
                num_games_wteam = season_wteam_dict['NoOfGames']
                num_games_lteam = season_lteam_dict['NoOfGames']

		fgm_wteam = season_wteam_dict.get('fgm',0) / float(season_wteam_dict['NoOfGames'])
		fga_wteam = season_wteam_dict.get('fga',0) / float(season_wteam_dict['NoOfGames'])
		fgm3_wteam = season_wteam_dict.get('fgm3',0) / float(season_wteam_dict['NoOfGames'])
                fga3_wteam = season_wteam_dict.get('fga3',0) / float(season_wteam_dict['NoOfGames'])
		ftm_wteam = season_wteam_dict.get('ftm',0) / float(season_wteam_dict['NoOfGames'])
                fta_wteam = season_wteam_dict.get('fta',0) / float(season_wteam_dict['NoOfGames'])
		or_wteam = season_wteam_dict.get('or',0) / float(season_wteam_dict['NoOfGames'])
                dr_wteam = season_wteam_dict.get('dr',0) / float(season_wteam_dict['NoOfGames'])
		to_wteam = season_wteam_dict.get('to',0) / float(season_wteam_dict['NoOfGames'])
                pf_wteam = season_wteam_dict.get('pf',0) / float(season_wteam_dict['NoOfGames'])
		ast_wteam = season_wteam_dict.get('ast',0) / float(season_wteam_dict['NoOfGames'])
                stl_wteam = season_wteam_dict.get('stl',0) / float(season_wteam_dict['NoOfGames'])
		blk_wteam = season_wteam_dict.get('blk',0) / float(season_wteam_dict['NoOfGames'])

		fgm_lteam = season_lteam_dict.get('fgm',0) / float(season_lteam_dict['NoOfGames'])
                fga_lteam = season_lteam_dict.get('fga',0) / float(season_lteam_dict['NoOfGames'])
                fgm3_lteam = season_lteam_dict.get('fgm3',0) / float(season_lteam_dict['NoOfGames'])
                fga3_lteam = season_lteam_dict.get('fga3',0) / float(season_lteam_dict['NoOfGames'])
                ftm_lteam = season_lteam_dict.get('ftm',0) / float(season_lteam_dict['NoOfGames'])
                fta_lteam = season_lteam_dict.get('fta',0) / float(season_lteam_dict['NoOfGames'])
                or_lteam = season_lteam_dict.get('or',0) / float(season_lteam_dict['NoOfGames'])
                dr_lteam = season_lteam_dict.get('dr',0) / float(season_lteam_dict['NoOfGames'])
                to_lteam = season_lteam_dict.get('to',0) / float(season_lteam_dict['NoOfGames'])
                pf_lteam = season_lteam_dict.get('pf',0) / float(season_lteam_dict['NoOfGames'])
                ast_lteam = season_lteam_dict.get('ast',0) / float(season_lteam_dict['NoOfGames'])
                stl_lteam = season_lteam_dict.get('stl',0) / float(season_lteam_dict['NoOfGames'])
                blk_lteam = season_lteam_dict.get('blk',0) / float(season_lteam_dict['NoOfGames'])

		

		# Appending the features to out list #
		out_list.append([ win_perc_wteam, loss_perc_wteam, avg_score_wteam, home_win_perc_wteam, away_win_perc_wteam, neutral_win_perc_wteam, home_loss_perc_wteam, away_loss_perc_wteam, neutral_loss_perc_wteam, ot_wteam, num_games_wteam, win_perc_lteam, loss_perc_lteam, avg_score_lteam, home_win_perc_lteam, away_win_perc_lteam, neutral_win_perc_lteam, home_loss_perc_lteam, away_loss_perc_lteam, neutral_loss_perc_lteam, ot_lteam, num_games_lteam,   fgm_wteam,fga_wteam,fgm3_wteam,fga3_wteam,ftm_wteam,fta_wteam,or_wteam,dr_wteam,to_wteam,pf_wteam,ast_wteam,stl_wteam,blk_wteam,  fgm_lteam,fga_lteam,fgm3_lteam,fga3_lteam,ftm_lteam,fta_lteam,or_lteam,dr_lteam,to_lteam,pf_lteam,ast_lteam,stl_lteam,blk_lteam,  1])
		out_list.append([win_perc_lteam, loss_perc_lteam, avg_score_lteam, home_win_perc_lteam, away_win_perc_lteam, neutral_win_perc_lteam, home_loss_perc_lteam, away_loss_perc_lteam, neutral_loss_perc_lteam, ot_lteam, num_games_lteam, win_perc_wteam, loss_perc_wteam, avg_score_wteam, home_win_perc_wteam, away_win_perc_wteam, neutral_win_perc_wteam, home_loss_perc_wteam, away_loss_perc_wteam, neutral_loss_perc_wteam, ot_wteam, num_games_wteam,   fgm_lteam,fga_lteam,fgm3_lteam,fga3_lteam,ftm_lteam,fta_lteam,or_lteam,dr_lteam,to_lteam,pf_lteam,ast_lteam,stl_lteam,blk_lteam,   fgm_wteam,fga_wteam,fgm3_wteam,fga3_wteam,ftm_wteam,fta_wteam,or_wteam,dr_wteam,to_wteam,pf_wteam,ast_wteam,stl_wteam,blk_wteam,  0])

	out_df = pd.DataFrame(np.array(out_list))
	out_df.columns = header_list
	return out_df

def prepareTestData(fixture_file, season_dict, seeds_dict):
	fixture_file_handle = open(fixture_file, 'r')
        reader = csv.DictReader(fixture_file_handle)
        header_list = ['id', 'WinPercentage','LossPercentage','AverageScore','HomeWinPerc','AwayWinPerc','NeutralWinPerc','HomeLossPerc','AwayLossPerc','NeutralLossPerc','NumOTPerc', 'NoOfGames',    'OppWinPercentage','OppLossPercentage','OppAverageScore','OppHomeWinPerc','OppAwayWinPerc','OppNeutralWinPerc','OppHomeLossPerc','OppAwayLossPerc','OppNeutralLossPerc', 'OppNumOTPerc', 'OppNoOfGames', 'fgm','fga','fgm3','fga3','ftm','fta','or','dr','to','pf','ast','stl','blk', 'ofgm','ofga','ofgm3','ofga3','oftm','ofta','oor','odr','oto','opf','oast','ostl','oblk']

        out_list = []
	out_list = []
        for row in reader:
		id_val = row['id']
                season = id_val.split("_")[0]

                # Get the winning and losing team #
                wteam = id_val.split("_")[1]
                lteam = id_val.split("_")[2]
                # Get the stats of both teams #
                season_wteam_dict = season_dict[season][wteam]
                season_lteam_dict = season_dict[season][lteam]

                # Get the seeds of both teams #
                #seed_wteam = seeds_dict[season][wteam]
                #seed_lteam = seeds_dict[season][lteam]
                # Get win percentage for both teams #
                win_perc_wteam = season_wteam_dict.get('NoOfWins',0) / float(season_wteam_dict['NoOfGames'])
                win_perc_lteam = season_lteam_dict.get('NoOfWins',0) / float(season_lteam_dict['NoOfGames'])
                # Get Loss percentage for both teams #
                loss_perc_wteam = season_wteam_dict.get('NoOfLoss',0) / float(season_wteam_dict['NoOfGames'])
                loss_perc_lteam = season_lteam_dict.get('NoOfLoss',0) / float(season_lteam_dict['NoOfGames'])
                # Get the average score for both teams #
                avg_score_wteam = season_wteam_dict.get('TotalScore',0) / float(season_wteam_dict['NoOfGames'])
                avg_score_lteam = season_lteam_dict.get('TotalScore',0) / float(season_lteam_dict['NoOfGames'])
                # Get the home win percentage for both teams #
                home_win_perc_wteam = season_wteam_dict.get('NoOfHomeWins',0) / float(season_wteam_dict['NoOfGames'])
                home_win_perc_lteam = season_lteam_dict.get('NoOfHomeWins',0) / float(season_lteam_dict['NoOfGames'])
                # Get the Away win percentage for both teams #
                away_win_perc_wteam = season_wteam_dict.get('NoOfAwayWins',0) / float(season_wteam_dict['NoOfGames'])
                away_win_perc_lteam = season_lteam_dict.get('NoOfAwayWins',0) / float(season_lteam_dict['NoOfGames'])
                # Get the neutral win percentage #
                neutral_win_perc_wteam = season_wteam_dict.get('NoOfNeutralWins',0) / float(season_wteam_dict['NoOfGames'])
                neutral_win_perc_lteam = season_lteam_dict.get('NoOfNeutralWins',0) / float(season_lteam_dict['NoOfGames'])
                # Get the home loss percentage for both teams #
                home_loss_perc_wteam = season_wteam_dict.get('NoOfHomeLoss',0) / float(season_wteam_dict['NoOfGames'])
                home_loss_perc_lteam = season_lteam_dict.get('NoOfHomeLoss',0) / float(season_lteam_dict['NoOfGames'])
                # Get the Away loss percentage for both teams #
                away_loss_perc_wteam = season_wteam_dict.get('NoOfAwayLoss',0) / float(season_wteam_dict['NoOfGames'])
                away_loss_perc_lteam = season_lteam_dict.get('NoOfAwayLoss',0) / float(season_lteam_dict['NoOfGames'])
		# Get the neutral loss percentage #
                neutral_loss_perc_wteam = season_wteam_dict.get('NoOfNeutralLoss',0) / float(season_wteam_dict['NoOfGames'])
                neutral_loss_perc_lteam = season_lteam_dict.get('NoOfNeutralLoss',0) / float(season_lteam_dict['NoOfGames'])
                # Get the number of overtime matches for both teams #
                ot_wteam = season_wteam_dict['NumOT']
                ot_lteam = season_lteam_dict['NumOT']
                # Get the number of matches for both teams #
                num_games_wteam = season_wteam_dict['NoOfGames']
                num_games_lteam = season_lteam_dict['NoOfGames']

		fgm_wteam = season_wteam_dict.get('fgm',0) / float(season_wteam_dict['NoOfGames'])
                fga_wteam = season_wteam_dict.get('fga',0) / float(season_wteam_dict['NoOfGames'])
                fgm3_wteam = season_wteam_dict.get('fgm3',0) / float(season_wteam_dict['NoOfGames'])
                fga3_wteam = season_wteam_dict.get('fga3',0) / float(season_wteam_dict['NoOfGames'])
                ftm_wteam = season_wteam_dict.get('ftm',0) / float(season_wteam_dict['NoOfGames'])
                fta_wteam = season_wteam_dict.get('fta',0) / float(season_wteam_dict['NoOfGames'])
                or_wteam = season_wteam_dict.get('or',0) / float(season_wteam_dict['NoOfGames'])
                dr_wteam = season_wteam_dict.get('dr',0) / float(season_wteam_dict['NoOfGames'])
                to_wteam = season_wteam_dict.get('to',0) / float(season_wteam_dict['NoOfGames'])
                pf_wteam = season_wteam_dict.get('pf',0) / float(season_wteam_dict['NoOfGames'])
                ast_wteam = season_wteam_dict.get('ast',0) / float(season_wteam_dict['NoOfGames'])
                stl_wteam = season_wteam_dict.get('stl',0) / float(season_wteam_dict['NoOfGames'])
                blk_wteam = season_wteam_dict.get('blk',0) / float(season_wteam_dict['NoOfGames'])

                fgm_lteam = season_lteam_dict.get('fgm',0) / float(season_lteam_dict['NoOfGames'])
                fga_lteam = season_lteam_dict.get('fga',0) / float(season_lteam_dict['NoOfGames'])
                fgm3_lteam = season_lteam_dict.get('fgm3',0) / float(season_lteam_dict['NoOfGames'])
                fga3_lteam = season_lteam_dict.get('fga3',0) / float(season_lteam_dict['NoOfGames'])
                ftm_lteam = season_lteam_dict.get('ftm',0) / float(season_lteam_dict['NoOfGames'])
                fta_lteam = season_lteam_dict.get('fta',0) / float(season_lteam_dict['NoOfGames'])
                or_lteam = season_lteam_dict.get('or',0) / float(season_lteam_dict['NoOfGames'])
                dr_lteam = season_lteam_dict.get('dr',0) / float(season_lteam_dict['NoOfGames'])
                to_lteam = season_lteam_dict.get('to',0) / float(season_lteam_dict['NoOfGames'])
                pf_lteam = season_lteam_dict.get('pf',0) / float(season_lteam_dict['NoOfGames'])
                ast_lteam = season_lteam_dict.get('ast',0) / float(season_lteam_dict['NoOfGames'])
                stl_lteam = season_lteam_dict.get('stl',0) / float(season_lteam_dict['NoOfGames'])
                blk_lteam = season_lteam_dict.get('blk',0) / float(season_lteam_dict['NoOfGames'])


                # Appending the features to out list #
                out_list.append([id_val, win_perc_wteam, loss_perc_wteam, avg_score_wteam, home_win_perc_wteam, away_win_perc_wteam, neutral_win_perc_wteam, home_loss_perc_wteam, away_loss_perc_wteam, neutral_loss_perc_wteam, ot_wteam, num_games_wteam, win_perc_lteam, loss_perc_lteam, avg_score_lteam, home_win_perc_lteam, away_win_perc_lteam, neutral_win_perc_lteam, home_loss_perc_lteam, away_loss_perc_lteam, neutral_loss_perc_lteam, ot_lteam, num_games_lteam, fgm_wteam,fga_wteam,fgm3_wteam,fga3_wteam,ftm_wteam,fta_wteam,or_wteam,dr_wteam,to_wteam,pf_wteam,ast_wteam,stl_wteam,blk_wteam,  fgm_lteam,fga_lteam,fgm3_lteam,fga3_lteam,ftm_lteam,fta_lteam,or_lteam,dr_lteam,to_lteam,pf_lteam,ast_lteam,stl_lteam,blk_lteam])

        out_df = pd.DataFrame(np.array(out_list))
        out_df.columns = header_list
        return out_df



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
	regular_seasons_file = data_path + "regular_season_detailed_results.csv"
	tourney_seeds_file= data_path + "tourney_seeds.csv"
	tourney_results_file = data_path + "tourney_detailed_results.csv"
	test_fixture_file = data_path + "sample_submission.csv"
	train_file = data_path + "train_v4.csv"
	test_file = data_path + "test_v4.csv"

	season_dict = getSeasonStats(regular_seasons_file)
	seeds_dict = getSeedStats(tourney_seeds_file)

	for year in season_dict.keys():
		for team in seeds_dict[year]:
			season_dict[year][team]	

	train_df = prepareTrainData(tourney_results_file, season_dict, seeds_dict)
	train_df.to_csv(train_file, index=False)

	test_df = prepareTestData(test_fixture_file, season_dict, seeds_dict)
	test_df.to_csv(test_file, index=False)
