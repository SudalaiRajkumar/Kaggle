import csv
import numpy as np
import pandas as pd

from config_v5 import fineline_dict, header_list4, upc_dict, header_list5

map_type_dv_dict = {3:0, 4:1, 5:2, 6:3, 7:4, 8:5, 9:6, 12:7, 14:8, 15:9, 18:10, 19:11, 20:12, 21:13, 22:14, 23:15, 24:16, 25:17, 26:18, 27:19, 28:20, 29:21, 30:22, 31:23, 32:24, 33:25, 34:26, 35:27, 36:28, 37:29, 38:30, 39:31, 40:32, 41:33, 42:34, 43:35, 44:36, 999:37}
map_dept_dict = {'COMM BREAD': 14, 'OPTICAL - FRAMES': 47, '1-HR PHOTO': 1, 'LIQUOR,WINE,BEER': 41, 'FABRICS AND CRAFTS': 20, 'MENS WEAR': 44, 'SEAFOOD': 59, 'AUTOMOTIVE': 3, 'BEDDING': 7, 'COOK AND DINE': 16, 'OPTICAL - LENSES': 48, 'HARDWARE': 26, 'SLEEPWEAR/FOUNDATIONS': 64, 'FINANCIAL SERVICES': 21, 'OTHER DEPARTMENTS': 49, 'ELECTRONICS': 19, 'LADIESWEAR': 38, 'HOME MANAGEMENT': 29, 'HOUSEHOLD PAPER GOODS': 32, 'FROZEN FOODS': 22, 'FURNITURE': 23, 'INFANT CONSUMABLE HARDLINES': 35, 'MENSWEAR': 45, 'PAINT AND ACCESSORIES': 50, 'GROCERY DRY GOODS': 25, 'BOYS WEAR': 9, 'SERVICE DELI': 61, 'ACCESSORIES': 2, 'DSD GROCERY': 18, 'MEDIA AND GAMING': 43, -999: 0, 'JEWELRY AND SUNGLASSES': 36, 'PLUS AND MATERNITY': 56, 'LARGE HOUSEHOLD GOODS': 39, 'HOUSEHOLD CHEMICALS/SUPP': 31, 'CAMERAS AND SUPPLIES': 11, 'BATH AND SHOWER': 5, 'SEASONAL': 60, 'IMPULSE MERCHANDISE': 33, 'BRAS & SHAPEWEAR': 10, 'PHARMACY OTC': 53, 'SPORTING GOODS': 65, 'BEAUTY': 6, 'PETS AND SUPPLIES': 52, 'LADIES SOCKS': 37, 'HOME DECOR': 28, 'WIRELESS': 68, 'DAIRY': 17, 'PERSONAL CARE': 51, 'TOYS': 67, 'CONCEPT STORES': 15, 'HEALTH AND BEAUTY AIDS': 27, 'OFFICE SUPPLIES': 46, 'LAWN AND GARDEN': 40, 'SHOES': 63, 'SHEER HOSIERY': 62, 'PRE PACKED DELI': 57, 'INFANT APPAREL': 34, 'HORTICULTURE AND ACCESS': 30, 'PLAYERS AND ELECTRONICS': 55, 'BAKERY': 4, 'PRODUCE': 58, 'CANDY, TOBACCO, COOKIES': 12, 'MEAT - FRESH & FROZEN': 42, 'PHARMACY RX': 54, 'BOOKS AND MAGAZINES': 8, 'GIRLS WEAR, 4-6X  AND 7-14': 24, 'SWIMWEAR/OUTERWEAR': 66, 'CELEBRATION': 13}
weekday_dict = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}

def getHeader(train):
	header_list1 = ["VisitNumber", "DayOfWeek", "NumberOfRows", "NoOfUPCs", "NumberOfItems", "NumberOfDepts", "NumberOfFineLine" ]
	header_list2 = ['Dept_-999', 'Dept_1-HR_PHOTO', 'Dept_ACCESSORIES', 'Dept_AUTOMOTIVE', 'Dept_BAKERY', 'Dept_BATH_AND_SHOWER', 'Dept_BEAUTY', 'Dept_BEDDING', 'Dept_BOOKS_AND_MAGAZINES', 'Dept_BOYS_WEAR', 'Dept_BRAS_&_SHAPEWEAR', 'Dept_CAMERAS_AND_SUPPLIES', 'Dept_CANDY,_TOBACCO,_COOKIES', 'Dept_CELEBRATION', 'Dept_COMM_BREAD', 'Dept_CONCEPT_STORES', 'Dept_COOK_AND_DINE', 'Dept_DAIRY', 'Dept_DSD_GROCERY', 'Dept_ELECTRONICS', 'Dept_FABRICS_AND_CRAFTS', 'Dept_FINANCIAL_SERVICES', 'Dept_FROZEN_FOODS', 'Dept_FURNITURE', 'Dept_GIRLS_WEAR,_4-6X__AND_7-14', 'Dept_GROCERY_DRY_GOODS', 'Dept_HARDWARE', 'Dept_HEALTH_AND_BEAUTY_AIDS', 'Dept_HOME_DECOR', 'Dept_HOME_MANAGEMENT', 'Dept_HORTICULTURE_AND_ACCESS', 'Dept_HOUSEHOLD_CHEMICALS/SUPP', 'Dept_HOUSEHOLD_PAPER_GOODS', 'Dept_IMPULSE_MERCHANDISE', 'Dept_INFANT_APPAREL', 'Dept_INFANT_CONSUMABLE_HARDLINES', 'Dept_JEWELRY_AND_SUNGLASSES', 'Dept_LADIES_SOCKS', 'Dept_LADIESWEAR', 'Dept_LARGE_HOUSEHOLD_GOODS', 'Dept_LAWN_AND_GARDEN', 'Dept_LIQUOR,WINE,BEER', 'Dept_MEAT_-_FRESH_&_FROZEN', 'Dept_MEDIA_AND_GAMING', 'Dept_MENS_WEAR', 'Dept_MENSWEAR', 'Dept_OFFICE_SUPPLIES', 'Dept_OPTICAL_-_FRAMES', 'Dept_OPTICAL_-_LENSES', 'Dept_OTHER_DEPARTMENTS', 'Dept_PAINT_AND_ACCESSORIES', 'Dept_PERSONAL_CARE', 'Dept_PETS_AND_SUPPLIES', 'Dept_PHARMACY_OTC', 'Dept_PHARMACY_RX', 'Dept_PLAYERS_AND_ELECTRONICS', 'Dept_PLUS_AND_MATERNITY', 'Dept_PRE_PACKED_DELI', 'Dept_PRODUCE', 'Dept_SEAFOOD', 'Dept_SEASONAL', 'Dept_SERVICE_DELI', 'Dept_SHEER_HOSIERY', 'Dept_SHOES', 'Dept_SLEEPWEAR/FOUNDATIONS', 'Dept_SPORTING_GOODS', 'Dept_SWIMWEAR/OUTERWEAR', 'Dept_TOYS', 'Dept_WIRELESS']
	#header_list3 = ["MinCountUPC", "MaxCountUPC", "MeanCountUPC", 'DeptScan_-999', 'DeptScan_1-HR_PHOTO', 'DeptScan_ACCESSORIES', 'DeptScan_AUTOMOTIVE', 'DeptScan_BAKERY', 'DeptScan_BATH_AND_SHOWER', 'DeptScan_BEAUTY', 'DeptScan_BEDDING', 'DeptScan_BOOKS_AND_MAGAZINES', 'DeptScan_BOYS_WEAR', 'DeptScan_BRAS_&_SHAPEWEAR', 'DeptScan_CAMERAS_AND_SUPPLIES', 'DeptScan_CANDY,_TOBACCO,_COOKIES', 'DeptScan_CELEBRATION', 'DeptScan_COMM_BREAD', 'DeptScan_CONCEPT_STORES', 'DeptScan_COOK_AND_DINE', 'DeptScan_DAIRY', 'DeptScan_DSD_GROCERY', 'DeptScan_ELECTRONICS', 'DeptScan_FABRICS_AND_CRAFTS', 'DeptScan_FINANCIAL_SERVICES', 'DeptScan_FROZEN_FOODS', 'DeptScan_FURNITURE', 'DeptScan_GIRLS_WEAR,_4-6X__AND_7-14', 'DeptScan_GROCERY_DRY_GOODS', 'DeptScan_HARDWARE', 'DeptScan_HEALTH_AND_BEAUTY_AIDS', 'DeptScan_HOME_DECOR', 'DeptScan_HOME_MANAGEMENT', 'DeptScan_HORTICULTURE_AND_ACCESS', 'DeptScan_HOUSEHOLD_CHEMICALS/SUPP', 'DeptScan_HOUSEHOLD_PAPER_GOODS', 'DeptScan_IMPULSE_MERCHANDISE', 'DeptScan_INFANT_APPAREL', 'DeptScan_INFANT_CONSUMABLE_HARDLINES', 'DeptScan_JEWELRY_AND_SUNGLASSES', 'DeptScan_LADIES_SOCKS', 'DeptScan_LADIESWEAR', 'DeptScan_LARGE_HOUSEHOLD_GOODS', 'DeptScan_LAWN_AND_GARDEN', 'DeptScan_LIQUOR,WINE,BEER', 'DeptScan_MEAT_-_FRESH_&_FROZEN', 'DeptScan_MEDIA_AND_GAMING', 'DeptScan_MENS_WEAR', 'DeptScan_MENSWEAR', 'DeptScan_OFFICE_SUPPLIES', 'DeptScan_OPTICAL_-_FRAMES', 'DeptScan_OPTICAL_-_LENSES', 'DeptScan_OTHER_DEPARTMENTS', 'DeptScan_PAINT_AND_ACCESSORIES', 'DeptScan_PERSONAL_CARE', 'DeptScan_PETS_AND_SUPPLIES', 'DeptScan_PHARMACY_OTC', 'DeptScan_PHARMACY_RX', 'DeptScan_PLAYERS_AND_ELECTRONICS', 'DeptScan_PLUS_AND_MATERNITY', 'DeptScan_PRE_PACKED_DELI', 'DeptScan_PRODUCE', 'DeptScan_SEAFOOD', 'DeptScan_SEASONAL', 'DeptScan_SERVICE_DELI', 'DeptScan_SHEER_HOSIERY', 'DeptScan_SHOES', 'DeptScan_SLEEPWEAR/FOUNDATIONS', 'DeptScan_SPORTING_GOODS', 'DeptScan_SWIMWEAR/OUTERWEAR', 'DeptScan_TOYS', 'DeptScan_WIRELESS', 'RatioItemsUPC', 'RatioItemsDept', 'RatioItemsFineLine', 'NoItemsLessZero']
	header_list3 = ["MinCountUPC", "MaxCountUPC", "MeanCountUPC", 'RatioItemsUPC', 'RatioItemsDept', 'RatioItemsFineLine', 'NoItemsLessZero']
	
	header_list = header_list1 + header_list2 + header_list3 + header_list4 + header_list5

	if train:
		return header_list + ["DV"]
	else:
		return header_list

def getDeptCount(depts):
	dept_list = [0]*len(map_dept_dict.keys())
	for dept in depts:
		dept_no = map_dept_dict[dept]
		dept_list[dept_no] += 1
	return dept_list

fineline_len = len(fineline_dict.keys())
def getFineLineCount(finelines):
	fineline_list = [0]*fineline_len
	for fineline in finelines:
		fineline_no = fineline_dict.get(fineline,fineline_dict[-999])
		fineline_list[fineline_no] += 1
	return fineline_list

upc_len = len(upc_dict.keys())
def getUpcCount(upcs):
        upc_list = [0]*upc_len
        for upc in upcs:
                upc_no = upc_dict.get(upc,upc_dict[-999.0])
                upc_list[upc_no] += 1
        return upc_list

def getDeptScanCounts(depts, scans):
	dept_list = [0]*len(map_dept_dict.keys())
	for index, dept in enumerate(depts):
		dept_no = map_dept_dict[dept]
		dept_list[dept_no] += scans[index]
	return dept_list
		

def getVariables(name, grouped_df, train=0):
	try:
		out_list = [name, weekday_dict[np.array(grouped_df["Weekday"])[0]], grouped_df.shape[0]]
	except:
		raise

	no_upc = len( np.unique(grouped_df["Upc"]) )
	out_list.append(no_upc)

	no_items = int(np.sum(grouped_df["ScanCount"]) )
	out_list.append(no_items)

	no_depts = len( np.unique(grouped_df["DepartmentDescription"]) )
	out_list.append(no_depts)

	no_fineline = len( np.unique(grouped_df["FinelineNumber"]) )
	out_list.append(no_fineline)

	depts = grouped_df["DepartmentDescription"].tolist()
	out_list.extend( getDeptCount(depts) )

	min_count_in_upc = int(np.min(grouped_df["ScanCount"]))
	out_list.append(min_count_in_upc)

	max_count_in_upc = int(np.max(grouped_df["ScanCount"]))
        out_list.append(max_count_in_upc)

	mean_count_in_upc = int(np.mean(grouped_df["ScanCount"]))
        out_list.append(mean_count_in_upc)

	#scans = grouped_df["ScanCount"].tolist()
	#out_list.extend( getDeptScanCounts(depts, scans) )

	ratio_items_upc = no_items / no_upc
	out_list.append(ratio_items_upc)

	ratio_items_dept = no_items / no_depts
	out_list.append(ratio_items_dept)

	ratio_items_fineline = no_items / no_fineline
	out_list.append(ratio_items_fineline)

	no_items_less0 = np.sum( np.array(grouped_df["ScanCount"])<0 )
	out_list.append(no_items_less0)

	finelines = grouped_df["FinelineNumber"].tolist()
        out_list.extend( getFineLineCount(finelines) )

	upcs = grouped_df["Upc"].tolist()
        out_list.extend( getUpcCount(upcs) )

	if train:
		out_list.append( map_type_dv_dict[ np.array(grouped_df["TripType"])[0] ])

	return out_list
	

if __name__ == "__main__":
	data_path = "../Data/"
	train_file = data_path + "train.csv"
	test_file = data_path + "test.csv"
	train_out_file = data_path + "train_mod_v5.csv"
	test_out_file = data_path + "test_mod_v5.csv"
	train_dv_out_file = data_path + "train_mod_v5_dv.csv"

	train_df = pd.read_csv(train_file)
	test_df = pd.read_csv(test_file)
	train_out_handle = open(train_out_file, "w")
	test_out_handle = open(test_out_file, "w")
	train_dv_out_handle = open(train_dv_out_file, "w")
	train_writer = csv.writer(train_out_handle)
	test_writer = csv.writer(test_out_handle)
	train_dv_writer = csv.writer(train_dv_out_handle)

	train_df = train_df.fillna(-999)
	test_df = test_df.fillna(-999)

	train_header = getHeader(train=0)
	train_writer.writerow( train_header )
	test_header = getHeader(train=0)
	test_writer.writerow( test_header )
	train_dv_header = ["VisitNumber", "DV"]
	train_dv_writer.writerow(train_dv_header)
	train_header_len = len(train_header)
	test_header_len = len(test_header)
	train_dv_header_len = len(train_dv_header) 

	print "Processing train.."
	print train_df.shape
	grouped_train_df = train_df.groupby("VisitNumber")
	counter = 0
	for name, group in grouped_train_df:
		out_row = getVariables(name, group, train=1)
		dv = out_row[-1]
		out_row = out_row[:-1]
		dv_row = [name, dv]
		assert len(out_row) == train_header_len
		assert len(dv_row) == train_dv_header_len
		train_writer.writerow(out_row)
		train_dv_writer.writerow(dv_row)
		counter += 1
		if counter%10000 == 0:
			print counter

	print "Processing test.."
	grouped_test_df = test_df.groupby("VisitNumber")
	counter = 0
        for name, group in grouped_test_df:
                out_row = getVariables(name, group, train=0)
		assert len(out_row) == test_header_len
                test_writer.writerow(out_row)
                counter += 1
                if counter%10000 == 0:
                        print counter

	train_out_handle.close()
	test_out_handle.close()
	train_dv_out_handle.close()

