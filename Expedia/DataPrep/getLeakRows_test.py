from csv import DictReader
from collections import defaultdict
from datetime import datetime

start = datetime.now()

def get_top5(d):
    return sorted(d, key=d.get, reverse=True)[:5]

destination_clusters = defaultdict(lambda: defaultdict(int))
destination_clusters2 = defaultdict(lambda: defaultdict(int))
destination_clusters3 = defaultdict(lambda: defaultdict(int))
destination_clusters4 = defaultdict(lambda: defaultdict(int))

print "Reading the train.."
for i, row in enumerate(DictReader(open("../../Data/train.csv"))):
	key = row["user_location_country"] + "_"  + row["user_location_region"] + "_" + row["user_location_city"] + "_" + row["hotel_market"] + "_"+ row["orig_destination_distance"]
	#key2 = row["user_id"] + "_" + row["srch_destination_id"]
	#key3 = row["srch_destination_id"] + "_" + row["hotel_market"]
	#key4 = row["hotel_market"]
	destination_clusters[key][row["hotel_cluster"]] += 1
	#destination_clusters2[key2][row["hotel_cluster"]] += 1
	#destination_clusters3[key3][row["hotel_cluster"]] += 1
	#destination_clusters4[key4][row["hotel_cluster"]] += 1
	if i % 1000000 == 0:
		print("%s\t%s"%(i, datetime.now() - start))

most_frequent = defaultdict(str)
most_frequent2 = defaultdict(str)
most_frequent3 = defaultdict(str)
most_frequent4 = defaultdict(str)

print "Getting top 5 list.."
for k in destination_clusters:
        top5_list = get_top5(destination_clusters[k])
        most_frequent[k] = top5_list[:]
del destination_clusters
import gc
gc.collect()

#for k in destination_clusters2:
#        top5_list = get_top5(destination_clusters2[k])
#        most_frequent2[k] = top5_list[:]
#del destination_clusters2
#gc.collect()
#
#for k in destination_clusters3:
#        top5_list = get_top5(destination_clusters3[k])
#        most_frequent3[k] = top5_list[:]
#del destination_clusters3
#gc.collect()
#
#for k in destination_clusters4:
#        top5_list = get_top5(destination_clusters4[k])
#        most_frequent4[k] = top5_list[:]
#del destination_clusters4
#gc.collect()






print "Predicting on test.."
with open("../../Data/test_leak_preds.csv", "w") as outfile:
	outfile.write("id,hotel_cluster\n")
	for i, row in enumerate(DictReader(open("../../Data/test.csv"))):
		key = row["user_location_country"] + "_"  + row["user_location_region"] + "_" + row["user_location_city"] + "_" + row["hotel_market"] + "_"+ row["orig_destination_distance"]
	        #key2 = row["user_id"] + "_" + row["srch_destination_id"]
        	#key3 = row["srch_destination_id"] + "_" + row["hotel_market"]
		#key4 = row["hotel_market"]

		if row["orig_destination_distance"] == "":
			top5_list = []
		else:
			top5_list = most_frequent[key][:]
		if isinstance(top5_list, str):
			top5_list = []

		
		#if len(top5_list) < 5:
		#	temp_top5_list = most_frequent2.get(key2,[])
		#	for v in temp_top5_list:
		#		if v not in top5_list:
                #                        top5_list.append(v)
                #                        if len(top5_list) == 5:
                #                                break
	
		#if len(top5_list) < 5:
                #        temp_top5_list = most_frequent3[key3]
                #        for v in temp_top5_list:
                #                if v not in top5_list:
		#			top5_list.append(v)
		#			if len(top5_list) == 5:
		#				break

		#if len(top5_list) < 5:
                #        temp_top5_list = most_frequent4[key4]
                #        for v in temp_top5_list:
                #                if v not in top5_list:
                #                        top5_list.append(v)
                #                        if len(top5_list) == 5:
                #                                break

		top5_clusters = " ".join(top5_list)

		outfile.write("%d,%s\n"%(i,top5_clusters))
		if i % 1000000 == 0:
			print("%s\t%s"%(i, datetime.now() - start))
del most_frequent
del most_frequent2
del most_frequent3
del most_frequent4
gc.collect()

