import sys
import csv

from datetime import datetime

with open("../../Data/val_bookings_woleak.csv") as train_file:
                reader = csv.reader(train_file)
                #leak_reader = csv.DictReader(leak_file)

                out_file = open("../../Data/val_bookings_woleak_wodist.csv","w")
                out_writer = csv.writer(out_file)
                out_file2 = open("../../Data/val_bookings_woleak_dist.csv","w")
                out_writer2 = csv.writer(out_file2)

                header = reader.next()
		dist_index = header.index("orig_destination_distance")
                out_writer.writerow(header)
		out_writer2.writerow(header)

                leak_count = 0
                for index, row in enumerate(reader):
                        if row[dist_index] == "":
                                out_writer.writerow(row)
                        else:
                                out_writer2.writerow(row)
                                leak_count +=1
                print "With Dist count is : ", leak_count
		print index

                out_file.close()

