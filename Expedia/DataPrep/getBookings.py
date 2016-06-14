"""
Code to get the bookings from the train file
__author__ : SRK
"""
import csv

train_file_handle = open("../../Data/train.csv")
train_out_file_handle = open("../../Data/train_bookings.csv","w")

reader = csv.reader(train_file_handle)
writer = csv.writer(train_out_file_handle)

header = reader.next()
writer.writerow(["id"] + header)

is_booking_index = header.index("is_booking")
print "Booking index is : ", is_booking_index

total_count = 0
count = 0
for row in reader:
	if row[is_booking_index] == "1":
		writer.writerow([total_count] + row)
		count += 1
	total_count += 1
	if total_count % 100000 == 0:
		print total_count, count

print "Total count : ", total_count
print "Booking count : ", count	

train_file_handle.close()
train_out_file_handle.close()
