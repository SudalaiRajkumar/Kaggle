import csv
from datetime import datetime

with open("../../Data/val.csv") as train_file:
	with open("../../Data/val_leak_preds.csv") as leak_file:
		reader = csv.reader(train_file)
		leak_reader = csv.DictReader(leak_file)

		out_file = open("../../Data/val_woleak.csv","w")
		out_writer = csv.writer(out_file)
		out_file2 = open("../../Data/val_withleak.csv","w")
                out_writer2 = csv.writer(out_file2)

		header = reader.next()
        	out_writer.writerow(header)

		leak_count = 0
		for index, row in enumerate(reader):
			leak_row = leak_reader.next()
			if leak_row["hotel_cluster"] == "":
				out_writer.writerow(row)
			else:
				out_writer2.writerow(row)
				leak_count +=1
		print "Leak count is : ", leak_count

		out_file.close()


with open("../../Data/test.csv") as train_file:
        with open("../../Data/test_leak_preds.csv") as leak_file:
                reader = csv.reader(train_file)
                leak_reader = csv.DictReader(leak_file)

                out_file = open("../../Data/test_woleak.csv","w")
                out_writer = csv.writer(out_file)
		out_file2 = open("../../Data/test_withleak.csv","w")
                out_writer2 = csv.writer(out_file2)

                header = reader.next()
                out_writer.writerow(header)

                leak_count = 0
                for index, row in enumerate(reader):
                        leak_row = leak_reader.next()
                        if leak_row["hotel_cluster"] == "":
                                out_writer.writerow(row)
                        else:
				out_writer2.writerow(row)
                                leak_count +=1
                print "Leak count is : ", leak_count

                out_file.close()


# get only the bookings from the validation sample #
train_file_handle = open("../../Data/val_woleak.csv")
train_out_file_handle = open("../../Data/val_bookings_woleak.csv","w")

reader = csv.reader(train_file_handle)
writer = csv.writer(train_out_file_handle)

header = reader.next()
writer.writerow(header)

is_booking_index = header.index("is_booking")
print "Booking index is : ", is_booking_index

total_count = 0
count = 0
for row in reader:
        if row[is_booking_index] == "1":
                writer.writerow(row)
                count += 1
        total_count += 1
        if total_count % 100000 == 0:
                print total_count, count

print "Total count : ", total_count
print "Booking count : ", count

train_file_handle.close()
train_out_file_handle.close()

