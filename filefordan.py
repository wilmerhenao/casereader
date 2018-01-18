import dose_to_points_data_pb2

# example of old file
#f = open("2c47da29-2762-41e2-bb59-36a15f618717", "rb")
# example of new file
f = open("0a199e18-0201-45dc-a75a-70d517881721", "rb")
dpdata = dose_to_points_data_pb2.DoseToPointsData()
dpdata.ParseFromString(f.read())
f.close()
for pd in dpdata.PointDoses:
    print('index', pd.Index)
    for bd in pd.BeamletDoses:
        print(bd.Index, bd.Dose)
