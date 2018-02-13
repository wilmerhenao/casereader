import DoseToPoints_pb2

# example of old file (corresponding to scrambled voxels)
#f = open("0a199e18-0201-45dc-a75a-70d517881721", "rb")
# example of new file
f = open("/mnt/fastdata/Data/spine360/by-Structure/PsVM2m_2_90_2/042a9857-3cc8-4fef-9cb8-7b888b7ff116", "rb")
dpdata = DoseToPoints_pb2.DoseToPointsData()
dpdata.ParseFromString(f.read())
f.close()
for pd in dpdata.PointDoses:
    print('index', pd.Index)
    for bd in pd.BeamletDoses:
        print(bd.Index, bd.Dose)
