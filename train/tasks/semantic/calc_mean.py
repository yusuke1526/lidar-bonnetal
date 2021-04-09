#!/usr/bin/env python3
import numpy as np
import os
import sys

from statistics import mean, stdev

class CalcMean:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self):
        H = 64
        W = 1024
        fov_up = 15.0
        fov_down = -25.0
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.reset()

    def reset(self):
        self.points = np.zeros((0,3), dtype=np.float32)
        self.remissions = np.zeros((0,1), dtype=np.float32)
        #self.scan_range = np.zeros((0,1), dtype=np.float32)
        #print(self.scan_range.shape)
        self.scan_range = []

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)       # [H,W] mask


    def open_scan(self, filename):
        self.reset()

        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        print(scan.shape)

        points = scan[:, 0:3]
        remissions = scan[:, 3]
        arrr = self.high_fov_cut(scan)
        return arrr

    def high_fov_cut(self, scan):
        self.reset()
        points = scan[:, 0:3]
        remissions = scan[:, 3]
        self.points = points
        self.remissions = remissions

        #print(scan.shape)
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi
        fov_down = self.proj_fov_down / 180.0 * np.pi
        fov = abs(fov_down) + abs(fov_up)

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)
        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]
        #print(len(scan_z))

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        #print(yaw.shape)
        #print("yaw:", yaw)
        pitch = np.arcsin(scan_z / depth)
        #print("pitch:", pitch)

        proj_x = 0.5 * (yaw / np.pi + 1.0) #in [0.0, 1.0]
        #print(proj_x.shape)
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov # in [0.0, 1.0]

        tmp = np.append(scan, proj_x.reshape(-1,1), axis=1)
        unit = np.append(tmp, proj_y.reshape(-1,1), axis=1)
        #print(unit.shape)
        #proj_y < 0.3 cut
        #proj_y = [i for i in proj_y if i > 0.3]
        #print(len(proj_y))

        arrr = np.delete(unit, np.where(unit[:, 5] < 0.34)[0], 0)
        #print(arrr.shape)
        arrr = np.delete(arrr, [4,5] ,1)
        print(arrr.shape)

        return arrr

    def calc(self, filename):
        self.reset()

        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        print(scan.shape)

        #calc mean, std
        for i in range(scan.shape[0]):
            self.scan_range.append(np.sqrt(scan[i][0]**2 + scan[i][1]**2 + scan[i][2]**2))
            #self.scan_range = np.append(self.scan_range, np.sqrt(scan[i][0]**2 + scan[i][1]**2 + scan[i][2]**2))

        #print(len(self.scan_range))
        scan_range_mean = mean(self.scan_range)
        scan_range_std = stdev(self.scan_range)
        #scan_range /= scan.shape[0]
        scan_mean = np.mean(scan, axis=0)
        scan_std = np.std(scan, axis=0)
        print(scan_mean)
        print(scan_std)
        print(scan_range_mean)
        print(scan_range_std)
        return scan_mean, scan_std, scan_range_mean, scan_range_std

EXTENSIONS_SCAN = ['.bin']
def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)
        
if __name__ == "__main__":
    calc = True
    higher_cut = False
    root = "./dataset"
    roots = os.path.join(root, "sequences", "101")
    dests = os.path.join(root, "sequences", "101", "velodyne")
    try:
        os.makedirs(dests, exist_ok=True)
    except FileExistsError:
        pass
    scan_path = os.path.join(roots, "velodyne")
    scan_files = [os.path.join(dp,f) for dp, dn, fn in os.walk(os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
    scan_files.sort()
    print(scan_files)
    #sys.exit()

    #scan_file = "./dataset/sequences/28/velodyne/000.bin"
    scan = CalcMean()
    scan_mean_list = []
    scan_std_list = []
    scan_range_mean_list = []
    scan_range_std_list = []
    for i in range(len(scan_files)):
        scan_file = scan_files[i]

        if higher_cut:
            arrr = scan.open_scan(scan_file)
            zero = "{0:03d}".format(i)
            bin_file = os.path.join(dests, zero+".bin")
            arrr.astype('float32').tofile(bin_file)
            #sys.exit()

        if calc:
            scan_mean, scan_std, scan_range_mean, scan_range_std = scan.calc(scan_file)
            scan_mean_list.append(scan_mean)
            scan_std_list.append(scan_std)
            scan_range_mean_list.append(scan_range_mean)
            scan_range_std_list.append(scan_range_std)
            #sys.exit()

    if calc:
        a = np.array(scan_mean_list)
        t_mean = np.mean(a, axis=0)
        b = np.array(scan_std_list)
        t_std = np.mean(b, axis=0)
        #t_mean = mean(scan_mean_list)
        #t_std = mean(scan_std_list)
        #print(scan_range_mean_list)
        t_range_mean = mean(scan_range_mean_list)
        t_range_std = mean(scan_range_std_list)
        print("tmean:", t_mean)
        print("tstd:", t_std)
        print("trange_mean:", t_range_mean)
        print("trange_std:", t_range_std)
