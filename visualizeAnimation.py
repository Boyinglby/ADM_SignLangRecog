import pandas as pd
from pointcloud import PointCloud3D

def create_anim_series(file_base_name):
    """
    file_base_name: str e.g. bye_apurve_
    """
    for i in range(1,10):
        filename = 'data/'+ file_base_name + str(i) +'.txt'
        data = pd.read_csv(filename, sep=' ').values
        Example = PointCloud3D(data)
        Example.plot_animation(f"{file_base_name}{i}_noTransf.gif")

        # rotate transform and translate (spine mid to [0,0,0]) 
        Example.transform()

        # plot and example animation of transformed sequential data
        Example.plot_animation(f"{file_base_name}{i}_Transf.gif")

create_anim_series("bye_mahendra_")