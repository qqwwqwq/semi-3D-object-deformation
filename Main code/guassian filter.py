import joblib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
from kinect_smoothing.utils import plot_image_frame, plot_trajectories, plot_trajectory_3d
from kinect_smoothing import HoleFilling_Filter, Denoising_Filter
image_path='data/sample_img.pkl'
rootpath="/mnt/storage/buildwin/desk_backword"
image_frame =[np.load(rootpath+'/11.13/'+str(49)+'/dep1111.npy')]
print('original depth image frames')
plot_image_frame(image_frame)
# hole_filter = HoleFilling_Filter(flag='min')
# hf_image_frame = hole_filter.smooth_image_frames(image_frame)
# print('hole filled image frames (filled invalid values)')
# plot_image_frame(hf_image_frame)
noise_filter = Denoising_Filter(flag='gaussian',sigma=10,ksize=5)
denoise_image_frame = noise_filter.smooth_image_frames(image_frame)
print('denoised image frames (optional)')
plot_image_frame(denoise_image_frame)
np.save(rootpath+'/11.13/'+str(49)+'/dep1111_smoothed.npy',denoise_image_frame[0])