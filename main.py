import mne
import h5py
import numpy as np
import pandas as pd
import os.path as op
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from ipynb.fs.full.fx_bids import load_bids

dir_bids = r'E:\EEG-project\EEG_source_localization\data\Localize-MI' # base directory of the BIDS dataset
task = 'seegstim' # task name
subj = 'sub-07' # subject id
run = 'run-07' # rund id

epo = load_bids(dir_bids, subj, task, run) # load epochs
#前向模型的解，包含无固定方向和固定方向两种
fwd = mne.read_forward_solution(op.join(dir_bids, 'derivatives', 'sourcemodelling', subj, 'fwd', '%s_fwd.fif' % subj)) # load forward solution
seeg_ch_info = pd.read_csv(op.join(dir_bids, 'derivatives', 'epochs', subj, 'ieeg', '%s_task-%s_space-surface_electrodes.tsv' % (subj, task)), sep='\t') # load SEEG channel info 
events = pd.read_csv(op.join(dir_bids, 'derivatives', 'epochs', subj, 'eeg', '%s_task-%s_%s_epochs.tsv' % (subj, task, run)), sep='\t') # load events (contain the stimulating channel name)
trans = h5py.File(op.join(dir_bids, 'derivatives', 'sourcemodelling', subj, 'xfm', '%s_from-head_to-surface.h5' % subj)).get('trans')[()] # load head-to-surface transform

#plot
#f = epo.plot_sensors(kind='3d') # sensors' positions
#f = epo.plot(n_epochs=4, n_channels=60, picks=epo.ch_names, events=False) # EEG data

#设置平均导联
epo = epo.set_eeg_reference('average', projection=True)
epo.apply_proj()
#协方差的计算没有包含放电刺激时段
#auto方法选择'shrunk'方法计算（Ledoit, O. and M. Wolf, 2004），比empirical略好一点点
cov = mne.compute_covariance(epo, method='auto', tmin=-0.25, tmax=-0.05)

#plt.imshow(cov.data, cmap='jet', interpolation='nearest')
#plt.colorbar()
#plt.show()

#loose设置为0，只考虑垂直皮层方向的放电，loose设置为1，考虑所有方向的放电
#depth是对源位置的深度先验。如果为0，没有限制
inv = mne.minimum_norm.make_inverse_operator(epo.info, fwd, cov, loose=1, depth=0.1)

evo = epo.average()
evo = evo.crop(-0.002, 0.002)
#f = evo.plot_topomap(np.arange(-0.001, 0.002, 0.0004))

snr = 10
lambda2 = 1. / snr ** 2
stc = mne.minimum_norm.apply_inverse(evo, inv, method='eLORETA', lambda2=lambda2) #关键函数，决定反演算法选择
#stc = mne.minimum_norm.apply_inverse(evo, inv, method='dSPM', lambda2=lambda2)
#stc = mne.minimum_norm.apply_inverse(evo, inv, method='MNE', lambda2=lambda2)
#stc = mne.minimum_norm.apply_inverse(evo, inv, method='sLORETA', lambda2=lambda2)

#真实放电电极位置
stim_info = events.trial_type.unique()[0] # get the stimulating contacts' names
stim_ch = stim_info.split()[0]
stim_chs = [stim_ch.split('-')[0], ''.join(n for n in stim_ch.split('-')[0] if not n.isdigit()) + stim_ch.split('-')[1]] # get monopolar names
stim_coords = seeg_ch_info.loc[seeg_ch_info.name.isin(stim_chs)][['x', 'y', 'z']].values.squeeze() # find the coordinates of the stimulating channels
stim_coords_bip = np.mean(stim_coords, 0) # take the mean between the two contacts
hemi = 'lh' if '\'' in stim_ch else 'rh'

print('Stimulation channels: %s (coordinates: %s)' % (stim_ch, np.round(stim_coords_bip, 2).squeeze()))
print('Stimulation intensity: %s' % stim_info.split()[-1])
print('hemisphere: %s' % hemi)

#反演放电位置
#peak_ix = stc.get_peak(hemi=hemi, vert_as_index=True) # get the peak of activation
#peak_id = stc.get_peak(hemi=hemi)
peak_ix = stc.get_peak(hemi=hemi, vert_as_index=True, tmin=-0.0001, tmax=0.0001) # get the peak of activation
peak_id = stc.get_peak(hemi=hemi, tmin=-0.0001, tmax=0.0001)

#f = plt.plot(stc.times, stc.data[peak_ix[0], :])
#plt.xlabel('time (s)')
#plt.ylabel('Current')
#f.show()

surf = [mne.transforms.apply_trans(trans, s['rr']) for s in fwd['src']] # get surface coordinates and apply the head-to-surface transform
tris= [s['tris'] for s in fwd['src']] # get mesh triangle faces

hemi_ix = 0 if '\'' in stim_ch else 1 # determine hemisphere being stimulated
curr_max_coords = surf[hemi_ix][peak_id[0],:]

f = mlab.figure()

for s, t in zip(surf, tris):
    mlab.triangular_mesh(s[:,0], s[:,1], s[:,2], t, color=(0.7, 0.7, 0.7), opacity=0.5)

mlab.points3d(*curr_max_coords, scale_factor=0.01, color=(0.0, 0.0, 1.0)) # show the source location in blue
mlab.points3d(*stim_coords_bip, scale_factor=0.01, color=(0.0, 1.0, 0.0)) # show the coordinates of the stimulation site in green

mlab.show()

print('Distance between location of the max current value and stimulation coordinates = %.2f mm' % (euclidean(stim_coords_bip, curr_max_coords)*1e3))