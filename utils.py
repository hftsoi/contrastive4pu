import uproot
import numpy as np
import glob
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.colors import LogNorm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


def cell_to_grid(data, has_cell_HS):
    # cell vector xyz to angles eta phi
    def compute_eta_phi(x, y, z):
        phi = np.arctan2(y, x)
        p = np.sqrt(x**2 + y**2 + z**2)
        eta = 0.5 * np.log((p + z) / (p - z))
        return eta, phi

    eta_list = []
    phi_list = []

    for x_arr, y_arr, z_arr in zip(data['cell_x'], data['cell_y'], data['cell_z']):
        eta, phi = compute_eta_phi(x_arr, y_arr, z_arr)
        eta_list.append(eta)
        phi_list.append(phi)

    data['cell_eta'] = np.array(eta_list, dtype=object)
    data['cell_phi'] = np.array(phi_list, dtype=object)

    # binning for the grid data
    eta_bins = np.arange(-2.5, 2.5 + 0.1, 0.1)
    phi_bins = np.arange(-np.pi, np.pi + np.pi/32, np.pi/32)

    # data['cell_XXX'] is an array of 1d arrays with different shapes (no. of cell deposits)
    n_events = len(data['cell_eta'])

    # grid data format, separating HS and PU deposits
    if has_cell_HS:
        X_isHS = np.zeros((n_events, len(phi_bins)-1, len(eta_bins)-1, 6))
        X_isPU = np.zeros((n_events, len(phi_bins)-1, len(eta_bins)-1, 6))
    else:
        X = np.zeros((n_events, len(phi_bins)-1, len(eta_bins)-1, 6))

    # loop over events
    for i in range(n_events):
        # these contain deposit info per cell
        cell_eta    = data['cell_eta'][i]
        cell_phi    = data['cell_phi'][i]
        cell_et     = data['cell_et'][i] / 1000. # MeV -> GeV
        cell_sampling = data['cell_sampling'][i]
        if has_cell_HS:
            cell_isHS   = data['cell_isHS'][i]

        abs_eta = np.abs(cell_eta)
        # conditions for separating channels according to the atlas calorimeter sampling layers
        cond_ch0 = ((cell_sampling == 0) & (abs_eta < 1.5)) | \
                ((cell_sampling == 4) & (abs_eta > 1.5) & (abs_eta < 1.8))
        
        cond_ch1 = ((cell_sampling == 1) & (abs_eta < 1.5)) | \
                ((cell_sampling == 5) & (abs_eta > 1.5) & (abs_eta < 2.5))
        
        cond_ch2 = ((cell_sampling == 2) & (abs_eta < 1.5)) | \
                ((cell_sampling == 6) & (abs_eta > 1.5) & (abs_eta < 2.5))
        
        cond_ch3 = ((cell_sampling == 3) & (abs_eta < 1.5)) | \
                ((cell_sampling == 7) & (abs_eta > 1.5) & (abs_eta < 2.5))
        
        cond_ch4 = ((cell_sampling == 12) & (abs_eta < 1)) | \
                ((cell_sampling == 18) & (abs_eta > 1.1) & (abs_eta < 1.5)) | \
                ((cell_sampling == 8)  & (abs_eta > 1.5) & (abs_eta < 2.5))
        
        cond_ch5 = ((cell_sampling == 13) & (abs_eta < 0.9)) | \
                ((cell_sampling == 19) & (abs_eta > 1)   & (abs_eta < 1.5)) | \
                ((cell_sampling == 15) & (abs_eta > 0.9) & (abs_eta < 1))   | \
                ((cell_sampling == 9)  & (abs_eta > 1.5) & (abs_eta < 2.5))
        
        conditions = [cond_ch0, cond_ch1, cond_ch2, cond_ch3, cond_ch4, cond_ch5]
        
        # loop over channel
        if has_cell_HS:
            for ch in range(len(conditions)):
                mask = conditions[ch]
                # further separate HS and PU deposits
                hs_mask = mask & (cell_isHS == 1)
                pu_mask = mask & (cell_isHS == 0)
                
                # get HS cells per layer
                eta_hs = cell_eta[hs_mask]
                phi_hs = cell_phi[hs_mask]
                et_hs  = cell_et[hs_mask]
                hist_hs, _, _ = np.histogram2d(eta_hs, phi_hs, bins=[eta_bins, phi_bins], weights=et_hs)
                
                # get PU cells per layer
                eta_pu = cell_eta[pu_mask]
                phi_pu = cell_phi[pu_mask]
                et_pu  = cell_et[pu_mask]
                hist_pu, _, _ = np.histogram2d(eta_pu, phi_pu, bins=[eta_bins, phi_bins], weights=et_pu)
                
                X_isHS[i, :, :, ch] = hist_hs.T
                X_isPU[i, :, :, ch] = hist_pu.T

        else:
            for ch in range(len(conditions)):
                mask = conditions[ch]
                
                # get HS cells per layer
                eta = cell_eta[mask]
                phi = cell_phi[mask]
                et  = cell_et[mask]
                hist, _, _ = np.histogram2d(eta, phi, bins=[eta_bins, phi_bins], weights=et)
                
                X[i, :, :, ch] = hist.T

    if has_cell_HS:
        return eta_bins, phi_bins, X_isHS, X_isPU
    else:
        return eta_bins, phi_bins, X


def plot_layers(event_idx, X, label):

    eta_bins = np.arange(-2.5, 2.5 + 0.1, 0.1)
    phi_bins = np.arange(-np.pi, np.pi + np.pi/32, np.pi/32)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    for ch in range(6):
        ax = axes[ch // 3, ch % 3]
        
        if event_idx is not None:
            heatmap_data = X[event_idx, :, :, ch]
        else:
            heatmap_data = X[:, :, ch]
        
        mesh = ax.pcolormesh(eta_bins,
                             phi_bins,
                             heatmap_data,
                             cmap='viridis',
                             norm = LogNorm(vmin = 1e-1, vmax = 1e1))
        
        ax.set_xlabel('Eta')
        ax.set_ylabel('Phi')
        ax.set_title(f'Layer {ch} {label}')
        
        fig.colorbar(mesh, ax=ax, label='ET [GeV]')

    plt.tight_layout()
    plt.show()


def augment_pu(image, target_pu, shift_phi):
    # input pu is 200, the target pu should be less than that to do removal
    survival_prob = target_pu / 200

    # draw random number in [0,1] and compare with survival_prob to keep or dump a pixel
    #tf.random.set_seed(42)
    random_numbers = tf.random.uniform(tf.shape(image))
    survival_mask = tf.cast(random_numbers < survival_prob, tf.float32)

    # zero out pixels to achieve target pu level
    image_augmented = image * survival_mask

    if shift_phi:
        shift_amount = tf.random.uniform([], minval=0, maxval=image.shape[0], dtype=tf.int32)
        image_augmented = tf.roll(image_augmented, shift=shift_amount, axis=0)

    return image_augmented


def generate_pair_sig_for_contrastive(x_hs, x_pu, target_pu_1, target_pu_2):
    view1 = x_hs + augment_pu(image=x_pu, target_pu=target_pu_1, shift_phi=True)
    view2 = x_hs + augment_pu(image=x_pu, target_pu=target_pu_2, shift_phi=True)
    return view1, view2


def generate_pair_bkg_for_contrastive(x_bkg, target_pu_1, target_pu_2):
    view1 = augment_pu(image=x_bkg, target_pu=target_pu_1, shift_phi=True)
    view2 = augment_pu(image=x_bkg, target_pu=target_pu_2, shift_phi=True)
    return view1, view2


def generate_sig_label_for_classification(x_hs, x_pu, target_pu):
    x = x_hs + augment_pu(image=x_pu, target_pu=target_pu, shift_phi=True)
    return x, tf.constant(1, dtype=tf.int32)


def generate_bkg_label_for_classification(x_bkg, target_pu):
    x = augment_pu(image=x_bkg, target_pu=target_pu, shift_phi=True)
    return x, tf.constant(0, dtype=tf.int32)


def generate_batch_for_contrastive(X_hs, X_pu, X_bkg, pu_min, pu_max, batch_size):
    n_sig = X_hs.shape[0]
    n_bkg = X_bkg.shape[0]
    half_batch = batch_size // 2

    while True:
        view1_list = []
        view2_list = []

        idx_sig = np.random.choice(n_sig, half_batch, replace=True)
        for i in idx_sig:
            x_hs = tf.convert_to_tensor(X_hs[i])
            x_pu = tf.convert_to_tensor(X_pu[i])
            random_pu = tf.random.uniform([2], minval=pu_min, maxval=pu_max)
            view1, view2 = generate_pair_sig_for_contrastive(x_hs=x_hs, x_pu=x_pu, target_pu_1=random_pu[0], target_pu_2=random_pu[1])
            view1_list.append(view1.numpy())
            view2_list.append(view2.numpy())

        idx_bkg = np.random.choice(n_bkg, half_batch, replace=True)
        for i in idx_bkg:
            x_bkg = tf.convert_to_tensor(X_bkg[i])
            random_pu = tf.random.uniform([2], minval=pu_min, maxval=pu_max)
            view1, view2 = generate_pair_bkg_for_contrastive(x_bkg=x_bkg, target_pu_1=random_pu[0], target_pu_2=random_pu[1])
            view1_list.append(view1.numpy())
            view2_list.append(view2.numpy())

        yield (np.array(view1_list), np.array(view2_list))


def generate_batch_for_classifier(X_hs, X_pu, X_bkg, pu_min, pu_max, batch_size):
    n_sig = X_hs.shape[0]
    n_bkg = X_bkg.shape[0]
    half_batch = batch_size // 2

    while True:
        x_list = []
        y_list = []

        idx_sig = np.random.choice(n_sig, half_batch, replace=True)
        for i in idx_sig:
            x_hs = tf.convert_to_tensor(X_hs[i])
            x_pu = tf.convert_to_tensor(X_pu[i])
            random_pu = tf.random.uniform([], minval=pu_min, maxval=pu_max)
            x, y = generate_sig_label_for_classification(x_hs=x_hs, x_pu=x_pu, target_pu=random_pu)
            x_list.append(x.numpy())
            y_list.append(y.numpy())

        idx_bkg = np.random.choice(n_bkg, half_batch, replace=True)
        for i in idx_bkg:
            x_bkg = tf.convert_to_tensor(X_bkg[i])
            random_pu = tf.random.uniform([], minval=pu_min, maxval=pu_max)
            x, y = generate_bkg_label_for_classification(x_bkg=x_bkg, target_pu=random_pu)
            x_list.append(x.numpy())
            y_list.append(y.numpy())

        yield (np.array(x_list), np.array(y_list))


def generate_dataset_for_classifier(X_hs, X_pu, X_bkg, target_pu):
    x_list = []
    y_list = []

    for i in range(X_hs.shape[0]):
        x_hs = tf.convert_to_tensor(X_hs[i])
        x_pu = tf.convert_to_tensor(X_pu[i])
        x = x_hs + augment_pu(image=x_pu, target_pu=target_pu, shift_phi=True)
        y = 1
        x_list.append(x)
        y_list.append(y)

    for i in range(X_bkg.shape[0]):
        x_bkg = tf.convert_to_tensor(X_bkg[i])
        x = augment_pu(image=x_bkg, target_pu=target_pu, shift_phi=True)
        y = 0
        x_list.append(x)
        y_list.append(y)

    return np.array(x_list), np.array(y_list)


