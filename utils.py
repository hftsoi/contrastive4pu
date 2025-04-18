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
import matplotlib.gridspec as gridspec


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


def plot_layers(event_idx, X, label, n_layers):
    eta_bins = np.arange(-2.5, 2.5 + 0.1, 0.1)
    phi_bins = np.arange(-np.pi, np.pi + np.pi/32, np.pi/32)

    if n_layers == 6:
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

    elif n_layers == 1:
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        ax = axes
        
        if event_idx is not None:
            heatmap_data = X[event_idx, :, :, 0]
        else:
            heatmap_data = X[:, :, 0]
        
        mesh = ax.pcolormesh(eta_bins,
                            phi_bins,
                            heatmap_data,
                            cmap='viridis',
                            norm = LogNorm(vmin = 1e-1, vmax = 1e1))
        
        ax.set_xlabel('Eta')
        ax.set_ylabel('Phi')
        #ax.set_title(f'Layer {ch} {label}')
        
        fig.colorbar(mesh, ax=ax, label='ET [GeV]')

    plt.tight_layout()
    plt.show()


def augment_pu(image, target_pu, shift_phi, threshold):
    # among the low-energy cells, randomly remove a portion of them according to the target/original pu ratio
    low_energy_mask = image < threshold
    random_tensor = tf.random.uniform(tf.shape(image))
    removal_prob = 1 - target_pu / 200
    drop_mask = tf.logical_and(low_energy_mask, random_tensor < removal_prob)
    image_after_removal = image * (1 - tf.cast(drop_mask, tf.float32))
    
    # then scale the remaining cell energies
    total_e_before_removal = tf.reduce_sum(image)
    total_e_after_removal = tf.reduce_sum(image_after_removal)
    total_e_scale = (total_e_before_removal * tf.cast(target_pu, tf.float32) / 200) / total_e_after_removal
    
    image_augmented = image_after_removal * total_e_scale
    
    if shift_phi:
        shift_amount = tf.random.uniform([], minval=0, maxval=tf.shape(image_augmented)[0], dtype=tf.int32)
        image_augmented = tf.roll(image_augmented, shift=shift_amount, axis=0)
    
    #return image_augmented, total_e_before_removal, total_e_after_removal, total_e_scale
    return image_augmented


def generate_pair_sig_for_contrastive(x_hs, x_pu, target_pu_1, target_pu_2, threshold):
    view1 = x_hs + augment_pu(image=x_pu, target_pu=target_pu_1, shift_phi=True, threshold=threshold)
    view2 = x_hs + augment_pu(image=x_pu, target_pu=target_pu_2, shift_phi=True, threshold=threshold)
    return view1, view2


def generate_pair_bkg_for_contrastive(x_bkg, target_pu_1, target_pu_2, threshold):
    view1 = augment_pu(image=x_bkg, target_pu=target_pu_1, shift_phi=True, threshold=threshold)
    view2 = augment_pu(image=x_bkg, target_pu=target_pu_2, shift_phi=True, threshold=threshold)
    return view1, view2


def generate_sig_label_for_classification(x_hs, x_pu, target_pu, threshold):
    x = x_hs + augment_pu(image=x_pu, target_pu=target_pu, shift_phi=True, threshold=threshold)
    return x, tf.constant(1, dtype=tf.int32)


def generate_bkg_label_for_classification(x_bkg, target_pu, threshold):
    x = augment_pu(image=x_bkg, target_pu=target_pu, shift_phi=True, threshold=threshold)
    return x, tf.constant(0, dtype=tf.int32)


def generate_batch_for_contrastive(X_hs, X_pu, X_bkg, pu_min, pu_max, batch_size, threshold):
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
            view1, view2 = generate_pair_sig_for_contrastive(x_hs=x_hs, x_pu=x_pu, target_pu_1=random_pu[0], target_pu_2=random_pu[1], threshold=threshold)
            view1_list.append(view1.numpy())
            view2_list.append(view2.numpy())

        idx_bkg = np.random.choice(n_bkg, half_batch, replace=True)
        for i in idx_bkg:
            x_bkg = tf.convert_to_tensor(X_bkg[i])
            random_pu = tf.random.uniform([2], minval=pu_min, maxval=pu_max)
            view1, view2 = generate_pair_bkg_for_contrastive(x_bkg=x_bkg, target_pu_1=random_pu[0], target_pu_2=random_pu[1], threshold=threshold)
            view1_list.append(view1.numpy())
            view2_list.append(view2.numpy())

        yield (np.array(view1_list), np.array(view2_list))


def generate_batch_for_classifier(X_hs, X_pu, X_bkg, pu_min, pu_max, batch_size, threshold):
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
            x, y = generate_sig_label_for_classification(x_hs=x_hs, x_pu=x_pu, target_pu=random_pu, threshold=threshold)
            x_list.append(x.numpy())
            y_list.append(y.numpy())

        idx_bkg = np.random.choice(n_bkg, half_batch, replace=True)
        for i in idx_bkg:
            x_bkg = tf.convert_to_tensor(X_bkg[i])
            random_pu = tf.random.uniform([], minval=pu_min, maxval=pu_max)
            x, y = generate_bkg_label_for_classification(x_bkg=x_bkg, target_pu=random_pu, threshold=threshold)
            x_list.append(x.numpy())
            y_list.append(y.numpy())

        yield (np.array(x_list), np.array(y_list))


def generate_dataset_for_classifier(X_hs, X_pu, X_bkg, target_pu, threshold):
    x_list = []
    y_list = []

    for i in range(X_hs.shape[0]):
        x_hs = tf.convert_to_tensor(X_hs[i])
        x_pu = tf.convert_to_tensor(X_pu[i])
        x = x_hs + augment_pu(image=x_pu, target_pu=target_pu, shift_phi=True, threshold=threshold)
        y = 1
        x_list.append(x)
        y_list.append(y)

    for i in range(X_bkg.shape[0]):
        x_bkg = tf.convert_to_tensor(X_bkg[i])
        x = augment_pu(image=x_bkg, target_pu=target_pu, shift_phi=True, threshold=threshold)
        y = 0
        x_list.append(x)
        y_list.append(y)

    return np.array(x_list), np.array(y_list)


def plot_roc(Y_test, Y_pred_embedding_nofinetune, Y_pred_embedding_finetune, Y_pred_standalone, test_pu):
    plt.figure(figsize=(6,5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(test_pu)))
    
    for i, pu in enumerate(test_pu):
        y_true = Y_test[i]
        y_pred_embed_nofinetuning = Y_pred_embedding_nofinetune[i]
        fpr_embed_nofinetuning, tpr_embed_nofinetuning, _ = roc_curve(y_true, y_pred_embed_nofinetuning)
        auc_embed_nofinetuning = auc(fpr_embed_nofinetuning, tpr_embed_nofinetuning)

        y_pred_embed_finetuning = Y_pred_embedding_finetune[i]
        fpr_embed_finetuning, tpr_embed_finetuning, _ = roc_curve(y_true, y_pred_embed_finetuning)
        auc_embed_finetuning = auc(fpr_embed_finetuning, tpr_embed_finetuning)
        
        y_pred_stand = Y_pred_standalone[i]
        fpr_stand, tpr_stand, _ = roc_curve(y_true, y_pred_stand)
        auc_stand = auc(fpr_stand, tpr_stand)
        
        plt.plot(fpr_stand, tpr_stand, 
                 label=f'PU {pu} - Standalone (AUC={auc_stand:.4f})',
                 linestyle='-', marker=None, color=colors[i])
        
        plt.plot(fpr_embed_nofinetuning, tpr_embed_nofinetuning, 
                 label=f'PU {pu} - Embedding, no fine-tuning (AUC={auc_embed_nofinetuning:.4f})',
                 linestyle='--', marker=None, color=colors[i])
        
        plt.plot(fpr_embed_finetuning, tpr_embed_finetuning, 
                 label=f'PU {pu} - Embedding, fine-tuning (AUC={auc_embed_finetuning:.4f})',
                 linestyle='dotted', marker=None, color=colors[i])
    
    plt.xlabel('Bkg. eff.')
    plt.ylabel('Sig. eff.')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(True)
    #plt.yscale('log')
    plt.xscale('log')
    plt.xlim((0.00001, 1))
    plt.ylim((0, 1))
    plt.show()


def plot_sig_eff_vs_pu_at_single_bkgeff(Y_test, Y_pred_embedding_nofinetune, Y_pred_embedding_finetune, Y_pred_standalone, test_pu, bkg_eff_list):
    plt.figure(figsize=(6,5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(bkg_eff_list)))
    
    for j, bkg_eff in enumerate(bkg_eff_list):
        sig_eff_embed_nofinetune = []
        sig_eff_embed_finetune = []
        sig_eff_stand = []
        
        for i in range(len(test_pu)):
            y_true = Y_test[i]
            
            y_pred_embed_nofinetune = Y_pred_embedding_nofinetune[i].ravel()
            fpr_embed_nofinetune, tpr_embed_nofinetune, _ = roc_curve(y_true, y_pred_embed_nofinetune)
            idx_embed_nofinetune = np.argmin(np.abs(fpr_embed_nofinetune - bkg_eff))
            sig_eff_embed_nofinetune.append(tpr_embed_nofinetune[idx_embed_nofinetune])

            y_pred_embed_finetune = Y_pred_embedding_finetune[i].ravel()
            fpr_embed_finetune, tpr_embed_finetune, _ = roc_curve(y_true, y_pred_embed_finetune)
            idx_embed_finetune = np.argmin(np.abs(fpr_embed_finetune - bkg_eff))
            sig_eff_embed_finetune.append(tpr_embed_finetune[idx_embed_finetune])
            
            y_pred_stand = Y_pred_standalone[i].ravel()
            fpr_stand, tpr_stand, _ = roc_curve(y_true, y_pred_stand)
            idx_stand = np.argmin(np.abs(fpr_stand - bkg_eff))
            sig_eff_stand.append(tpr_stand[idx_stand])
        
        plt.plot(test_pu, sig_eff_stand, marker='.', linestyle='-', color=colors[j],
                 label=f'Standalone (bkg. eff.={bkg_eff:.4f})')
        plt.plot(test_pu, sig_eff_embed_nofinetune, marker='.', linestyle='--', color=colors[j],
                 label=f'Embedding, no fine-tuning (bkg. eff.={bkg_eff:.4f})')
        plt.plot(test_pu, sig_eff_embed_finetune, marker='.', linestyle='dotted', color=colors[j],
                 label=f'Embedding, fine-tuning (bkg. eff.={bkg_eff:.4f})')
    
    plt.xlabel('PU')
    plt.ylabel('Sig. eff.')
    #plt.ylim((0.9, 1))
    plt.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()


def plot_eff_vs_pu_at_single_threshold(Y_test, Y_pred_embedding_nofinetune, Y_pred_embedding_finetune, Y_pred_standalone, test_pu, threshold_by_target_bkgeff_pu):
    # fix a single threshold from a ref bkg eff at a ref pu
    threshold_by_target_bkgeff, threshold_by_target_pu = threshold_by_target_bkgeff_pu
    test_pu_array = np.array(test_pu)
    idx = np.argmin(np.abs(test_pu_array - threshold_by_target_pu))
    
    # determine the fixed threshold for the embedding classifier
    y_true_ref = Y_test[idx]
    y_pred_embed_nofinetune_ref = Y_pred_embedding_nofinetune[idx].ravel()
    bkg_scores_embed_nofinetune = y_pred_embed_nofinetune_ref[y_true_ref == 0]
    emb_nofinetune_threshold = np.percentile(bkg_scores_embed_nofinetune, 100 * (1 - threshold_by_target_bkgeff))

    y_pred_embed_finetune_ref = Y_pred_embedding_finetune[idx].ravel()
    bkg_scores_embed_finetune = y_pred_embed_finetune_ref[y_true_ref == 0]
    emb_finetune_threshold = np.percentile(bkg_scores_embed_finetune, 100 * (1 - threshold_by_target_bkgeff))
    
    # determine the fixed threshold for the standalone classifier
    y_pred_stand_ref = Y_pred_standalone[idx].ravel()
    bkg_scores_stand = y_pred_stand_ref[y_true_ref == 0]
    stand_threshold = np.percentile(bkg_scores_stand, 100 * (1 - threshold_by_target_bkgeff))
    
    emb_nofinetune_bkgeff, emb_finetune_bkgeff, stand_bkgeff = [], [], []
    emb_nofinetune_sigeff, emb_finetune_sigeff, stand_sigeff = [], [], []
    
    for i, pu in enumerate(test_pu):
        y_true = Y_test[i]
        y_pred_embed_nofinetune = Y_pred_embedding_nofinetune[i].ravel()
        y_pred_embed_finetune = Y_pred_embedding_finetune[i].ravel()
        
        fpr_embed_nofinetune = np.mean(y_pred_embed_nofinetune[y_true == 0] >= emb_nofinetune_threshold)
        tpr_embed_nofinetune = np.mean(y_pred_embed_nofinetune[y_true == 1] >= emb_nofinetune_threshold)
        emb_nofinetune_bkgeff.append(fpr_embed_nofinetune)
        emb_nofinetune_sigeff.append(tpr_embed_nofinetune)

        fpr_embed_finetune = np.mean(y_pred_embed_finetune[y_true == 0] >= emb_finetune_threshold)
        tpr_embed_finetune = np.mean(y_pred_embed_finetune[y_true == 1] >= emb_finetune_threshold)
        emb_finetune_bkgeff.append(fpr_embed_finetune)
        emb_finetune_sigeff.append(tpr_embed_finetune)

        y_pred_stand = Y_pred_standalone[i].ravel()
        
        fpr_stand = np.mean(y_pred_stand[y_true == 0] >= stand_threshold)
        tpr_stand = np.mean(y_pred_stand[y_true == 1] >= stand_threshold)
        stand_bkgeff.append(fpr_stand)
        stand_sigeff.append(tpr_stand)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].plot(test_pu, stand_bkgeff, marker='.', linestyle='-', color='black', label='Standalone')
    axes[0].plot(test_pu, emb_nofinetune_bkgeff, marker='.', linestyle='-', color='blue', label='Embedding, no fine-tuning')
    axes[0].plot(test_pu, emb_finetune_bkgeff, marker='.', linestyle='-', color='red', label='Embedding, fine-tuning')
    axes[0].set_xlabel('PU')
    axes[0].set_ylabel('Bkg. eff.')
    axes[0].set_title(f'Single threshold fixed by a target bkg.eff.={threshold_by_target_bkgeff} at PU={test_pu[idx]}', fontsize=10)
    axes[0].grid(True)
    axes[0].axvline(x=test_pu[idx], color='orange', linestyle='-', alpha=1, label='Ref. PU')
    #axes[0].axhline(y=emb_bkgeff[idx], color='blue', linestyle='-', alpha=0.5)
    axes[0].legend()
    
    axes[1].plot(test_pu, stand_sigeff, marker='.', linestyle='-', color='black', label='Standalone')
    axes[1].plot(test_pu, emb_nofinetune_sigeff, marker='.', linestyle='-', color='blue', label='Embedding, no fine-tuning')
    axes[1].plot(test_pu, emb_finetune_sigeff, marker='.', linestyle='-', color='red', label='Embedding, fine-tuning')
    axes[1].set_xlabel('PU')
    axes[1].set_ylabel('Sig. eff.')
    axes[1].set_title(f'Single threshold fixed by a target bkg.eff.={threshold_by_target_bkgeff} at PU={test_pu[idx]}', fontsize=10)
    axes[1].grid(True)
    axes[1].axvline(x=test_pu[idx], color='orange', linestyle='-', alpha=1, label='Ref. PU')
    #axes[1].axhline(y=emb_sigeff[idx], color='blue', linestyle='-', alpha=0.5)
    #axes[1].axhline(y=stand_sigeff[idx], color='blue', linestyle='-', alpha=0.5)
    axes[1].set_ylim((0,1))
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

