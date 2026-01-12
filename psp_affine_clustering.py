#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim

# Optional import (only needed for --solver bpdn or --solver lasso)
try:
    import spgl1
except Exception:
    spgl1 = None


# ----------------------------
# Patch helpers (your originals)
# ----------------------------
def return_overlapping_tiles(image, tile_width, step_size):
    """
    Return flattened tiles and a per-pixel patch-count map.
    """
    height, width = image.shape
    tiles1d = []
    patch_count = np.zeros_like(image, dtype=int)
    for i in range(0, height - tile_width + 1, step_size):
        for j in range(0, width - tile_width + 1, step_size):
            tile = image[i:i+tile_width, j:j+tile_width]
            tiles1d.append(tile.flatten())
            patch_count[i:i+tile_width, j:j+tile_width] += 1
    tiles1d = np.array(tiles1d)
    return tiles1d, patch_count


def reconstruct_image(imtiles1d, patch_count, tile_width, original_shape, step_size):
    """
    Reassemble image from tiles and average overlaps.
    """
    height, width = original_shape
    reconstructed_image = np.zeros((height, width), dtype=float)
    patch_idx = 0
    for i in range(0, height - tile_width + 1, step_size):
        for j in range(0, width - tile_width + 1, step_size):
            patch = imtiles1d[patch_idx].reshape(tile_width, tile_width)
            reconstructed_image[i:i+tile_width, j:j+tile_width] += patch
            patch_idx += 1
    # safe divide
    safe = patch_count.astype(float).copy()
    safe[safe == 0] = 1.0
    reconstructed_image /= safe
    return reconstructed_image


# ----------------------------
# SSC helpers (your originals)
# ----------------------------
def calculate_medoid(cluster):
    distances = cdist(cluster, cluster, metric='euclidean')
    total_distances = np.sum(distances, axis=1)
    medoid_index = np.argmin(total_distances)
    return cluster[medoid_index]


def get_cluster_medoids(data, cluster_indices):
    unique_clusters = np.unique(cluster_indices)
    clusters = {c: [] for c in unique_clusters}
    for i, c in enumerate(cluster_indices):
        clusters[c].append(data[i])
    medoids = {c: calculate_medoid(np.array(points)) for c, points in clusters.items()}
    means = {c: np.mean(np.array(points), axis=0) for c, points in clusters.items()}
    return clusters, medoids, means


def pca_for_cluster(cluster):
    """Return PCA transform, cumulative explained variance, and components."""
    assert isinstance(cluster, np.ndarray)
    pca = PCA()
    cluster_pca = pca.fit_transform(cluster)
    pca_vectors = pca.components_
    data_dim = cluster.shape[-1]
    padding_size = max(0, data_dim - len(pca.explained_variance_ratio_))
    expl_var_ratio_cumul = np.cumsum(np.pad(pca.explained_variance_ratio_, (0, padding_size),
                                            'constant', constant_values=0))
    return cluster_pca, expl_var_ratio_cumul, pca_vectors


def fit_to_basis(data_vectors, basis_vectors):
    """
    basis_vectors: r x N (rows = mean + PCs)
    data_vectors : m x N
    """
    G = basis_vectors @ basis_vectors.T
    P = basis_vectors.T @ (np.linalg.pinv(G) @ basis_vectors)
    approximations = data_vectors @ P
    errors = np.linalg.norm(data_vectors - approximations, axis=1)
    return approximations, errors


# ----------------------------
# Solver wrapper (ElasticNet / SPGL1-BPDN / SPGL1-LASSO)
# ----------------------------
def solve_coefficients(A, b, args):
    """
    Solve for x in:
      elasticnet:   min_x 0.5||Ax-b||_2^2 + alpha * ||x||_1  (with l1_ratio=1 → Lasso; supports positivity)
      bpdn:         min_x ||x||_1  s.t. ||Ax-b||_2 <= sigma
      lasso (spgl1): min_x 0.5||Ax-b||_2^2 + tau * ||x||_1
    A: (n, M) dictionary (columns are atoms)
    b: (n,)
    returns x: (M,)
    """
    if args.solver == "elasticnet":
        enet = ElasticNet(alpha=args.enet_alpha,
                          l1_ratio=args.enet_l1_ratio,
                          positive=args.enet_positive,
                          fit_intercept=False,
                          max_iter=args.enet_max_iters,
                          selection="cyclic")
        enet.fit(A, b)
        return enet.coef_.astype(float)

    if spgl1 is None:
        raise RuntimeError(f"spgl1 is not installed, but --solver {args.solver} was requested.")

    # Column normalization (recommended for SPGL1)
    if args.spgl1_normalize_columns:
        col_norms = np.linalg.norm(A, axis=0) + 1e-12
        A_use = A / col_norms
    else:
        col_norms = None
        A_use = A

    if args.solver == "bpdn":
        if args.bpdn_sigma_mode == "relative":
            sigma = args.bpdn_sigma_rel * (np.linalg.norm(b, 2) + 1e-12)
        else:
            sigma = float(args.bpdn_sigma_abs)
        x_use, resid, grad, info = spgl1.spg_bpdn(A_use, b, sigma, verbosity=args.spgl1_verbosity)
        x_use = np.asarray(x_use).reshape(-1)

    elif args.solver == "lasso":
        # tau is the L1 penalty weight
        tau = float(args.lasso_tau)
        x_use, resid, grad, info = spgl1.spg_lasso(A_use, b, tau, verbosity=args.spgl1_verbosity)
        x_use = np.asarray(x_use).reshape(-1)

    else:
        raise ValueError("Unknown solver: {}".format(args.solver))

    # de-normalize coefficients if columns were normalized
    if col_norms is not None:
        x = x_use / col_norms
    else:
        x = x_use
    return x


# ----------------------------
# Main (single image)
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Subspace Clustering with ElasticNet / SPGL1-BPDN / SPGL1-LASSO")

    # Data / noise / tiling
    parser.add_argument("--image-number", type=int, default=3)
    parser.add_argument("--dataset-dir", type=str, default="Dataset/baseline_data")
    parser.add_argument("--tile-w", type=int, default=8)
    parser.add_argument("--step-size", type=int, default=8)
    parser.add_argument("--std-dev", type=float, default=20.0)
    parser.add_argument("--results-dir", type=str, default=None)

    # Graph / clustering
    parser.add_argument("--clusters-K", type=int, default=20, help="Number of spectral clusters (L_hat)")

    # Solver choice
    parser.add_argument("--solver", type=str, choices=["elasticnet", "bpdn", "lasso"], default="lasso")

    # ElasticNet params
    parser.add_argument("--enet-alpha", type=float, default=10.0)
    parser.add_argument("--enet-l1-ratio", type=float, default=1.0)
    parser.add_argument("--enet-positive", action="store_true", default=False,
                        help="Use nonnegative coefficients for ElasticNet")
    parser.add_argument("--enet-max-iters", type=int, default=5000)

    # BPDN (SPGL1) params
    parser.add_argument("--bpdn-sigma-mode", type=str, choices=["relative", "absolute"], default="relative")
    parser.add_argument("--bpdn-sigma-rel", type=float, default=0.05,
                        help="sigma = bpdn_sigma_rel * ||b||_2 if sigma-mode=relative")
    parser.add_argument("--bpdn-sigma-abs", type=float, default=5.0,
                        help="fixed sigma if sigma-mode=absolute")

    # LASSO (SPGL1) param
    parser.add_argument("--lasso-tau", type=float, default=1.0,
                        help="L1 penalty weight for spg_lasso (objective = 0.5||Ax-b||^2 + tau||x||_1)")

    # SPGL1 common
    parser.add_argument("--spgl1-normalize-columns", dest="spgl1_normalize_columns",
                        action="store_true", default=True,
                        help="Normalize columns of A before solving (recommended)")
    parser.add_argument("--spgl1-verbosity", type=int, default=0)

    args = parser.parse_args()

    # Prepare output directory
    if args.results_dir is None:
        args.results_dir = f"results/FixedNum_{args.solver}/tilw{args.tile_w}_step{args.step_size}_noise{int(args.std_dev)}"
    os.makedirs(args.results_dir, exist_ok=True)

    # ---------------------------
    # Load + preprocess the image
    # ---------------------------
    img_path = os.path.join(args.dataset_dir, f"Image{args.image_number}.png")
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read: {img_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # crop to multiple of tile_w
    mindim = int((min(image.shape) // args.tile_w) * args.tile_w)
    image = image[:mindim, :mindim]
    print("image shape:", image.shape)

    # Save clean + noisy (distinct names)
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.savefig(os.path.join(args.results_dir, f"image_{args.image_number}_clean.png"),
                bbox_inches='tight', pad_inches=0)
    plt.close()

    noisy_image = np.uint8(np.clip(image + np.random.normal(scale=args.std_dev, size=image.shape), 0, 255))
    plt.imshow(noisy_image, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.savefig(os.path.join(args.results_dir, f"image_{args.image_number}_noisy.png"),
                bbox_inches='tight', pad_inches=0)
    plt.close()

    # ---------------------------
    # Tiling
    # ---------------------------
    im_tiles1d, patch_count = return_overlapping_tiles(noisy_image, args.tile_w, args.step_size)
    im_tiles1d = im_tiles1d.astype(float)
    print("tiles shape:", im_tiles1d.shape, "patch_count shape:", patch_count.shape)

    # ---------------------------
    # Step 1: self-expressive coding
    # ---------------------------
    Y = im_tiles1d
    N, n = Y.shape
    B = np.zeros((N, N))
    out_txt = os.path.join(args.results_dir, f"image_{args.image_number}_prints.txt")
    output_file = open(out_txt, "a")

    for i in range(N):
        y_i = Y[i, :]
        y_others = np.delete(Y, i, axis=0)
        A = y_others.T       # (n, N-1)
        b = y_i.T            # (n,)

        try:
            x = solve_coefficients(A, b, args)   # <-- solver switch
        except Exception as e:
            print(f"[warn] solver failed at i={i}: {e}")
            x = np.zeros(A.shape[1], dtype=float)

        result = x  # length N-1

        # if i % 10 == 0:
        #     print(f"{i} th tile result:", file=output_file)
        #     print("L1 norm b", np.linalg.norm(b, 1), file=output_file)
        #     print("L1 norm x", np.linalg.norm(result, 1), file=output_file)
        #     print("sum of x", np.sum(result), file=output_file)
        #     print("L2 norm Ax-b", np.linalg.norm(A @ result - b, 2), file=output_file)

        beta_i = np.insert(result, i, 0.0)
        B[i, :] = np.abs(beta_i)

    print("checking rough beta range:", B[1], file=output_file)
    output_file.close()

    # ---------------------------
    # Step 2: similarity graph
    # ---------------------------
    W = np.abs(B) + np.abs(B.T)

    # ---------------------------
    # Steps 3–5: normalized Laplacian + eigen-gaps (kept for reference)
    # ---------------------------
    D = np.diag(np.sum(W, axis=1))
    D_safe = D.copy()
    D_safe[D_safe == 0] = 1e-12
    D_sqrt_inv = np.linalg.inv(np.sqrt(D_safe))
    L_norm = D_sqrt_inv @ (D - W) @ D_sqrt_inv

    eigenvalues, _ = np.linalg.eigh(L_norm)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    differences = np.diff(sorted_eigenvalues)
    i_max = np.argmax(differences)
    # You fixed K in your code; we honor that:
    L_hat = int(args.clusters_K)

    # S = I - L_norm (unused, kept to mirror your original)
    _ = np.eye(L_norm.shape[0]) - L_norm

    # ---------------------------
    # Step 6: spectral clustering
    # ---------------------------
    spectral_clustering = SpectralClustering(n_clusters=L_hat, affinity='precomputed', random_state=0)
    labels = spectral_clustering.fit_predict(W)

    with open(out_txt, "a") as f:
        print("Estimated_Clusters:", L_hat, file=f)
        print("shape of labels:", labels.shape, file=f)
        try:
            print("determinant of W similarity matrix:", np.linalg.det(W), file=f)
        except Exception as e:
            print(f"determinant(W) failed: {e}", file=f)

    # ---------------------------
    # Visual diagnostics 
    # ---------------------------
    block_size = 128
    block_cnt = int(im_tiles1d.shape[0] / block_size)
    if block_cnt >= 1:
        fig, axes = plt.subplots(block_cnt, block_cnt, figsize=(30, 30))
        for i in range(block_cnt):
            for j in range(block_cnt):
                W_matrix = W[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                Wmin, Wmax = np.min(W_matrix), np.max(W_matrix)
                denom = (Wmax - Wmin) if (Wmax - Wmin) != 0 else 1.0
                normalized_W_matrix = (W_matrix - Wmin) / denom
                axes[i, j].imshow(normalized_W_matrix, cmap='viridis', interpolation='none')
                axes[i, j].axis('off')
        plt.savefig(os.path.join(args.results_dir, f"image_{args.image_number}_similarities.png"))
        plt.close()

    for patch_num_ex in [3, 7, 10, 328]:
        if patch_num_ex < W.shape[0]:
            data = W[patch_num_ex, :]
            top_count = min(5, len(data))
            top_indices = np.argpartition(data, -top_count)[-top_count:]
            top_indices = top_indices[np.argsort(data[top_indices])][::-1]
            with open(out_txt, "a") as f:
                print(f"top {top_count} similarities (patch {patch_num_ex}) indices:", top_indices, file=f)
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(data)), data)
            plt.xlabel('patch number')
            plt.ylabel('value in similarity graph')
            plt.ylim(0, 1)
            plt.title(f'variation in similarity scores with patch {patch_num_ex}')
            plt.savefig(os.path.join(args.results_dir, f"image_{args.image_number}_similarity_w_patch_{patch_num_ex}.png"))
            plt.close()

    if W.shape[0] > 3:
        patch_num_ex = 3
        data = W[patch_num_ex, :]
        top_count = min(3, len(data))
        top_indices = np.argpartition(data, -top_count)[-top_count:]
        top_indices = top_indices[np.argsort(data[top_indices])][::-1]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        trip = [im_tiles1d[idx] for idx in top_indices]
        vmin = min(t.min() for t in trip)
        vmax = max(t.max() for t in trip)
        titles = ['Highest patch', '2nd highest patch', '3rd highest patch']
        for k in range(top_count):
            imgk = trip[k].reshape(args.tile_w, args.tile_w)
            axes[k].imshow(imgk, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[k].set_title(titles[k])
            axes[k].axis('off')
        plt.savefig(os.path.join(args.results_dir, f"image_{args.image_number}_top3_similarity_patches.png"))
        plt.close()

    # ----------------------------------------------------
    # Cluster medoids / means → PCA bases → reconstruction
    # ----------------------------------------------------
    Spectral_cluster_indices = labels
    clustered_data, cluster_medoids, cluster_means = get_cluster_medoids(im_tiles1d, Spectral_cluster_indices)
    print('keys of clustered_data:', clustered_data.keys())
    first_key = list(clustered_data.keys())[0]
    print('clustered_data[first] shape', np.array(clustered_data[first_key]).shape)
    print('cluster_medoids[first] shape', cluster_medoids[first_key].shape)
    print('cluster_means[first] shape', cluster_means[first_key].shape)

    def get_centered_clusters(clustered_data, cluster_means):
        return {c: (np.array(clustered_data[c]) - cluster_means[c]) for c in clustered_data.keys()}

    centered_clusters = get_centered_clusters(clustered_data, cluster_means)

    # Build dynamic/fixed PCA bases
    t_exp = 0.9
    dynamic_psi = dict()
    fixed_psi = dict()
    dim_comp = 0.5
    fixed_cut = int(dim_comp * (args.tile_w * args.tile_w))

    for c, points in centered_clusters.items():
        cluster_pca, expln_var_cum, pca_vectors = pca_for_cluster(points)
        cutidx = int(np.argmax(expln_var_cum >= t_exp))
        dynamic_basis = np.vstack((np.array([cluster_means[c]]), pca_vectors[:cutidx]))
        fixed_basis = np.vstack((np.array([cluster_means[c]]), pca_vectors[:fixed_cut]))
        dynamic_psi[c] = dynamic_basis
        fixed_psi[c] = fixed_basis
        print(f'Cluster {c} dynamic_basis vectors shape:', dynamic_basis.shape)
        print(f'Cluster compression (≥{t_exp:.2f} var) = {dynamic_basis.shape[0] / points.shape[0]:.4f}')

    # Compare reconstruction errors per cluster (dynamic vs fixed)
    dyn_errors, fix_errors = [], []
    for c, points in clustered_data.items():
        pts = np.array(points)
        dyn_approx, dyn_errs = fit_to_basis(pts, dynamic_psi[c])
        fix_approx, fix_errs = fit_to_basis(pts, fixed_psi[c])
        dyn_errors.append(np.mean(dyn_errs))
        fix_errors.append(np.mean(fix_errs))
        print(f"cluster {c} mean L2 error: dynamic={np.mean(dyn_errs):.4f}, fixed({dim_comp:.2f})={np.mean(fix_errs):.4f}")

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
    axs[0].bar(np.arange(len(dyn_errors)), dyn_errors, label=f'Dynamic (≥{t_exp:.2f} var)')
    axs[0].set_xlabel('Cluster'); axs[0].set_ylabel('Mean L2 error'); axs[0].legend()
    axs[1].bar(np.arange(len(fix_errors)), fix_errors, label=f'Fixed top {dim_comp:.2f}')
    axs[1].set_xlabel('Cluster'); axs[1].set_ylabel('Mean L2 error'); axs[1].legend()
    plt.savefig(os.path.join(args.results_dir, f"image_{args.image_number}_Choice_Basis.png"))
    plt.close()

    # Reconstruct full image using fixed basis per cluster (as in your code)
    def visualise_approx(im_tiles1d, cluster_indices):
        approx_data1d = np.zeros_like(im_tiles1d)
        error_data = np.zeros_like(im_tiles1d)
        for i in range(len(cluster_indices)):
            fix_approx, fix_errs = fit_to_basis(im_tiles1d[i][np.newaxis, :], fixed_psi[cluster_indices[i]])
            approx_data1d[i] = fix_approx
            error_data[i] = fix_errs
        approx_image = reconstruct_image(approx_data1d, patch_count, args.tile_w, (mindim, mindim), args.step_size)
        return approx_image

    approx_image = visualise_approx(im_tiles1d, Spectral_cluster_indices)
    approx_image = approx_image[:mindim, :mindim]

    # Save reconstructions
    plt.imshow(approx_image, cmap='viridis')
    plt.colorbar()
    plt.savefig(os.path.join(args.results_dir, f"image_{args.image_number}_ApproxImage.png"))
    plt.close()

    plt.imshow(approx_image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.savefig(os.path.join(args.results_dir, f"image_{args.image_number}_ApproxImage_Gray.png"),
                bbox_inches='tight', pad_inches=0)
    plt.close()

    # Metrics (use uint8 for PSNR; SSIM with data_range=255)
    approx_u8 = np.uint8(np.clip(approx_image, 0, 255))
    clean_u8 = image.astype(np.uint8)

    MSE = np.mean((approx_u8.astype(np.float32) - clean_u8.astype(np.float32)) ** 2)
    PSNR = cv2.PSNR(clean_u8, approx_u8)
    ssim_value, ssim_map = ssim(approx_u8.astype(np.float32), clean_u8.astype(np.float32),
                                data_range=255, full=True)

    with open(out_txt, "a") as f:
        print('MSE:', MSE, file=f)
        print('PSNR:', PSNR, file=f)
        print('SSIM:', ssim_value, file=f)

    plt.imshow(ssim_map, cmap='viridis')
    plt.colorbar()
    plt.savefig(os.path.join(args.results_dir, f"image_{args.image_number}_SSIM_Map.png"))
    plt.close()


if __name__ == "__main__":
    main()