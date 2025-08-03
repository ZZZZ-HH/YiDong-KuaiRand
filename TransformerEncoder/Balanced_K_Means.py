import torch 
import torch.nn as nn
import numpy as np
import random

def BalancedKMeans(V: torch.tensor, K: int, max_iter: int = 100):
    num_items, _ = V.shape
    w = num_items // K

    initial_indices = random.sample(range(num_items), K)
    centroids = V[initial_indices].clone() # Cl = {cl_1, ..., cl_K}
    for iteration in range(max_iter):
        assignments = [[] for _ in range(K)]
        unassigned_indices = list(range(num_items))

        for k in range(K):
            if not unassigned_indices:
                break

            U_vectors = V[unassigned_indices]
            distances = torch.norm(U_vectors - centroids[k], dim=1)
            _ , sorted_indices_in_U_vectors = torch.sort(distances)
            global_sorted_item_indices = [unassigned_indices[idx.item()] for idx in sorted_indices_in_U_vectors]
            num_to_assign = min(w, len(global_sorted_item_indices))
            assigned_indices_for_k = global_sorted_item_indices[:num_to_assign]
            assignments[k] = assigned_indices_for_k

            if len(assignments[k]) > 0:
                centroids[k] = torch.mean(V[assignments[k]], dim=0)
            else:
                pass

            assigned_set = set(assigned_indices_for_k)
            unassigned_indices = [idx for idx in unassigned_indices if idx not in assigned_set]

        if unassigned_indices:
            print(f"Iteration {iteration}: {len(unassigned_indices)} items remaining after balanced assignment. Distributing them.")
            remaining_vectors = V[unassigned_indices]
            all_dists_to_centroids = torch.cdist(remaining_vectors, centroids)
            closest_centroid_indices = torch.argmin(all_dists_to_centroids, dim=1)
            for i, item_idx in enumerate(unassigned_indices):
                closest_k = closest_centroid_indices[i].item()
                assignments[closest_k].append(item_idx)
            for k in range(K):
                if len(assignments[k]) > 0:
                    centroids[k] = torch.mean(V[assignments[k]], dim=0)
        new_labels = np.array([k for k, indices in enumerate(assignments) for _ in indices])
        if iteration == 0:
            old_labels = new_labels.copy()
        elif np.array_equal(new_labels, old_labels):
            break
        else:
            old_labels = new_labels.copy()

    print(f"Balanced K-means finished after {iteration+1} iterations.")
    return centroids, new_labels

def residual_quantize(X: torch.tensor, L=3, K=8):
    N, _ = X.shape
    residual = X.clone()
    tokens = np.zeros((N, L), dtype=int)
    codebooks = []

    for l in range(L):
        codebook, labels = BalancedKMeans(residual, K, 200)
        tokens[:, l] = labels
        selected_centers = codebook[labels]
        residual = residual - selected_centers
        codebooks.append(codebook)

    return tokens, codebooks
