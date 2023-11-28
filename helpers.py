import numpy as np
import random
import pandas as pd

def nin(x, y):
    return not (x in y)

# Ordered matrices ----
# These functions are only valid when columns/rows are ordered in terms of mutation appearing order. They will yield incorrect values for unordered B matrices

def get_parent_ordered(B, mutation_idx):
    row = np.copy(B[mutation_idx, :])
    row[mutation_idx] = 0
    ascendants = [i for i, val in enumerate(row) if val == 1]
    if len(ascendants):
        return max(ascendants)
    else:
        return ascendants
    

def get_descendants_ordered(B, mutation_idx):
    column = np.copy(B[:, mutation_idx])
    column[mutation_idx] = 0
    return [i for i, val in enumerate(column) if val == 1]


def get_ascendants_ordered(B, mutation_idx):
    row = np.copy(B[mutation_idx, :])
    row[mutation_idx] = 0
    return np.where(row == 1)[0]


def get_new_B_row_ordered(node_idx, B, parent_idx):
    new_row = np.copy(B[parent_idx, :])
    new_row[node_idx] = 1
    return new_row


# Unordered matrices ----
# These functions are valid for unordered B matrices. Based on Ancestree's 2nd condition for a valid B matrix

def get_parent(B, node_idx):
    n = B.shape[0]
    indeces = np.arange(n)
    subtraction_values = np.array([np.sum(B[node_idx, :] - B[x, :]) for x in indeces])
    possible_nodes = indeces[np.where(subtraction_values == 1)[0]]
    parent_logical = np.array([np.all(B[node_idx, :] >= B[x, :]) for x in possible_nodes])
    parent = possible_nodes[np.where(parent_logical)][0] if len(np.where(parent_logical)[0]) > 0 else None
    return parent if parent is not None else np.nan


def get_ascendants(B, node_idx):
    parent = get_parent(B, node_idx)
    if not np.isnan(parent):
        return [parent] + get_ascendants(B, parent)
    else:
        return []


def get_children(B, node_idx):
    n = B.shape[0]
    indeces = np.arange(n)
    subtraction_values = np.array([np.sum(B[x, :] - B[node_idx, :]) for x in indeces])
    possible_nodes = indeces[np.where(subtraction_values == 1)[0]]
    children_logical = np.array([np.all(B[x, :] >= B[node_idx, :]) for x in possible_nodes])
    children = possible_nodes[np.where(children_logical)]
    if len(children) > 0:
        return children.tolist()
    else:
        return []


def get_descendants(B, node_idx):
    children = get_children(B, node_idx)
    if len(children) > 0:
        descendants = [child for child in children]
        for child in children:
            descendants.extend(get_descendants(B, child))
        return descendants
    else:
        return []


# Other helpful functions ----

def get_new_parent_opts(node, parents, B):
    descendants = get_descendants(B, node)
    excluded_new_parent_opts = [node] + [parents[node]] + descendants
    n = B.shape[0]
    new_parent_opts = np.setdiff1d(np.arange(n), excluded_new_parent_opts)
    return new_parent_opts.tolist()


def get_mutation_idx(B, node_idx):
    parent = get_parent(B, node_idx)
    if np.isnan(parent):
        mut_idx = np.where(B[node_idx, :] != 0)[0]
    else:
        mut_idx = np.where(B[parent, :] != B[node_idx, :])[0]
    
    return mut_idx.tolist()  # Convert to a Python list


def get_new_B_row(mut_idx, B, parent_idx):
    new_row = np.copy(B[parent_idx, :])
    new_row[mut_idx] = 1
    return new_row


def get_leaf_mutations(B):
    colsums = np.sum(B, axis=0)
    leaf_muts = np.where(colsums == 1)[0]
    return leaf_muts.tolist()


def get_siblings(B, mut_idx):
    n = B.shape[0]
    all_node_parents = [get_parent(B, x) for x in np.arange(n)]
    siblings = [i for i in np.arange(n) if all_node_parents[i] == all_node_parents[mut_idx] and i != mut_idx]
    return siblings


def get_edges(B):
    n = B.shape[0]
    all_node_parents = [get_parent(B, x) for x in np.arange(n)]
    edges = pd.DataFrame({'parent': all_node_parents, 'child': np.arange(n)})
    # Remove the edge at the root (where parent is NA)
    edges = edges.dropna(subset=['parent']).astype(int)
    return edges

