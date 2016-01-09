import numpy as np
import itertools


def get_top_block_links(com_nodes_set, other_nodes_set, num_links, top_measure, com_internal_links=True):
    if com_internal_links:
        other_nodes_set |= com_nodes_set

    sorted_other_nodes = sorted(other_nodes_set, key=lambda x: top_measure[x], reverse=True)
    sorted_com_nodes = sorted(com_nodes_set, key=lambda x: top_measure[x], reverse=True)
    new_edges = list()
    while len(new_edges) < num_links:
        remaining_links = num_links - len(new_edges)
        block_size = int(np.sqrt(remaining_links)) + 1
        all_com_nodes = False
        if block_size > len(com_nodes_set):
            all_com_nodes = True
            block_size = int(remaining_links/len(com_nodes_set)) + 1

        sorted_com_nodes_block = sorted_com_nodes if all_com_nodes else sorted_com_nodes[:block_size]
        sorted_other_nodes_block = sorted_other_nodes[:block_size]

        new_edges.extend(itertools.islice(
                        ((src, dest) for dest in sorted_com_nodes_block for src in sorted_other_nodes_block if src != dest),
                        remaining_links))
    assert len(new_edges) == num_links
    return new_edges


def get_random_links(com_nodes_set, other_nodes_set, num_links, com_internal_links=True):
    if com_internal_links:
        other_nodes_set |= com_nodes_set

    new_edges = list()
    other_nodes = np.array(list(other_nodes_set))
    com_nodes = np.array(list(com_nodes_set))
    while len(new_edges) < num_links:
        remaining_links = num_links - len(new_edges)
        srcs = np.random.choice(other_nodes, size=remaining_links, replace=True)
        dests = np.random.choice(com_nodes, size=remaining_links, replace=True)
        new_edges.extend(list(filter(lambda (s, d): s != d, zip(srcs, dests))))
    assert len(new_edges) == num_links
    return new_edges
