import torch

from src import ELEMENT_TYPES, IGNORED_QUERY_ELEMENTS

def evaluate_ranking(data, top_k=10):
    """
    Evaluates how well top-ranked nodes match the query's relevant page range,
    while ignoring uninformative nodes based on node type.

    Args:
    - query_embeddings: Tensor (num_queries, embedding_dim) → Query embeddings.
    - node_embeddings: Tensor (num_nodes, embedding_dim) → Node embeddings.
    - page_ids: List of integers (num_nodes,) → Page numbers corresponding to each node.
    - node_types: List of integers (num_nodes,) → Node type IDs.
    - query_page_ranges: List of (start_page, end_page) tuples (num_queries,) → Relevant page range for each query.
    - ignored_node_types: Set of node type IDs to ignore.
    - top_k: Number of top-ranked nodes to check.

    Returns:
    - accuracy: Percentage of top-ranked nodes that fall within the correct page range.
    """
    num_queries = data.query_text_embedding.shape[0]
    correct_queries = 0  # Number of queries where *all* top-K nodes are correct

    ignored_node_types = [ELEMENT_TYPES.index(node_type) for node_type in IGNORED_QUERY_ELEMENTS]

    # Mask: Ignore nodes with uninformative text embeddings
    informative_mask = torch.tensor([nt not in ignored_node_types for nt in data.node_type], device=data.text_embedding.device)

    # Filter out uninformative nodes
    filtered_node_embeddings = data.text_embedding[informative_mask]  # Shape: (num_valid_nodes, embedding_dim)
    filtered_page_ids = [pid for i, pid in enumerate(data.page_id) if informative_mask[i]]  # Keep valid page IDs

    # Compute cosine similarity (query vs. all valid nodes)
    similarities = torch.mm(data.query_text_embedding, filtered_node_embeddings.T)  # Shape: (num_queries, num_valid_nodes)

    for i in range(num_queries):
        query_page_range = data.query_page_range[i]

        # Sort nodes by similarity (descending order)
        ranked_indices = torch.argsort(similarities[i], descending=True)  # Sorted node indices

        # Get the top-k ranked nodes
        top_nodes = ranked_indices[:top_k]

        print([ELEMENT_TYPES[data.node_type[j]] for j in top_nodes])

        # Check if *all* top-K nodes fall within the page range
        all_correct = all(
            query_page_range[0] <= filtered_page_ids[node_idx] <= query_page_range[1]
            for node_idx in top_nodes
        )

        if all_correct:
            correct_queries += 1

    # Compute strict accuracy: percentage of queries where *all* top-K nodes were correct
    strict_accuracy = correct_queries / num_queries
    print(f"Strict Top-{top_k} Accuracy: {strict_accuracy:.4f}")

    return strict_accuracy

# evaluate_ranking(data, top_k=5)