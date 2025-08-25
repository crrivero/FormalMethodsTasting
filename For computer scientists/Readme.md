# For computer scientists
You can find a number of Jupyter notebooks for:

- Computer organization:
  + Verify that an OR-of-ANDs can be represented as a NAND-of-NANDs [link](NAND.ipynb). <details><summary>Click to expand description</summary> A sum-of-products boolean (OR of ANDS) formula can be represented using only NAND gates. The objective of the notebook is to verify this statement for a very simple case, i.e., when a formula is the AND of two ORs: <br> <img width="500" alt="rooks" src="https://github.com/user-attachments/assets/1b176d1b-f328-495b-b2b6-46e0dcc6559a" />
 </details>

- Artificial intelligence:
  + Solve the ROOKS problem [link](ROOKS.ipynb). <details><summary>Click to expand description</summary> The ROOKS problem is a classic combinatorial puzzle in constraint satisfaction problems: the objective is to place N rooks on an NxN chessboard so that no two rooks are attacking each other (i.e., no two rooks are in the same row or column). For example, a solution on an 8x8 board is shown below. The objective of the notebook is to find all solutions for the ROOKS problem on a 4x4 board. <br> <img width="377" height="375" alt="rooks" src="https://github.com/user-attachments/assets/be4e42d4-2877-474f-ae1d-f1203317d694" /></details>


- Theory:
  + Find maximum matching in a bipartite graph [link](MATCHING.ipynb). <details><summary>Click to expand description</summary> A matching is a collection of edges in a graph where each edge is disjoint, i.e., shares no vertices. The objective of this notebook is to find a matching in a given bipartite graph.</details>

  + Construct a solver for k-Clique [link](CLIQUE.ipynb). <details><summary>Click to expand description</summary>For a graph G, a k-clique is a subset of vertices, S, of G such that: (1) each pair of vertices in S is adjacent, and (2) the cardinality of S is k. Given a graph G and integer k, the CLIQUE problem is to determine if a given graph has a k-clique. The objective of this notebook is to use Z3 to make a CLIQUE solver.</details>

  + Verify a sender gadget for K3K3-NONARROWING. [link](K3K3-NONARROWING.ipnyb). <details><summary>Click to expand description</summary>A K3 is the complete graph on three vertices. Given a graph G, the K3K3-NONARROWING problem is to determine whether a graph has a K3K3-good coloring, where a K3K3-good coloring is a red/blue coloring of G’s edges such that there are no red K3’s and no blue K3’s. Such a coloring is called a K3K3-good coloring. This is illustrated below: <br> <img width="500" alt="K3K3-example" src="https://github.com/user-attachments/assets/86cc4f5c-14b4-4489-a484-c069d9448aa9" /> <br> K3K3-NONARROWING is an NP-complete problem that asks whether a given graph has a red/blue edge-coloring such that there are no red K3's and no blue K3's. The NP-hardness proof of this problem relies on a special graph called a K3K3-sender: a graph G with "sender edges" e and f such that e and f are distinct and e has the same color as f in all K3K3-good colorings of G. The objective of this notebook is to construct and verify the correctness of a K3K3-sender.</details>
