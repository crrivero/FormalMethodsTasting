# For computer scientists
You can find a number of Jupyter notebooks for:

- Computer organization:
  + Verify that an OR-of-ANDs can be represented as a NAND-of-NANDs [link](NAND.ipynb).

- Artificial intelligence:
  + Solve the ROOKS problem [link](ROOKS.ipynb).

- Theory:
  + Find maximum matching in a bipartite graph [link](MATCHING.ipynb).
  + Construct a solver for k-Clique [link](CLIQUE.ipynb).
  + Verify a sender gadget for K3K3-NONARROWING. [link](K3K3-NONARROWING.ipnyb). <details>
  <summary>Click to expand description</summary>
A K3 is the complete graph on three vertices. Given a graph G, the K3K3-NONARROWING problem is to determine whether a graph has a K3K3-good coloring, where a K3K3-good coloring is a red/blue coloring of G’s edges such that there are no red K3’s and no blue K3’s. Such a coloring is called a K3K3-good coloring. This is illustrated below: <img width="500" alt="K3K3-example" src="https://github.com/user-attachments/assets/86cc4f5c-14b4-4489-a484-c069d9448aa9" /> K3K3-NONARROWING is an NP-complete problem that asks whether a given graph has a red/blue edge-coloring such that there are no red K3's and no blue K3's. The NP-hardness proof of this problem relies on a special graph called a K3K3-sender: a graph G with "sender edges" e and f such that e and f are distinct and e has the same color as f in all K3K3-good colorings of G. The objective of this notebook was to construct and verify the correctness of a K3K3-sender.
</details>

