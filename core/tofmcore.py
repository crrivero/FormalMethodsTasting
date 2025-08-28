from z3 import *
import re
import IPython.display

html_to_latex_dict = {
  r"&not;": r" \neg ",
  r"&or;": r" \lor ",
  r"&and;": r" \land ",
  r"&rArr;": r" \implies ",
  r"&lt;": r" < ",
  r"&gt;": r" > ",
  r"&le;": r" \leq ",
  r"&ge;": r" \geq ",
  r"&ne;": r" \not= ",
  r"&middot;": r" \cdot ",
  r"&exist;": r" \exists ",
  r"&forall;": r" \forall ",
}
html_to_latex_dict_pattern = re.compile("|".join(map(re.escape, html_to_latex_dict.keys())))

def HTMLtoLaTeX( s ):
  s2 = re.sub(r"If\((.*?), 1, 0\)", r'\1', s)
  s3 = re.sub(r"<sup>(.*?)</sup>", r'^\1', s2)
  return html_to_latex_dict_pattern.sub(lambda m: html_to_latex_dict[m.group(0)], s3)

def showSolver( solver ):
  set_pp_option("html_mode", True)
  for x in solver.assertions():
    IPython.display.display( IPython.display.Math( ( HTMLtoLaTeX( str(x) ) ) ) )
  set_pp_option("html_mode", False)

# Code to enumerate over all models in a Z3 solver
# Source: https://theory.stanford.edu/%7Enikolaj/programmingz3.html and https://stackoverflow.com/questions/11867611/z3py-checking-all-solutions-for-equation/
# s = solver, initial_terms = set of variables
def all_smt(s, initial_terms):
    def block_term(s, m, t):
        s.add(t != m.eval(t, model_completion=True))
    def fix_term(s, m, t):
        s.add(t == m.eval(t, model_completion=True))
    def all_smt_rec(terms):
        if sat == s.check():
           m = s.model()
           yield m
           for i in range(len(terms)):
               s.push()
               block_term(s, m, terms[i])
               for j in range(i):
                   fix_term(s, m, terms[j])
               yield from all_smt_rec(terms[i:])
               s.pop()
    yield from all_smt_rec(list(initial_terms))

def list_all_solutions( s, initial_terms ):
  return list( all_smt(s, initial_terms) )


############# Notebook specific things

import re
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

### Generic Helpers

def split_camel_case(s):
    match = re.match(r'^([a-z]+|[A-Z][a-z]*)(.*)$', s)
    if match:
        return match.group(1), match.group(2)
    else:
        return s, ""  # if no split point found

# min and max functions to use in Z3 constraints
def maxZ3_3( z1, z2, z3 ):
  return If( z1 >= z2, If( z1 >= z3, z1, z3 ), If( z2 >= z3, z2, z3 ) )

def minZ3_3( z1, z2, z3 ):
  return If( z1 <= z2, If( z1 <= z3, z1, z3 ), If( z2 <= z3, z2, z3 ) )

def maxZ3_2( z1, z2 ):
  return If( z1 >= z2, z1, z2 )

def minZ3_2( z1, z2 ):
  return If( z1 <= z2, z1, z2 )


### Matching functions

def draw_bipartite_graph(left_nodes, right_nodes, edges, matching=None):
    """
    Draws a bipartite graph with optional highlighting for matching edges.
    Fixes cropping by adding margins and using tight layout.
    """
    # Normalize matching edges
    matching_set = set(frozenset(e) for e in matching) if matching else set()

    # Assign positions for a nice left-right layout
    pos = {}
    pos.update((node, (0, i)) for i, node in enumerate(sorted(left_nodes, reverse=True)))
    pos.update((node, (1, i)) for i, node in enumerate(sorted(right_nodes, reverse=True)))

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(left_nodes)
    G.add_nodes_from(right_nodes)
    G.add_edges_from(edges)

    # Separate edges
    matching_edges = [tuple(e) for e in edges if frozenset(e) in matching_set]
    non_matching_edges = [tuple(e) for e in edges if frozenset(e) not in matching_set]

    plt.figure(figsize=(5, 5))  # slightly bigger than before

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=["skyblue" if n in left_nodes else "lightgreen" for n in G.nodes()],
        node_size=2000,
        edgecolors='black'
    )
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=non_matching_edges, edge_color="gray", width=1.5)
    if matching_edges:
        nx.draw_networkx_edges(G, pos, edgelist=matching_edges, edge_color="red", width=3)

    plt.axis("off")
    plt.margins(0.2)   # add space around graph
    plt.tight_layout()
    plt.show()

def print_matching_solution( solution ):
  left_vertices = set({})
  right_vertices = set({})
  edges = []
  matching = []
  for i in solution:
    val = solution[i]
    a, b = split_camel_case( str(i) )
    left_vertices.add( a )
    right_vertices.add( b )
    edges.append( (a,b) )
    if ( val ):
      matching.append( (a,b) )
  draw_bipartite_graph( left_vertices, right_vertices, edges, matching )

### Rooks

def solToRookArray( N, sol ):
  rook_array = np.zeros((N,N), dtype=int)
  # to view the solution we have defined a speicla function
  for i in range(1,N+1):
    for j in range(1,N+1):
      xij = Bool('x_{'+str(i)+str(j)+'}')
      if ( sol[xij] ):
        rook_array[i-1][j-1] = 1
  return rook_array

def draw_chessboard_with_rooks( N, sol ):
    N = 3
    rook_array = solToRookArray( N, sol )

    # Create checkerboard pattern
    chessboard = np.indices((N, N)).sum(axis=0) % 2

    # Define RGB colors
    light_gray = [1, 1, 1]  # RGB for light gray
    dark_gray = [0.65, 0.65, 0.65]   # RGB for dark gray

    # Build RGB board
    rgb_board = np.zeros((N, N, 3))
    for i in range(N):
        for j in range(N):
            rgb_board[i, j] = light_gray if chessboard[i, j] == 0 else dark_gray

    plt.figure(figsize=(3, 3))
    # Plot the board
    plt.imshow(rgb_board, extent=[0, N, 0, N])
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal')

    # Add rooks using Unicode ♖
    for i in range(N):
        for j in range(N):
            if rook_array[i, j] == 1:
                plt.text(j + 0.5, N - 1 - i + 0.4, '♖',
                         fontsize=20 * (8 / N),  # Scale font for larger boards
                         ha='center', va='center', color='black')

    plt.show()

def draw_multiple_chessboards_with_rooks(N, solutions, boards_per_row=3, board_size=3):
    if ( len(solutions) > 24 ):
      print("Too many solutions, only printing first 24")
    sols = solutions[0:24]
    rook_arrays = [ solToRookArray( N, sol ) for sol in sols ]
    num_boards = len(rook_arrays)
    rows = (num_boards + boards_per_row - 1) // boards_per_row

    fig, axes = plt.subplots(rows, boards_per_row, figsize=(board_size * boards_per_row, board_size * rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case rows or cols = 1

    for idx, rook_positions in enumerate(rook_arrays):
        ax = axes[idx]
        N = rook_positions.shape[0]
        assert rook_positions.shape == (N, N), f"Board {idx} must be NxN"

        # Create checkerboard pattern
        chessboard = np.indices((N, N)).sum(axis=0) % 2

        # Define RGB colors
        light_gray = [1, 1, 1]
        dark_gray = [0.65, 0.65, 0.65]

        # Build RGB board
        rgb_board = np.zeros((N, N, 3))
        for i in range(N):
            for j in range(N):
                rgb_board[i, j] = light_gray if chessboard[i, j] == 0 else dark_gray

        ax.imshow(rgb_board, extent=[0, N, 0, N])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        # Add rooks
        for i in range(N):
            for j in range(N):
                if rook_positions[i, j] == 1:
                    ax.text(j + 0.5, N - 1 - i + 0.5, '♖',
                            fontsize=20 * (8 / N),
                            ha='center', va='center', color='black')

    # Hide any unused subplots
    for k in range(num_boards, len(axes)):
        axes[k].axis('off')

    plt.tight_layout()
    plt.show()

def rooks_output_string( all_solutions ):
  return ''.join(sorted([ ''.join(''.join(str(cell) for cell in row) for row in solToRookArray( 4, sol )) for sol in all_solutions[:26] ]))


### Arrowing

def drawJ4 ( G ):
  pos = { 0: (-1,1), 1: (1,1), 2: (1,-1), 3:(-1,-1) }
  nx.draw_networkx_nodes(G, pos, G.nodes(), node_color="black", node_size=150)
  nx.draw_networkx_labels(G, pos, font_size=10, font_color='white')
  nx.draw_networkx_edges(G, pos,edgelist=G.edges(),width=5,alpha=0.7,edge_color="black")

def drawG ( G ):
  pos = { 0: (-3,2), 1: (-3,-2), 2: (0,1), 3:(-0.5,-1), 4:(0.5,-1), 5:(3,2), 6:(3,-2) }
  nx.draw_networkx_nodes(G, pos, G.nodes(), node_color="black", node_size=150)
  nx.draw_networkx_labels(G, pos, font_size=10, font_color='white')
  sE = [(0,1), (5,6)]
  rE = list(G.edges()).copy()
  rE.remove( (0,1) )
  rE.remove( (5,6) )
  nx.draw_networkx_edges(G, pos,edgelist=sE,width=5,alpha=0.7,edge_color="black")
  nx.draw_networkx_edges(G, pos,edgelist=rE,width=2,alpha=0.7,edge_color="black")

def drawGandColoring( G, eMapInv, colorings, index ):
  pos = { 0: (-3,2), 1: (-3,-2), 2: (0,1), 3:(-0.5,-1), 4:(0.5,-1), 5:(3,2), 6:(3,-2) }
  nx.draw_networkx_nodes(G, pos, G.nodes(), node_color="black", node_size=150)
  nx.draw_networkx_labels(G, pos, font_size=10, font_color='white')

  bE = [ eMapInv[v.name()] for v in colorings[index] if ( colorings[index][v] == True ) ]
  rE = [ eMapInv[v.name()] for v in colorings[index] if ( colorings[index][v] == False ) ]

  nx.draw_networkx_edges(G, pos,edgelist=bE,width=3.5,alpha=0.7,edge_color="blue")
  nx.draw_networkx_edges(G, pos,edgelist=rE,width=3.5,alpha=0.7,edge_color="red")


### BST

# Source: Joel from https://stackoverflow.com/questions/33439810/preserving-the-left-and-right-child-while-printing-python-graphs-using-networkx
def binary_tree_layout(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5,
                  pos = None, parent = None):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node of current branch
       width: horizontal space allocated for this branch - avoids overlap with other branches
       vert_gap: gap between levels of hierarchy
       vert_loc: vertical location of root
       xcenter: horizontal location of root
       pos: a dict saying where all nodes go if they have been assigned
       parent: parent of this branch.
       each node has an attribute "left: or "right"'''
    if pos == None:
        pos = {root:(xcenter,vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    neighbors = list(G.neighbors(root))
    if parent != None:
        neighbors.remove(parent)
    if len(neighbors)!=0:
        dx = width/2.
        leftx = xcenter - dx/2
        rightx = xcenter + dx/2
        for neighbor in neighbors:
            if G.nodes[neighbor]['child_status'] == 'left':
                pos = binary_tree_layout(G,neighbor, width = dx, vert_gap = vert_gap,
                                    vert_loc = vert_loc-vert_gap, xcenter=leftx, pos=pos,
                    parent = root)
            elif G.nodes[neighbor]['child_status'] == 'right':
                pos = binary_tree_layout(G,neighbor, width = dx, vert_gap = vert_gap,
                                    vert_loc = vert_loc-vert_gap, xcenter=rightx, pos=pos,
                    parent = root)
    return pos


def printBST( B, verbose = False ):
  G = nx.Graph()

  queue = [B.root]
  G.add_node(B.root.ID)
  while len(queue) > 0:
    current_node = queue.pop()
    G.nodes[current_node.ID]['value'] = current_node.value
    if ( verbose ):
      G.nodes[current_node.ID]['IDV'] = (current_node.ID, current_node.value)
    if current_node.left != None:
      G.add_edge( current_node.ID, current_node.left.ID)
      G.nodes[current_node.left.ID]['child_status'] = 'left'
      queue.append(current_node.left)
    if current_node.right != None:
      G.add_edge( current_node.ID, current_node.right.ID)
      G.nodes[current_node.right.ID]['child_status'] = 'right'
      queue.append(current_node.right)

  pos = binary_tree_layout( G, B.root.ID )
  # nodes
  options = {"node_size": 300, "alpha": 0.9}
  nx.draw_networkx_nodes(G, pos, node_color="#bbb", **options)
  nx.draw_networkx_edges(G, pos, width=2.0, arrows=True)
  if ( verbose ):
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G,'IDV'), font_size=12)
  else:
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G,'value'), font_size=12)

### CLIQUE 

def visualize_graph( edges, clique=[] ):
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)

    # Set colors
    node_colors = ["lightcoral" if n in clique else "skyblue" for n in G.nodes()]
    edge_colors = ["red" if u in clique and v in clique else "gray" for u, v in G.edges()]

    # Layout and draw
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors,
        node_size=1200, font_size=10, width=2)
    plt.show()

def visualize_clique_solution( edges, solution ):
    verts = set([ e[0] for e in edges ]).union( [e[1] for e in edges] )
    clique = [v for v in verts if is_true(solution[v])]
    visualize_graph(edges, clique)

### PPI CLIQUE

def visualize_ppi_network( edges, clique=[] ):
    visualize_graph(edges, clique)

def visualize_ppi_solution( edges, solution ):
    visualize_clique_solution( edges, solution )


### COMPATIBLE COMPONENT CLIQUE

def visualize_compatible_components(edges, clique=[]):
    visualize_graph(edges, clique)

def visualize_compatible_components_solution( edges, solution ):
    visualize_clique_solution( edges, solution )

### BIG O

import math
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
from z3 import *

def _z3_value_to_float(zv):
    """Convert a Z3 numeric value to float if possible."""
    if zv is None:
        return None
    if isinstance(zv, (int, float)):
        return float(zv)
    try:
        if hasattr(zv, "numerator_as_long") and hasattr(zv, "denominator_as_long"):
            return zv.numerator_as_long() / zv.denominator_as_long()
        s = zv.as_decimal(30)
        if s.endswith('?'):
            s = s[:-1]
        try:
            return float(Fraction(s))
        except Exception:
            return float(s)
    except Exception:
        try:
            return float(Fraction(str(zv)))
        except Exception:
            return None

def plot_f_vs_cg(f_expr, g_expr, var, c, n0, width=5, num_points=400, figsize=(8,5)):
    """
    Plot f(n) vs c*g(n), centered at n0, with a marker showing if c*g(n0) >= f(n0).
    Works whether c and n0 are Python numbers or Z3 model values.
    """
    # Ensure c and n0 are floats
    c_val = _z3_value_to_float(c)
    n0_val = _z3_value_to_float(n0)
    if c_val is None or n0_val is None:
        raise ValueError("Could not convert c or n0 to numeric values.")

    xmin, xmax = n0_val - width, n0_val + width
    xs = np.linspace(xmin, xmax, num_points)

    ys_f, ys_cg, xs_valid = [], [], []

    sort = var.sort()
    is_real = (sort.name() == 'Real')
    is_int = (sort.name() == 'Int')

    for xv in xs:
        try:
            z_x = IntVal(int(round(xv))) if is_int else RealVal(str(float(xv)))
            rf = simplify(substitute(f_expr, (var, z_x)))
            rg = simplify(substitute(g_expr, (var, z_x)))
            f_val = _z3_value_to_float(rf)
            g_val = _z3_value_to_float(rg)
            cg_val = None if g_val is None else c_val * g_val

            if (f_val is not None and cg_val is not None and
                not (math.isinf(f_val) or math.isnan(f_val)) and
                not (math.isinf(cg_val) or math.isnan(cg_val))):
                xs_valid.append(xv)
                ys_f.append(f_val)
                ys_cg.append(cg_val)
        except Exception:
            pass

    # Evaluate at n0
    z_n0 = IntVal(int(round(n0_val))) if is_int else RealVal(str(float(n0_val)))
    f_n0 = _z3_value_to_float(simplify(substitute(f_expr, (var, z_n0))))
    g_n0 = _z3_value_to_float(simplify(substitute(g_expr, (var, z_n0))))
    cg_n0 = None if g_n0 is None else c_val * g_n0

    holds = (cg_n0 is not None and f_n0 is not None and cg_n0 >= f_n0)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs_valid, ys_f, label=str(f_expr))
    ax.plot(xs_valid, ys_cg, label=f"{c_val}·("+str(g_expr)+")")
    ax.axvline(n0_val, color='red', linestyle='--', linewidth=1, label=f"n = n0 = {n0_val}")

    ax.set_xlabel("n")
    ax.set_xlim(xmin, xmax)
    ax.set_title( "c = " + str(c_val) + ", n0 = " + str(n0_val) )
    ax.grid(True)
    ax.legend()
    plt.show()

def bigOPlot( f, g, makeBigOSolver ):
  s = makeBigOSolver( f, g )
  if ( s.check() == sat ):
    sol = s.model()
    c = Real('c')
    n0 = Real('n_0')
    n = Real('n')
    plot_f_vs_cg(f, g, n, sol[c], sol[n0], width = 1)
  else:
    print( "f(n) =/= O(g(n))" )


def bigOmegaPlot( f, g, makeBigOmegaSolver ):
  s = makeBigOmegaSolver( f, g )
  if ( s.check() == sat ):
    sol = s.model()
    c = Real('c')
    n0 = Real('n_0')
    n = Real('n')
    plot_f_vs_cg(f, g, n, sol[c], sol[n0], width = 1)
  else:
    print( "f(n) =/= Omega(g(n))" )


