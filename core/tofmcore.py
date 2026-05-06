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

# This function was written with the help of ChatGPT
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

# This function was written with the help of ChatGPT
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

# This function was written using ChatGPT
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

# This function was written with the help of ChatGPT
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
  # to view the solution we have defined a speical function
  for i in range(1,N+1):
    for j in range(1,N+1):
      xij = Bool('x_{'+str(i)+str(j)+'}')
      if ( sol[xij] ):
        rook_array[i-1][j-1] = 1
  return rook_array

# This function was written using ChatGPT
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

# This function was written using ChatGPT
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

# This function was written using ChatGPT
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

# This function was written using ChatGPT
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

# This function was written using ChatGPT
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

# This function was written with the help of ChatGPT
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


import matplotlib.pyplot as plt
# written using gemini
def visualize_truss_solution( f_AB, f_AC, f_BC ):

  # 2. Define geometry
  joints = {'A': (0, 0), 'B': (3, 0), 'C': (0, 4)}
  members = [('A', 'B', f_AB), ('A', 'C', f_AC), ('B', 'C', f_BC)]

  # 3. Setup Plot
  fig, ax = plt.subplots(figsize=(6, 6))
  ax.set_aspect('equal')
  ax.axis('off')

  # 4. Plot Members
  max_force = max(abs(f_AB), abs(f_AC), abs(f_BC))
  for node1, node2, force in members:
    x_values = [joints[node1][0], joints[node2][0]]
    y_values = [joints[node1][1], joints[node2][1]]

    # Color: Blue for Tension (+), Red for Compression (-)
    color = 'blue' if force > 0 else 'red'

    # Thickness proportional to magnitude
    thickness = 1 + 5 * (abs(force) / max_force)

    ax.plot(x_values, y_values, color=color, linewidth=thickness, zorder=1)

    # Add force label in the middle of the member
    mid_x = sum(x_values) / 2
    mid_y = sum(y_values) / 2
    ax.text(mid_x, mid_y, f"{abs(force):.1f} N",
            ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

  # 5. Plot Joints
  for name, (x, y) in joints.items():
    ax.plot(x, y, 'ko', markersize=10, zorder=2) # Black dots for joints
    ax.text(x - 0.2, y + 0.2, name, fontsize=12, fontweight='bold')

  # 6. Add Custom Legend
  ax.plot([], [], color='blue', linewidth=3, label='Tension')
  ax.plot([], [], color='red', linewidth=3, label='Compression')
  ax.legend(loc='upper right')

  plt.title("Truss Internal Forces Visualization", fontsize=14)
  plt.show()



# written using gemini
def plot_gears(n1, n2, n3, n4):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Centers (Stage 1)
    c1 = (0, 0)
    r1 = n1 / 2
    c2 = (r1 + n2/2, 0)
    r2 = n2 / 2
    
    # Centers (Stage 2 - Shaft 2 is shared)
    c3 = c2
    r3 = n3 / 2
    c4 = (c2[0] + r3 + n4/2, 0)
    r4 = n4 / 2
    
    # Draw Pitch Circles
    gears = [(c1, r1, 'Input (N1)'), (c2, r2, 'N2'), (c3, r3, 'N3'), (c4, r4, 'Output (N4)')]
    colors = ['#3498db', '#e74c3c', '#f1c40f', '#2ecc71']
    
    for i, (pos, rad, label) in enumerate(gears):
        circle = plt.Circle(pos, rad, color=colors[i], alpha=0.6, label=f"{label}: {int(rad*2)}T")
        ax.add_artist(circle)
        
    ax.set_xlim(-r1 - 5, c4[0] + r4 + 5)
    ax.set_ylim(-max(r2, r4) - 5, max(r2, r4) + 5)
    ax.set_aspect('equal')
    plt.title("Powertrain Layout: Compound Spur Gear Train")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


### Lewis Structures ###
def draw_lewis_from_model(m):
    """Converts a model to the format required by draw_lewis_structure"""
    # List of elements
    elements = []
    # List of bond variables
    bonds = []
    lone_pairs = []

    # Add element names and lone pairs to the list
    for d in m.decls():
        name = d.name()
        if len(name) == 1 or name[1] == '^':
            elements.append(name)
            lone_pairs.append(m[d].as_long())

    # Add bond pair counts
    for d in m.decls():
        name = d.name()
        # Bond Pairs
        if len(name) > 1 and name[1] != '^':
            bonds.append((elements.index(name[0]), elements.index(name[1:4]), m[d].as_long()))

    draw_lewis_structure(elements, bonds, lone_pairs)

# Written by Gemini
def draw_lewis_structure(elements, bonds, lone_pairs):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_aspect('equal')
    ax.axis('off')

    # 1. Coordinate Setup (Circle Layout)
    n = len(elements)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {i: np.array([np.cos(a), np.sin(a)]) / 3 for i, a in enumerate(angles)}

    # 2. Draw Bonds with Multi-bond Offsets
    for idx1, idx2, count in bonds:
        p1, p2 = pos[idx1], pos[idx2]
        vec = p2 - p1
        perp = np.array([-vec[1], vec[0]])
        perp = perp / np.linalg.norm(perp) * 0.05

        # Offset multi-bonds slightly so they don't overlap
        offsets = np.linspace(-0.6, 0.6, count) if count > 1 else [0]
        for opt in offsets:
            shift = perp * opt
            ax.plot([p1[0] + shift[0], p2[0] + shift[0]],
                    [p1[1] + shift[1], p2[1] + shift[1]],
                    color='black', lw=2, zorder=1)

    # 3. Draw Atoms and Distributed Lone Pairs
    for i, (el, lp_count) in enumerate(zip(elements, lone_pairs)):
        x, y = pos[i]
        ax.text(x, y, el[0], fontsize=28, fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', pad=1.5), zorder=2)

        # Calculate angles to neighbors to find "open" slots
        neighbor_angles = []
        for b1, b2, _ in bonds:
            if b1 == i: neighbor_angles.append(np.arctan2(pos[b2][1] - y, pos[b2][0] - x))
            if b2 == i: neighbor_angles.append(np.arctan2(pos[b1][1] - y, pos[b1][0] - x))

        # Standard slots: 0, 90, 180, 270 degrees
        potential_slots = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        available_slots = []

        for slot in potential_slots:
            # Only use slot if it's not pointing toward a bond
            if not any(abs((slot - na + np.pi) % (2 * np.pi) - np.pi) < 0.5 for na in neighbor_angles):
                available_slots.append(slot)

        # Draw lone pairs in the best available slots
        for lp_idx in range(min(lp_count, len(available_slots))):
            slot_angle = available_slots[lp_idx]
            dist = 0.1
            # The two dots of the pair are slightly separated perpendicular to the slot angle
            dot_gap = 0.03
            for side in [-1, 1]:
                dx = x + dist * np.cos(slot_angle) + side * dot_gap * np.sin(slot_angle)
                dy = y + dist * np.sin(slot_angle) - side * dot_gap * np.cos(slot_angle)
                ax.scatter(dx, dy, s=40, color='red', zorder=3)  # Red for visibility
### Install Problem, all basically written by ChatGPT

# These two are from the Z3 article


def DependsOn(pack, deps):
    if is_expr(
        deps
    ):  # Allows us to do something like DependsOn(a, b) instead of DependsOn(a, [b])
        return Implies(pack, deps)
    else:
        return And([Implies(pack, dep) for dep in deps])


def Conflict(p1, p2):
    return Or(Not(p1), Not(p2))


def display_pkg_struct(
    solver_or_exprs,
    *,
    name="PkgGraph",
    rankdir="TB",
    fontname="Liberation Serif",
    fontsize=28,
    nodesep="0.7",
    ranksep="0.8",
):
    """
    Visual encoding:
      - blue box = explicitly requested package
      - black box = normal package
      - black arrows = dependencies
      - black dot = OR dependency
      - red double arrow = conflict

    Written by ChatGPT
    """

    # Normalize input
    if isinstance(solver_or_exprs, Solver):
        assertions = list(solver_or_exprs.assertions())
    else:
        assertions = list(solver_or_exprs)

    dot = Digraph(name)
    dot.attr(
        rankdir=rankdir, splines="true", nodesep=str(nodesep), ranksep=str(ranksep)
    )

    dot.attr("edge", penwidth="2", arrowsize="0.9")

    # ----------------------------
    # Helpers
    # ----------------------------

    def is_atom_bool(e):
        return is_app(e) and e.num_args() == 0

    def kind(e):
        return e.decl().kind() if is_app(e) else None

    def flatten(e, op_kind):
        if is_app(e) and kind(e) == op_kind:
            out = []
            for ch in e.children():
                out.extend(flatten(ch, op_kind))
            return out
        return [e]

    def get_not_atom(e):
        if is_app(e) and kind(e) == Z3_OP_NOT:
            x = e.arg(0)
            if is_atom_bool(x):
                return x
        return None

    # ----------------------------
    # Collect graph data
    # ----------------------------

    pkgs = set()
    requested = set()
    dep_edges = set()
    or_groups = []
    conflict_pairs = set()

    def ensure_pkg(atom):
        pkgs.add(str(atom))

    def consume_dep(pack, rhs):
        ensure_pkg(pack)

        # AND deps
        if is_app(rhs) and kind(rhs) == Z3_OP_AND:
            for term in flatten(rhs, Z3_OP_AND):
                consume_dep(pack, term)
            return

        # OR deps
        if is_app(rhs) and kind(rhs) == Z3_OP_OR:
            opts = []
            for term in flatten(rhs, Z3_OP_OR):
                if is_atom_bool(term):
                    ensure_pkg(term)
                    opts.append(str(term))
            if opts:
                or_groups.append((str(pack), opts))
            return

        # single dep
        if is_atom_bool(rhs):
            ensure_pkg(rhs)
            dep_edges.add((str(pack), str(rhs)))

    def consume_expr(e):

        # ⭐ REQUESTED PACKAGE DETECTION
        if is_atom_bool(e):
            ensure_pkg(e)
            requested.add(str(e))
            return

        if is_app(e) and kind(e) == Z3_OP_AND:
            for term in flatten(e, Z3_OP_AND):
                consume_expr(term)
            return

        if is_app(e) and kind(e) == Z3_OP_IMPLIES:
            lhs, rhs = e.arg(0), e.arg(1)
            if is_atom_bool(lhs):
                consume_dep(lhs, rhs)
            return

        # conflicts
        if is_app(e) and kind(e) == Z3_OP_OR:
            terms = flatten(e, Z3_OP_OR)
            negs = [get_not_atom(t) for t in terms]
            if all(n is not None for n in negs) and len(negs) >= 2:
                names = []
                for n in negs:
                    ensure_pkg(n)
                    names.append(str(n))

                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        conflict_pairs.add(frozenset((names[i], names[j])))

    for a in assertions:
        consume_expr(a)

    # ----------------------------
    # Emit nodes
    # ----------------------------

    dot.attr(
        "node",
        shape="box",
        fontsize=str(fontsize),
        fontname=fontname,
        penwidth="2",
        style="filled",
    )

    for p in sorted(pkgs):
        if p in requested:
            dot.node(p, p, fillcolor="#4DA3FF")  # nice blue
        else:
            dot.node(p, p, fillcolor="white")

    # dependencies
    for u, v in dep_edges:
        dot.edge(u, v)

    # OR dots
    dot.attr("node", shape="point", width="0.12", height="0.12")
    for i, (pack, opts) in enumerate(or_groups, 1):
        j = f"j{i}"
        dot.node(j, "")
        dot.edge(pack, j)
        for o in opts:
            dot.edge(j, o)

    # conflicts
    for pair in conflict_pairs:
        a, b = sorted(list(pair))
        dot.edge(a, b, dir="both", color="red", style="dashed", penwidth="2.5")

    return dot


def display_pkg_solution(
    solver_or_exprs,
    *,
    model=None,
    name="PkgGraph",
    rankdir="TB",
):
    """
    Coloring priority:

        requested  -> BLUE
        true       -> GREEN
        false      -> white
        undefined  -> gray

    Written by ChatGPT
    """

    if isinstance(solver_or_exprs, Solver):
        if solver_or_exprs.check() == unsat:
            print("No satisfying installation profile found!")
            return
        model = solver_or_exprs.model()
        assertions = list(solver_or_exprs.assertions())
    else:
        assertions = list(solver_or_exprs)

    dot = Digraph(name)
    dot.attr(rankdir=rankdir, splines="true", nodesep="0.7", ranksep="0.8")

    dot.attr("edge", penwidth="2", arrowsize="0.9")

    # -------------------------
    # Helpers
    # -------------------------

    def is_atom(e):
        return is_app(e) and e.num_args() == 0

    def kind(e):
        return e.decl().kind() if is_app(e) else None

    def flatten(e, op):
        if is_app(e) and kind(e) == op:
            out = []
            for c in e.children():
                out.extend(flatten(c, op))
            return out
        return [e]

    def get_not_atom(e):
        if is_app(e) and kind(e) == Z3_OP_NOT:
            x = e.arg(0)
            if is_atom(x):
                return x
        return None

    def model_val(atom):
        try:
            v = model.eval(atom, model_completion=False)
        except:
            return None

        if v is None:
            return None
        if v.eq(BoolVal(True)):
            return True
        if v.eq(BoolVal(False)):
            return False
        return None

    # -------------------------
    # Collect graph data
    # -------------------------

    pkgs = {}
    requested = set()
    deps = set()
    or_groups = []
    conflicts = set()

    def ensure(atom):
        pkgs[str(atom)] = atom

    def consume_dep(pack, rhs):
        ensure(pack)

        if is_app(rhs) and kind(rhs) == Z3_OP_AND:
            for t in flatten(rhs, Z3_OP_AND):
                consume_dep(pack, t)
            return

        if is_app(rhs) and kind(rhs) == Z3_OP_OR:
            opts = []
            for t in flatten(rhs, Z3_OP_OR):
                if is_atom(t):
                    ensure(t)
                    opts.append(str(t))
            if opts:
                or_groups.append((str(pack), opts))
            return

        if is_atom(rhs):
            ensure(rhs)
            deps.add((str(pack), str(rhs)))

    def consume_expr(e):

        # requested package
        if is_atom(e):
            ensure(e)
            requested.add(str(e))
            return

        if is_app(e) and kind(e) == Z3_OP_AND:
            for t in flatten(e, Z3_OP_AND):
                consume_expr(t)
            return

        if is_app(e) and kind(e) == Z3_OP_IMPLIES:
            lhs, rhs = e.arg(0), e.arg(1)
            if is_atom(lhs):
                consume_dep(lhs, rhs)
            return

        if is_app(e) and kind(e) == Z3_OP_OR:
            terms = flatten(e, Z3_OP_OR)
            negs = [get_not_atom(t) for t in terms]

            if all(n is not None for n in negs) and len(negs) >= 2:
                names = []
                for n in negs:
                    ensure(n)
                    names.append(str(n))

                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        conflicts.add(frozenset((names[i], names[j])))

    for a in assertions:
        consume_expr(a)

    # -------------------------
    # Emit nodes with PRIORITY
    # -------------------------

    dot.attr(
        "node",
        shape="box",
        style="filled",
        penwidth="2",
        fontname="Liberation Serif",
        fontsize="28",
    )

    for name, atom in pkgs.items():

        if name in requested:
            fill = "#4DA3FF"  # BLUE (highest priority)

        else:
            mv = model_val(atom)

            if mv is True:
                fill = "#7CFC90"  # GREEN
            elif mv is False:
                fill = "white"
            else:
                fill = "#E8E8E8"

        dot.node(name, name, fillcolor=fill)

    # dependencies
    for u, v in deps:
        dot.edge(u, v)

    # OR dots
    dot.attr("node", shape="point", width="0.12", height="0.12")
    for i, (pack, opts) in enumerate(or_groups, 1):
        j = f"j{i}"
        dot.node(j, "")
        dot.edge(pack, j)
        for o in opts:
            dot.edge(j, o)

    # conflicts
    for pair in conflicts:
        a, b = sorted(pair)
        dot.edge(a, b, dir="both", color="red", style="dashed", penwidth="2.5")

    return dot


def display_opium_graph():
    a, b, c, d, e, f, g, y, z = Bools("a b c d e f g y z")

    s = Solver()
    s.add(
        DependsOn(a, [b, c, z]),
        DependsOn(y, z),
        DependsOn(b, d),
        DependsOn(c, [Or(d, e), Or(f, g)]),
        Conflict(d, e),
        a,
        z,
    )

    return display_pkg_struct(s)


def display_int_graph():
    a, b, c, d, e, f, g = Bools("a b c d e f g")

    s = Solver()
    s.add(
        DependsOn(a, [b, c, d]),
        DependsOn(b, e),
        DependsOn(d, Or(f, g)),
        Conflict(e, g),
        a,
    )

    return display_pkg_solution(s)


def pkg_output_string(
    solver: Solver, *, algo: str = "sha256", digest_size: int = 16
) -> str:
    """
    Stable-ish fingerprint of a Z3 Solver's asserted formulas.
    - Order independent
    - Uses Z3's s-expression serialization
    - Returns hex digest

    algo: 'sha256' (default) or 'blake2b'
    digest_size: only used for blake2b (bytes)

    Written by ChatGPT
    """
    parts = [a.sexpr() for a in solver.assertions()]
    parts.sort()

    payload = "\n".join(parts).encode("utf-8")

    if algo == "sha256":
        return hashlib.sha256(payload).hexdigest()
    elif algo == "blake2b":
        return hashlib.blake2b(payload, digest_size=digest_size).hexdigest()
    else:
        raise ValueError("algo must be 'sha256' or 'blake2b'")


### Metamerism

# Generates objects and plots them too


# Simulates light with Gaussian distribution
def gaussian(wl, mu, sigma):
    return np.exp(-0.5 * ((wl - mu) / sigma) ** 2)


# Wavelengths
wl_min = 380
wl_max = 700
wl = np.arange(wl_min, wl_max + 1, 5)  # Wavelengths 5nm apart


def generate_illuminant(wl):
    E = 1.2 * gaussian(wl, 600, 120) + 0.8 * gaussian(  # warm tail
        wl, 450, 90
    )  # blue contribution
    E /= E.max()  # normalize => all values between 0 and 1

    return E


def plot_illuminant(wl, E):
    plt.plot(wl, E, c="orange")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative Power")
    plt.title("Simulated Illuminant SPD")
    plt.xticks(np.arange(min(wl), max(wl) + 1, 50))
    plt.grid(True)
    plt.show()


def generate_reflectant(wl):
    # Some reflectant, nothing in particular
    R = 0.3 + 0.4 * gaussian(wl, 520, 60) - 0.2 * gaussian(wl, 450, 30)
    R = np.clip(R, 0, 1)  # Make sure all values are between 0 and 1

    return R


def plot_reflectant(wl, R):
    plt.plot(wl, R, c="m")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative Power")
    plt.title("Object's Reflectance SPD")
    plt.xticks(np.arange(min(wl), max(wl) + 1, 50))
    plt.grid(True)
    plt.show()


def generate_and_plot_ERL(wl, E, R):
    L = E * R

    plt.plot(wl, E, c="orange", ls="--")
    plt.plot(wl, R, c="m", ls="--")
    plt.plot(wl, L, c="c")
    plt.legend(["Illuminant", "Reflectant", "Resultant"])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative Power")
    plt.title("Resultant SPD")
    plt.xticks(np.arange(min(wl), max(wl) + 1, 50))
    plt.grid(True)
    plt.show()

    return L


def generate_sensor(wl):
    # Sensor sensitivities, simplification of the human eye
    S = np.stack(
        [
            0.9 * gaussian(wl, 560, 40),  # L (red)   cone
            1.0 * gaussian(wl, 530, 35),  # M (green) cone
            0.7 * gaussian(wl, 420, 25),  # S (blue)  cone
        ],
        axis=1,
    )
    assert S.shape == (len(wl), 3)

    return S


def plot_sensor(wl, S):
    # Plotting SPD's for the three cones
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative Power")
    plt.title("Sensor Sensitivity SPD")
    plt.xticks(np.arange(wl_min, wl_max + 1, 50))
    plt.grid(True)

    colors = "rgb"
    for i, column in enumerate(S.T):
        plt.plot(wl, column, c=colors[i])
        plt.legend(["L (red)", "M (green)", "S (blue)"])

    plt.show()


### Chemistry Matching and Z-Index ###
def draw_all_matchings(s, all_sols, num_center):
  # Draw in a grid with 3 columns
  rows = (len(all_sols) // 3) + 1
  fig, axs = plt.subplots(rows,3, figsize=(12,12))
  axes = axs.flatten()

  for i, sol in enumerate(all_sols):
    draw_single_matching(sol, axes[i], num_center)

def draw_single_matching(m, ax, num_center):
  # Create the edges and the matchings from the solution
  edges = []
  matching = []
  for i in m:
    val = m[i]
    a = str(i)[3]
    b = str(i)[4]
    edges.append((a, b))
    if val:
      matching.append((a, b))

  # Create the graph from the edges.
  G = nx.Graph()
  G.add_edges_from(sorted(edges)) # Sorting ensures consisten layout of nodes
  edge_colors = ['red' if e in matching or (e[1], e[0]) in matching else 'black' for e in G.edges()]

  # Draw graph aligned to grid
  draw_chemical_graph(G, edge_colors, [str(i+1) for i in range(num_center)], ax)

# Written by Gemini
def draw_chemical_graph(G, edge_colors, centerline_nodes, ax):
    """
    centerline_nodes: a list of nodes to be placed on the y=0 axis.
    """
    pos = {}

    # 1. Position the centerline nodes
    for i, node in enumerate(centerline_nodes):
        pos[node] = (i, 0)

    # 2. Position the children above/below
    for parent in centerline_nodes:
        # Find neighbors not already in the centerline
        children = [n for n in G.neighbors(parent) if n not in centerline_nodes]

        for i, child in enumerate(children):
            # Alternate: even index above (1), odd index below (-1)
            x_offset = pos[parent][0]
            y_offset = 1 if i % 2 == 0 else -1
            pos[child] = (x_offset, y_offset)

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_size=700,
            node_color='white', edge_color=edge_colors, ax=ax)
def plot_two_reflectants(wl, R1, R2):
    # Plotting the objects' reflectance SPD
    plt.plot(wl, R1, c="m")
    plt.plot(wl, R2, c="brown")
    plt.legend(["Object 1 Reflectance", "Object 2 Reflectance"])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative Power")
    plt.title("Objects' Reflectance SPDs")
    plt.xticks(np.arange(min(wl), max(wl) + 1, 50))
    plt.grid(True)
    plt.show()


### Mate Matching

# Stolen + modified from other mating stuff

def split_male_female(s):
    match = re.match(r"^([A-Za-z]+\d+)([A-Za-z]+\d+)$", s)
    if match:
        return match.group(1), match.group(2)
    else:
        return s, ""  # if no split point found


def print_mate_matching_solution(solution):
    left_vertices = set({})
    right_vertices = set({})
    edges = []
    matching = []
    for i in solution:
        val = solution[i]
        a, b = split_male_female(str(i))
        left_vertices.add(a)
        right_vertices.add(b)
        edges.append((a, b))
        if val:
            matching.append((a, b))
    draw_bipartite_graph(left_vertices, right_vertices, edges, matching)
