from xii.meshing.make_mesh_cpp import make_mesh
import numpy as np


class Node:
    def __init__(self, x, y, children, index=0):
        self.x = x
        self.y = y
        self.children = children
        self.index = index

        
def visit_nodes(tree):
    if isinstance(tree, Node):
        yield tree
    for child in tree.children:
        yield from visit_nodes(child)

    
def visit_leaves(tree):
    yield from filter(lambda n: not n.children, visit_nodes(tree))


def visit_edges(tree):
    for node in visit_nodes(tree):
        for child in node.children:
            yield (node.index, child.index)
    
    
def binary_tree():
    # We start from o-o
    tree = Node(0.5, 0, [Node(0.5, 0.5, [], 1)], 0)

    node_index, level = 1, 1
    while True:
        level += 1
        leaves = list(visit_leaves(tree))
        for k, node in enumerate(leaves):
            # Make a new new child
            x, y = node.x, node.y
            xl, xr = x-1/2**level, x+1/2**level
            yl, yr = y+1/2**level, y+1/2**level

            node_index += 1
            node.children.append(Node(xl, yl, [], node_index))

            node_index += 1
            node.children.append(Node(xr, yr, [], node_index))
        yield tree


def tree_mesh(tree):
    nodes = [(node.x, node.y) for node in sorted(visit_nodes(tree), key=lambda n: n.index)]
    nodes = np.array(nodes)

    edges = list(visit_edges(tree))
    edges = np.array(edges, dtype='uintp')

    return make_mesh(nodes.flatten(), edges.flatten(), tdim=1, gdim=2)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import File
    
    generator = binary_tree()

    for k, tree in zip(range(4), generator):
       print()
       for edge in visit_edges(tree):
           print(edge)
       print()
            
    mesh = tree_mesh(tree)
    File('foo.pvd') << mesh
