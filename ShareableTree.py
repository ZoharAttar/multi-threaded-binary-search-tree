import sys
import re
import networkx as nx
import matplotlib.pyplot as plt
from threading import Lock
from datetime import datetime


class Node:
    """
    Class Node
    """

    def __init__(self, value):
        self.left = None
        self.data = value
        self.right = None
        self.counter = 0


class ShareableTree:

    def __init__(self, file_path=None):

        # Init the tree by loading the string from a text file
        # e.g., try https://www.gutenberg.org/cache/epub/834/pg834.txt
        # read the file string by string and add it to the tree. a string contains only [a-z][A-Z].
        # e.g.,
        #  I am afraid, Watson, that I shall have to go,” said Holmes,
        #  The words: I am afraid Watson that I shall have to go said Holmes
        self.read_counter = 0
        self.mutex = Lock()
        self.rw_mutex = Lock()
        self.root = None
        self.data = None
        with open(file_path, "r", encoding='utf8') as file:
            rows = file.readlines()
        words = []
        for row in rows:
            words.extend(re.findall(r'\b\w+\b', row))
        for word in words:
            self.add_str(word)

    def add_str(self, str_to_add):
        # add the str to the tree.
        # update the counter accordingly is the string exists (reduce the counter by 1). For the first string counter=1
        # return True
        self.rw_mutex.acquire()
        if self.root is None:
            self.root = Node(str_to_add)
            self.root.counter += 1
        elif self.find_node(self.root, str_to_add) is None:
            node = self.insert_node(self.root, str_to_add)
            node = self.find_node(self.root, str_to_add)
            node.counter += 1
        else:
            node = self.find_node(self.root, str_to_add)
            node.counter += 1
        self.rw_mutex.release()
        return True

    def del_str(self, str_to_del):
        # delete str from tree
        # update the counter accordingly if the string exists (reduce the counter by 1). If counter=0, remove the node.
        # return True if the str found
        # return False if str was not found
        self.rw_mutex.acquire()

        node = self.find_node(self.root, str_to_del)
        if node is None:
            self.rw_mutex.release()
            return False
        elif node.counter > 1:
            node.counter -= 1
            self.rw_mutex.release()
            return True
        else:
            self.delete(self.root, str_to_del)

        self.rw_mutex.release()
        return True

    def search_str(self, str_to_search):
        # return the str if exists
        # return None if not
        self.mutex.acquire()
        self.read_counter += 1
        if self.read_counter == 1:
            self.rw_mutex.acquire()
        self.mutex.release()

        found = self.find_node(self.root, str_to_search)
        if found is None:
            self.mutex.acquire()
            self.read_counter -= 1
            if self.read_counter == 0:
                self.rw_mutex.release()
            self.mutex.release()
            return found
        self.mutex.acquire()
        self.read_counter -= 1
        if self.read_counter == 0:
            self.rw_mutex.release()
        self.mutex.release()
        return found.data

    def find_node(self, root, str_to_find):
        # Base Cases: root is null or key is present at root
        if root is None or root.data == str_to_find:
            return root

        # Key is greater than root's key
        if root.data < str_to_find:
            return self.find_node(root.right, str_to_find)

        # Key is smaller than root's key
        return self.find_node(root.left, str_to_find)

    def buildtree(self, nodes, start, end):

        # base case
        if start > end:
            return None

        # Get the middle element and make it root
        mid = (start + end) // 2
        node = nodes[mid]

        # Using index in Inorder traversal, construct
        # left and right subtrees
        node.left = self.buildtree(nodes, start, mid - 1)
        node.right = self.buildtree(nodes, mid + 1, end)
        return node

    def inorder(self, node, nodes_list):

        # Base case
        if not node:
            return

        # Store nodes in Inorder (which is sorted
        # order for BST)
        self.inorder(node.left, nodes_list)
        nodes_list.append(node)
        self.inorder(node.right, nodes_list)

    def balance_tree(self):
        # Balance the tree (https://www.programiz.com/dsa/balced-binary-tree)an
        # After balancing, the maximum height difference will be 1
        # return None
        self.rw_mutex.acquire()

        nodes = []

        def inorder(node):
            if not node:
                return

            inorder(node.left, nodes)
            nodes.append(node)
            inorder(node.right, nodes)

        self.inorder(self.root, nodes)
        self.root = self.buildtree(nodes, 0, len(nodes) - 1)
        self.rw_mutex.release()
        return None

    def get_height(self):
        # return tree height (empty tree: return -1,  only root node: return 0, etc)
        self.mutex.acquire()
        self.read_counter += 1
        if self.read_counter == 1:
            self.rw_mutex.acquire()
        self.mutex.release()

        def height(node):
            if node is None:
                return -1
            elif node.left is None and node.right is None:
                return 0
            else:
                # Compute the depth of each subtree
                lDepth = height(node.left)
                rDepth = height(node.right)
                # Use the larger one
                if (lDepth > rDepth):
                    return lDepth + 1
                else:
                    return rDepth + 1

        h = height(self.root)
        self.mutex.acquire()
        self.read_counter -= 1
        if self.read_counter == 0:
            self.rw_mutex.release()
        self.mutex.release()
        return h

    def print_tree(self):
        # print to console the tree from smallest to largest value, delimited by ',': e.g:
        # >abc,bcd,def,HOLMES,SHERLOCK
        # return None
        self.mutex.acquire()
        self.read_counter += 1
        if self.read_counter == 1:
            self.rw_mutex.acquire()
        self.mutex.release()

        def inorder(root):
            if root:
                inorder(root.left)
                print(root.data, end=','),
                inorder(root.right)

        inorder(self.root)
        self.mutex.acquire()
        self.read_counter -= 1
        if self.read_counter == 0:
            self.rw_mutex.release()
        self.mutex.release()
        return None

    def show(self):
        # draw the tree using GUI
        # return None
        self.mutex.acquire()
        self.read_counter += 1
        if self.read_counter == 1:
            self.rw_mutex.acquire()
        self.mutex.release()

        G = nx.DiGraph()

        def add_edges_nodes(node):
            if node is None:
                return
            G.add_node(node.data, count=node.counter)
            if node.left:
                G.add_edge(*(node.data, node.left.data))
                add_edges_nodes(node.left)
            if node.right:
                G.add_edge(*(node.data, node.right.data))
                add_edges_nodes(node.right)

        def create_labels(node):
            if node is None:
                return {}
            labels = {node.data: f"{node.data, node.counter}\n\n"}
            labels.update(create_labels(node.left))
            labels.update(create_labels(node.right))
            return labels

        add_edges_nodes(self.root)

        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color='red')
        nx.draw_networkx_edges(G, pos, arrowsize=10, width=2, edge_color='blue')
        labels = create_labels(self.root)
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_family='DejaVu Sans')
        plt.axis('off')
        t = datetime.now()
        date_t = t.strftime("%m-%d-%y_%H-%M-%S")
        plt.savefig(f"tree_{date_t}.jpg")
        plt.show()
        self.mutex.acquire()
        self.read_counter -= 1
        if self.read_counter == 0:
            self.rw_mutex.release()
        self.mutex.release()
        return None

    def insert_node(self, root, key):
        if root is None:
            return Node(key)
        else:
            if root.data == key:
                return root
            elif root.data < key:
                root.right = self.insert_node(root.right, key)
            else:
                root.left = self.insert_node(root.left, key)
        return root

    def minValueNode(node):
        current = node

        # loop down to find the leftmost leaf
        while current.left is not None:
            current = current.left

        return current

    # Given a binary search tree and a key, this function
    # delete the key and returns the new root

    def delete(self, root, key):

        # Base Case
        if root is None:
            return root

        # If the key to be deleted
        # is smaller than the root's
        # key then it lies in  left subtree
        if key < root.data:
            root.left = self.delete(root.left, key)

        # If the kye to be delete
        # is greater than the root's key
        # then it lies in right subtree
        elif (key > root.data):
            root.right = self.delete(root.right, key)

        # If key is same as root's key, then this is the node
        # to be deleted
        else:

            # Node with only one child or no child
            if root.left is None:
                temp = root.right
                root = None
                return temp

            elif root.right is None:
                temp = root.left
                root = None
                return temp

                # Node with two children:
                # Get the inorder successor
                # (smallest in the right subtree)
            temp = self.minValueNode(root.right)

            # Copy the inorder successor's
            # content to this node
            root.data = temp.data

            # Delete the inorder successor
            root.right = self.delete(root.right, temp.data)

        return root


if __name__ == '__main__':
    st = ShareableTree(file_path=r"test2.txt")
    st.show()
    st.balance_tree()
    st.show()

