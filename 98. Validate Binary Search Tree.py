# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:

        def dfs(node, smallest, largest):
            if not node:
                return True
            elif not (smallest <= node.val <= largest):
                return False
            return dfs(node.left, smallest, node.val - 1) and dfs(node.right, node.val + 1, largest)

        # return dfs(root, -math.inf, math.inf)

        lastTraversed = float('-inf')

        def inOrdertraversal(root: TreeNode):
            nonlocal lastTraversed
            if not root:
                return True

            l = inOrdertraversal(root.left)

            if not l or lastTraversed >= root.val:
                return False

            lastTraversed = root.val

            r = inOrdertraversal(root.right)
            return r

        # return inOrdertraversal(root)

        def inOrdertraversalStack(root):
            if not root:
                return True
            lastTraversed, stack = float(-inf), []
            while root or stack:
                while root:
                    stack.append(root)
                    root = root.left
                node = stack.pop()
                if lastTraversed >= node.val:
                    return False
                lastTraversed = node.val
                root = node.right
            return True

        return inOrdertraversalStack(root)


"""
Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

The left  subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
"""
