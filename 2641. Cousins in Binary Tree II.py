# Definition for a binary tree node.
from typing import Optional
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def replaceValueInTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        # cousins of a node are at the same level, so bfs
        # root does not have cousins, so sum of cousins of root = 0
        # track sum of all childs at level i, and track all parents at level i-1

        queue, root.val = deque([root]), 0
        while queue:
            length, parents, child_sum = len(queue), [], 0
            for _ in range(length):
                current = queue.popleft()
                parents.append(current)
                if current.left:
                    queue.append(current.left)
                    child_sum += current.left.val
                if current.right:
                    queue.append(current.right)
                    child_sum += current.right.val

            for parent in parents:
                cousin_sum = child_sum
                if parent.left:
                    cousin_sum -= parent.left.val
                if parent.right:
                    cousin_sum -= parent.right.val
                if parent.left:
                    parent.left.val = cousin_sum
                if parent.right:
                    parent.right.val = cousin_sum
        return root
