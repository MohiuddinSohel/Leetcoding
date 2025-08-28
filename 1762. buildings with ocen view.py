from typing import List


class Solution:
    def findBuildings(self, heights: List[int]) -> List[int]:
        max_right, result = heights[-1], [len(heights) - 1]
        for i in range(len(heights) - 2, -1, -1):
            if heights[i] > max_right:
                result.append(i)
                max_right = heights[i]
        return result[::-1]