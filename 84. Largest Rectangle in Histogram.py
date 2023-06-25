from typing import List


class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        def largestRectangle_3_traverse():
            left_limit, right_limit = [0] * len(heights), [0] * len(heights)

            stack = []  # strictly increasing stack
            for i in range(len(heights)):
                while stack and heights[stack[-1]] >= heights[i]:
                    stack.pop()
                left_limit[i] = (stack[-1] + 1) if stack else 0
                stack.append(i)

            stack = []  # strictly increasing stack
            for i in range(len(heights) - 1, -1, -1):
                while stack and heights[stack[-1]] >= heights[i]:
                    stack.pop()
                right_limit[i] = (stack[-1] - 1) if stack else (len(heights) - 1)
                stack.append(i)

            return max(heights[i] * (right_limit[i] - left_limit[i] + 1) for i in range(len(heights)))

        return largestRectangle_3_traverse()

        def largestRectangle_1_traverse():
            stack = []
            area = i = 0
            while i < len(heights):  # non-decreasing stack
                if not stack or heights[i] >= heights[stack[-1]]:
                    stack.append(i)
                    i += 1
                else:
                    p = stack.pop()
                    width = i if not stack else (i - stack[-1] - 1)
                    area = max(area, heights[p] * width)

            while stack:
                p = stack.pop()
                width = i if not stack else (i - stack[-1] - 1)
                area = max(area, heights[p] * width)

            return area

        return largestRectangle_1_traverse()

"""
Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.



Example 1:


Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: The above is a histogram where width of each bar is 1.
The largest rectangle is shown in the red area, which has an area = 10 units.
Example 2:


Input: heights = [2,4]
Output: 4


Constraints:

1 <= heights.length <= 105
0 <= heights[i] <= 104
"""