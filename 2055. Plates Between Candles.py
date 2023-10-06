from collections import defaultdict
from typing import List


class Solution:
    def platesBetweenCandles(self, s: str, queries: List[List[int]]) -> List[int]:
        # left_nearest and right_nearest candle for each index is better one
        def search_greater_equal(target):
            l, r = 0, len(candles) - 1
            while l <= r:
                m = l + (r - l) // 2
                if candles[m] < target:
                    l = m + 1
                else:
                    r = m - 1
            return l

        def search_less_equal(target):
            l, r = 0, len(candles) - 1
            while l <= r:
                m = l + (r - l) // 2
                if candles[m] > target:
                    r = m - 1
                else:
                    l = m + 1
            return r

        candles, seen_candle, c_sum = [], False, 0
        prefix_sum = defaultdict(int)
        for i, c in enumerate(s):
            if c == '|':
                seen_candle = True
                prefix_sum[i] = c_sum
                candles.append(i)
            elif seen_candle:
                c_sum += 1

        result = [0] * len(queries)
        if not candles:
            return result
        for i, query in enumerate(queries):
            l, r = search_greater_equal(query[0]), search_less_equal(query[1])
            if l == len(candles) or r < 0 or candles[l] >= query[1] or candles[r] <= query[0]:
                continue
            result[i] = prefix_sum[candles[r]] - prefix_sum[candles[l]]
        return result




if __name__ == '__main__':
    s = "***|**|*****|**||**|*"
    queries = [[1,17],[4,5],[14,17],[5,11],[15,16]]
    print(Solution().platesBetweenCandles(s, queries))