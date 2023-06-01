from typing import List


class Solution:
    def stoneGameII(self, piles: List[int]) -> int:
        def dfsWithMemo(index, M):
            if index >= len(piles):
                return 0
            elif cache[index][M] == intMin:
                best = intMin
                for i in range(1, 2 * M + 1):
                    if index + i > len(piles):
                        break
                    gain = cSum[index + i] - cSum[index]
                    best = max(best, gain - dfsWithMemo(index + i, max(M, i)))
                cache[index][M] = best

            return cache[index][M]

        def iterative():  # TLE, because of unecessary state calculation
            intMin = float('-inf')
            cSum = [0]
            for p in piles:
                cSum.append(cSum[-1] + p)
            dp = [[intMin for _ in range(len(piles) + 1)] for _ in range(len(piles) + 1)]
            for i in range(len(piles) + 1):
                dp[len(piles)][i] = 0

            for index in range(len(piles) - 1, -1, -1):
                for M in range(1, len(piles) + 1):
                    for j in range(1, 2 * M + 1):
                        if index + j > len(piles):
                            break
                        gain = cSum[index + j] - cSum[index]
                        dp[index][M] = max(dp[index][M], gain - dp[index + j][max(M, j)])
            return (dp[0][1] + cSum[-1]) // 2

        return iterative()

        intMin = float('-inf')
        cSum = [0]
        for p in piles:
            cSum.append(cSum[-1] + p)
        cache = [[intMin for _ in range(len(piles) + 1)] for _ in range(len(piles))]
        diff = dfsWithMemo(0, 1)
        return (cSum[-1] + diff) // 2  # alice + bob = sum, alice - bob = diff


if __name__ == '__main__':
    piles = [2,7,9,4,4]
    print(Solution().stoneGameII(piles))




