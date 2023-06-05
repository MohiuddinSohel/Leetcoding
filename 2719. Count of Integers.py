class Solution:
    @staticmethod
    def count(num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        def count_integer_within_ab(index, lower_tight, upper_tight, current_sum):
            if current_sum > max_sum:
                return 0

            elif index == len(num2):
                return min_sum <= current_sum <= max_sum

            elif dp[index][lower_tight][upper_tight][current_sum] == -1:
                l, r, count = int(num1[index]) if lower_tight else 0, int(num2[index]) if upper_tight else 9, 0
                for current_digit in range(l, r + 1):
                    if current_sum + current_digit > max_sum:
                        break
                    new_lower_tight = lower_tight and (current_digit == int(num1[index]))
                    new_upper_tight = upper_tight and (current_digit == int(num2[index]))
                    count += count_integer_within_ab(index + 1, new_lower_tight, new_upper_tight,
                                                     current_sum + current_digit)
                    count %= mod

                dp[index][lower_tight][upper_tight][current_sum] = count

            return dp[index][lower_tight][upper_tight][current_sum]

        # make num1 as same length of num2 by adding 0 infront
        mod, num1 = 10 ** 9 + 7, ("0" * (len(num2) - len(num1))) + num1
        dp = [[[[-1 for _ in range(max_sum + 1)] for _ in range(2)] for _ in range(2)] for _ in range(len(num2))]
        return count_integer_within_ab(0, True, True, 0)


if __name__ == '__main__':
    num1, num2, min_num, max_num = "1", "12", 1, 8
    print(Solution.count(num1, num2, min_num, max_num))

