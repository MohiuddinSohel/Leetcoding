import bisect
import math
import re
from collections import defaultdict, Counter, deque
import heapq
from typing import List

import sortedcontainers
from sortedcontainers import SortedList


def min_errors_exclamation(errorString, x, y):
    # https://leetcode.com/company/amazon/discuss/6032414/Amazon-or-OA-SDE-II-or-11082024
    # on eexclamation mark !
    mod = 10 ** 9 + 7
    n = len(errorString)

    # Precompute prefix counts of 0s and 1s
    prefix_0 = [0] * (n + 1)
    prefix_1 = [0] * (n + 1)

    for i in range(n):
        prefix_0[i + 1] = prefix_0[i] + (errorString[i] == '0')
        prefix_1[i + 1] = prefix_1[i] + (errorString[i] == '1')

    # Variables to store the number of subsequences and errors
    total_01 = 0  # subsequences of '01'
    total_10 = 0  # subsequences of '10'
    min_errors = float('inf')

    # Track cumulative counts of 0s and 1s from the end
    count_0_after = 0
    count_1_after = 0

    # Traverse the string from the end to handle '!'
    for i in range(n - 1, -1, -1):
        if errorString[i] == '!':
            # Calculate the cost for replacing '!' with '0'
            replace_with_0 = (count_1_after * x + prefix_1[i] * y) % mod
            # Calculate the cost for replacing '!' with '1'
            replace_with_1 = (count_0_after * y + prefix_0[i] * x) % mod

            # Update minimum errors
            min_errors = min(min_errors, replace_with_0, replace_with_1)
        elif errorString[i] == '0':
            count_0_after += 1
            total_01 += count_1_after
            total_01 %= mod
        elif errorString[i] == '1':
            count_1_after += 1
            total_10 += count_0_after
            total_10 %= mod

    # Add errors from existing subsequences
    total_errors = (total_01 * x + total_10 * y + min_errors) % mod

    return total_errors

def perfect_anagram(s):
    index_map, max_len = defaultdict(int), -1
    same_count = 0
    for i, c in enumerate(s):
        if i != 0 and c == s[i-1]:
            same += 1
        else:
            same = 0
        if c in index_map:
            if same != i - index_map[c]:
                max_len = max(max_len, i - index_map[c])
        else:
            index_map[c] = i
    return max_len

def box_removed_less_capacity(arr, cap):
    # https://leetcode.com/company/amazon/discuss/5875380/Amazon-OA-question-SDE1
    arr.sort()
    l, min_len = 0, len(arr)
    for r in range(len(arr)):
        while l < r and arr[r] > cap * arr[l]:
            l += 1
        min_len = min(min_len, len(arr) - (r - l + 1))
    return min_len
print("unloaded", box_removed_less_capacity( [1, 2,2,2, 3,3,3, 4, 4, 4, 5,5,5, 6, 6,6] , 2))

def symmetrical_number(s):
    letters = [0] * 26
    for c in s:
        letters[ord(c) - ord('a')] += 1
    result, l, odd_character = list(s), 0, None
    for index in range(26):
        if letters[index] == 0:
            continue
        if letters[index] > 1:
            result[l: l + letters[index] // 2] = [chr(ord('a') + index)] * (letters[index] // 2)
            r = len(s) - 1 - l
            result[r - letters[index] // 2 + 1 : r + 1] = [chr(ord('a') + index)] * (letters[index] // 2)
            l += letters[index] // 2
    return ('').join(result)


def getMaximumTasks(limit, primary, secondary):
    #https://leetcode.com/company/amazon/discuss/5898896/Amazon-SDE-1-OA
    counter = 0
    primary.sort()
    secondary.sort(reverse = True)
    while primary and secondary:
        if primary[-1] + secondary[-1] <= limit:
            secondary.pop()
            counter += 1
        primary.pop()

    return counter
# print(getMaximumTasks(7, [4, 5, 2, 4], [5, 6, 3, 4]))

def getMaxAlternatingMusic(music: str, k: int) -> int:
    #https://leetcode.com/company/amazon/discuss/5898896/Amazon-SDE-1-OA
    n = len(music)
    def longest_with_flips(target: str) -> int:
        left = 0
        flips = 0
        max_length = 0

        for right in range(n):
            if music[right] != target[right % 2]:
                flips += 1

            while flips > k:
                if music[left] != target[left % 2]:
                    flips -= 1
                left += 1

            max_length = max(max_length, right - left + 1)

        return max_length
    return max(longest_with_flips('01'), longest_with_flips('10'))

#maximum cumulative PNL
def pnl(arr):
    # https://leetcode.com/company/amazon/discuss/5891502/Amazon-OA
    # maximum negative PNL
    heap = []
    c_sum = arr[0]
    max_count = 0
    for i in range(1, len(arr)):
        if c_sum >= arr[i]:
            c_sum -= arr[i]
            heapq.heappush(heap, -arr[i])
            max_count += 1
        elif heap and -heap[0] > arr[i]:
            c_sum += (-heapq.heappop(heap) * 2) - arr[i]
            heapq.heappush(heap, -arr[i])
        else:
            c_sum += arr[i]
    return max_count

def paint_canvas(n, m, k, paint):
    # https://leetcode.com/company/amazon/discuss/5895561/Amazon-SDE-1-OA
    # home decor print canvas smart canvas
    matrix = [[0] * (m + 2) for _ in range(n + 2)]
    time = 0
    for r, c in paint:
        time += 1
        top_left_region = min(matrix[r-1][c], matrix[r-1][c-1], matrix[r][c-1])
        top_right_region = min(matrix[r - 1][c], matrix[r - 1][c + 1], matrix[r][c + 1])
        bottom_left_region = min(matrix[r][c-1], matrix[r + 1][c - 1], matrix[r + 1][c])
        bottom_right_region = min(matrix[r + 1][c], matrix[r + 1][c + 1], matrix[r][c + 1])
        matrix[r][c] = 1 + max(top_left_region, top_right_region, bottom_right_region, bottom_left_region)
        if matrix[r][c] == k:
            return time
    return -1

# add multiply without operator
def add(a, b):
    # Iterate till there is no carry
    while b != 0:
        # Calculate carry
        carry = a & b
        # Sum without carry
        a = a ^ b
        # Shift the carry to add it in the next position
        b = carry << 1
    return a

def multiply(a, b):
    result = 0
    # Ensure b is positive, if not adjust a and b.
    if b < 0:
        a, b = -a, -b

    # Iterate through all bits of b
    while b > 0:
        # If the least significant bit of b is 1, add the shifted a to the result
        if b & 1:
            result = add(result, a)

        # Shift a left to handle the next bit in b
        a <<= 1
        # Shift b right to process the next bit
        b >>= 1

    return result

def total_charge_after_merge(charge):
    # https://leetcode.com/company/amazon/discuss/5889626/Amazon-Unsolvable-OA-question-Is-it-solvable-by-Greedy
    # merge charge
    a = sum(c for i,c in enumerate(charge) if i  & 1 != 0 and c >= 0)
    b = sum(c for i,c in enumerate(charge) if i  & 1 == 0 and c >= 0)
    # print(a, b)
    return max(a, b)

# n points on 1D line denoting the Warehouses. We want to place 2 distribution centers on the number line
def two_ware_house(arr):
    # https://leetcode.com/company/amazon/discuss/5884386/Amazon-OA-Question
    arr.sort()
    d = 0
    l, r = 0, len(arr) - 1
    while l <= r:
        m = l + (r - l) // 2
        if arr[m] - arr[0] <= arr[-1] - arr[m]:
            l = m + 1
        else:
            r = m - 1

    left, right = 0, r
    while left < right:
        d += (arr[right] - arr[left])
        left += 1
        # right -= 1

    left, right = r + 1, len(arr) - 1
    while left < right:
        d += (arr[right] - arr[left])
        # left += 1
        right -= 1
    return d

def div(a, b):
    sign = False
    if a <0 and b < 0:
        a, b = -a, -b
    elif a < 0:
        a = -a
        sign = True
    elif b < 0:
        b = -b
        sign = True
    return a//b if not sign else -(a//b)


def number_of_ware_hosue(arr, d):
    # https://leetcode.com/discuss/interview-question/5911956/Amazon-OA-SDE-2-or-Oct-24/2676896
    # https://leetcode.com/company/amazon/discuss/5833810/Amazon-Senior-SDE-Coding-Assessment
    # delivery center centre
    def get_total_distance(target):
        return sum(2 * abs(c - target) for c in arr)

    l, r = -10**9, 10**9
    while l <= r:
        m = l + (r - l)//2

        dist_mid = get_total_distance(m)
        dist_mid_next = get_total_distance(m + 1)
        if dist_mid > dist_mid_next:
            if dist_mid <= d:
                r = m - 1
            else:
                l = m + 1
        else:
            r = m - 1
    leftmost = l

    l, r = -10**9, 10**9
    while l <= r:
        m = l + (r - l)//2

        dist_mid = get_total_distance(m)
        dist_mid_next = get_total_distance(m + 1)
        if dist_mid < dist_mid_next:
            if dist_mid <= d:
                l = m + 1
            else:
                r = m - 1
        else:
            l = m + 1
    rightmost = r
    # print(leftmost, rightmost)

    return (rightmost - leftmost + 1) if leftmost >= -10**9 and rightmost <= 10**9 else -1

#  server wait time request
def wait_time(wait):
    #https://leetcode.com/company/amazon/discuss/5738176/Amazon-Online-Assessment-Aug-20243
    #https://leetcode.com/company/amazon/discuss/5864551/Amazon-or-SDE2-or-OA
    # maximum waiting time
    counter, time, result, q_len = Counter(wait), 0, [], len(wait)
    for i, w in enumerate(wait):
        if w <= time:
            continue
        if counter[time]:
            q_len -= counter[time]
            counter[time] = 0
        result.append(q_len)
        counter[w] -= 1
        q_len -= 1
        time += 1
    result.append(q_len)
    return result

# maximum server channel quality
def findMaximumQuality(packets, channels):
    # https://leetcode.com/discuss/interview-question/5864443/Amazon-or-SDE2-or-OA
    # channel quality median
    n = len(packets)
    if n == channels:
        return int(sum(packets))

    packets.sort()
    answer = sum(packets[n - channels + 1:]) # send highest channels -1  the highest one, remaining packets all in the remaining channel
    remaining_packets = n - channels + 1

    if remaining_packets % 2 == 0:
        mid = (remaining_packets - 1) // 2
        answer += (packets[mid] + packets[mid + 1]) / 2.0

    else:
        answer += packets[remaining_packets // 2]
    return int(math.ceil(answer))

def max_zero_in_array(arr):
    # https://leetcode.com/company/amazon/discuss/5847276/Amazon-SDE-or-OA
    # maximum zero after decrementing prefix by 1 all element in the prtefix
    count, current_min = 1, arr[0]
    for i in range(1, len(arr)):
        if arr[i] <= current_min:
            count += 1
            current_min = arr[i]
    return count
# find first occurance of prefix and last occurance of suffix from regex to s
def regex_with_single_star(s, regex):
    # https://leetcode.com/discuss/interview-question/5478303/Amazon-OA/
    # this code should not work, need to use KMP to search prefix and suffix
    r = prefix_start = prefix_end = 0
    for i, c in enumerate(s):
        if regex[r] == '*':
            prefix_end = i - 1
            break
        elif c == regex[r]:
            if r == 0:
                prefix_start = i
            r += 1
        else:
            r = 0

    suffix_start = suffix_end = len(s) - 1
    r = len(regex) - 1
    for i in range(len(s) - 1, -1, -1):
        if regex[r] == '*':
            suffix_start = i + 1
            break
        elif s[i] == regex[r]:
            if r == len(regex) - 1:
                suffix_end = i
            r -= 1
        else:
            r = len(regex) - 1
    if prefix_end >= suffix_start:
        return 0
    else:
        return suffix_end - prefix_start + 1

def gas_station_query(query, trucks):
    # https://leetcode.com/company/amazon/discuss/5833810/Amazon-Senior-SDE-Coding-Assessment
    sum_truck = sum(trucks)
    result = []
    for p1, p2 in query:
        result.append(p1 * trucks[p1 - 1] + (p2 - p1) * trucks[p2 - 1] + (len(trucks) - p2) * trucks[-1] - sum_truck)
    return result
print("truck", gas_station_query([[2, 4]], [3, 6, 10, 15, 20]))

def min_generation(generation):
    #https://leetcode.com/company/amazon/discuss/5798402/Amazon-OA-questions
    # min generations to make all neurons equal
    total_diff, max_n = 0, max(generation)
    for g in generation:
        total_diff += (max_n - g)
    n = total_diff // 3
    count = 2 *n
    if total_diff - 3 * n == 1:
        count += 1
    elif total_diff - 3 * n == 2:
        count += 2
    return count

def min_operation_to_make_palindrome(s):
    # https://leetcode.com/company/amazon/discuss/5798402/Amazon-OA-questions
    # change every occurence of x to y
    def find(x):
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        p_x, p_y = find(x), find(y)
        if p_x == p_y:
            return False
        if rank[p_x] > rank[p_y]:
            p_x, p_y = p_y,p_x
        parent[p_x] = p_y
        if rank[p_x] == rank[p_y]:
            rank[p_y] += 1
        return True

    parent = {key:key for key in s}
    rank = {key:0 for key in s}
    l , r = 0, len(s) - 1
    count = 0
    while l < r:
        if s[l] != s[r]:
            if union(s[l], s[r]):
                count += 1
        l += 1
        r -= 1
    return count

def consecutive_racer(speed, k):
    #https://leetcode.com/company/amazon/discuss/5750448/Amazon-Question-2024
    # https://leetcode.com/playground/apLb52Qp
    tracker = defaultdict(int)
    left = right = maxLen = maxCount = 0

    for right, c in enumerate(speed):
        tracker[c] += 1
        maxCount = max(maxCount, tracker[c])
        if right - left + 1 > maxCount + k:
            tracker[speed[left]] -= 1
            left += 1
        maxLen = max(maxLen, right - left + 1)
    return maxCount if maxLen - k <= maxCount else maxLen - k
# print("Racer", consecutive_racer([1, 4, 4, 2, 2, 4], 2))



def reduandant_string(word, a, b):
    # https://leetcode.com/discuss/interview-question/5478303/Amazon-OA/
    dic = defaultdict(int)
    dic[0] = 1
    vowel_cnt, consonant_cnt = 0, 0
    res = 0
    for c in word:
        if c in 'aeiou':
           vowel_cnt += 1
        else:
           consonant_cnt += 1
        curr = (a - 1) * vowel_cnt + (b - 1) * consonant_cnt
        res += dic[curr]
        dic[curr] += 1
    return res

def getLargest_index_len_outliers(f1, f2):
    # https://leetcode.com/company/amazon/discuss/5738176/Amazon-Online-Assessment-Aug-2024
    max_len = 0
    dp = [1] * len(f1)
    for i in range(len(f1)):
        for j in range(i):
            if (f1[j] < f1[i] and f2[j] < f2[i]) or (f1[j] > f1[i] and f2[j] > f2[i]):
                dp[i] = max(dp[i], 1 + dp[j])
        max_len = max(max_len, dp[i])
    return max_len if max_len > 1 else -1

def minimized_max_after_distribution_k(arr, k):
    sum_arr = sum(arr)
    max_arr = max(arr)
    if max_arr * len(arr) >= sum_arr + k:
        return max_arr
    else:
        remaining = max_arr * len(arr) - (sum_arr + k)
        return max_arr + remaining//len(arr) + (1 if remaining % len(arr) else 0)

def exceeding_threshold_amazon_sales(arr, k, threshold):
    if len(arr) < k:
        return 0
    arr.sort(reverse = True)
    count = left = c_sum = 0
    for right, a in enumerate(arr):
        c_sum += a
        if right - left + 1 == k:
            if c_sum <= threshold:
                return count
            elif c_sum > threshold:
                count += 1
                c_sum -= arr[left]
                left += 1
    return count

def max_zero_by_decreasing_prefix(arr):
    # https://leetcode.com/company/amazon/discuss/5847276/Amazon-SDE-or-OA
    count, current_min = 1, arr[0]
    for i in range(1, len(arr)):
        if arr[i] <= current_min:
            current_min = arr[i]
            count +=1
    return count

def substring_count_vowel_less_then_threshold(arr, threshold):
    def count(s, t):
        count = 0
        l = v_count =0
        for r in range(len(s)):
            v_count += 1 if s[r] in 'aeiou' else 0
            while v_count > t and l <= r:
                v_count -= 1 if s[l] in 'aeiou' else 0
                l += 1
            count += (r - l + 1)
        return count
    return [count(s, threshold) for s in arr]

def minSwap(arr, n):
    # min swaps when swapping element can be at any distance apart
    ans, temp, h = 0, arr.copy(), {}
    temp.sort()
    for i in range(n):
        h[arr[i]] = i
    for i in range(n):
        if (arr[i] != temp[i]):
            ans += 1
            init = arr[i]
            arr[i], arr[h[temp[i]]] = arr[h[temp[i]]], arr[i]
            h[init] = h[temp[i]]
            h[temp[i]] = i
    return ans

def countSmaller(self, nums: List[int]) -> List[int]:
    # minimum adjacent swap to make array sorted
    # implement segment tree
    def update_bit(index, value, tree, size):
        index += 1
        while index <= size:
            tree[index] += value
            index += (index & (-index))

    def query_bit(index, tree, size):
        index += 1
        count = 0
        while index > 0:
            count += tree[index]
            index -= (index & (-index))
        return count

    offset = 10 ** 4  # offset negative to non-negative
    size = 2 * 10 ** 4 + 1  # total possible values in nums
    tree = [0] * (size + 1)
    result = []
    for num in reversed(nums):
        # num + offset-1= to find number less than (num + offset)
        smaller_count = query_bit(num + offset - 1, tree, size)
        result.append(smaller_count)
        update_bit(num + offset, 1, tree, size)
    return reversed(result)

def missing_and_repeated_num(arr):
    #https://leetcode.com/company/amazon/discuss/5628696/Repeat-and-Missing-Number-Array
    repeated = missing = None
    for i, num in enumerate(arr):
        if arr[abs(num) - 1] < 0:
            repeated = abs(num)
        else:
            arr[abs(num) - 1] *= (-1)
    for i, num in enumerate(arr):
        if num > 0:
            missing = i + 1
            break
    return [repeated, missing]

# Pair largest one from second part to smallest one from first part
def getMinSizegamestoragependrive(gameSizes, k):
    # https://leetcode.com/company/amazon/discuss/5657213/Amazon-OA
    gameSizes.sort(reverse = True)
    maxi = gameSizes[0]

    for index in range(len(gameSizes) - k):
        maxi = max(gameSizes[k - index - 1] + gameSizes[k + index], maxi)

    return maxi

def assign_request_to_server(n, request):
    #https://leetcode.com/company/amazon/discuss/5580409/Amazon-OA-Hackerrank-Questions-or-Aug-2024
    def binary_search(l, r):
        if r < l:
            return 0
        target = arr[r]
        while l <= r:
            m = l + (r - l) // 2
            if arr[m] == target:
                r = m - 1
            elif arr[m] > target:
                l = m + 1
        return l

    arr, result = [0] * n, []
    for req in request:
        idx = binary_search(0, req - 1)
        arr[idx] += 1
        result.append(idx)
    return result

def consecuative_k_block_flip(s, k):
    # https://leetcode.com/company/amazon/discuss/5580409/Amazon-OA-Hackerrank-Questions-or-Aug-2024
    l = 0
    max_len =  current_flip = 0
    for r, c in enumerate(s):
        if r > 0 and s[r] == '1' and s[r-1] == '0':
            current_flip += 1
        while (current_flip > k or (s[r] == '0' and current_flip == k)) and l <= r:
            l += 1
            if l < len(s) and s[l] == '1' and s[l - 1] == '0':
                current_flip -= 1
        max_len = max(max_len, r - l + 1)
    return max_len

def max_sumarray_greater_k_by_len(arr, k):
    #https://leetcode.com/company/amazon/discuss/5635920/Sliding-window-questions-help-(-Amazon-and-Tiktok-OA-questions-)

    smallest_right, stack = [len(arr)] * len(arr), []
    for i, a in enumerate(arr):
        while stack and arr[stack[-1]] > a:
            smallest_right[stack.pop()] = i
        stack.append(i)

    smallest_left, stack = [-1] * len(arr), []
    for i in range(len(arr) - 1, -1, -1):
        while stack and arr[stack[-1]] > arr[i]:
            smallest_left[stack.pop()] = i
        stack.append(i)
    print(smallest_left, smallest_right)
    max_len = 0
    for i, a in enumerate(arr):
        if a * (smallest_right[i] - smallest_left[i] -1) > k:
            max_len = max(max_len, smallest_right[i] - smallest_left[i] -1)
    return max_len

def beautiness_of_subarray(arr, k):
    # https://leetcode.com/company/amazon/discuss/5635920/Sliding-window-questions-help-(-Amazon-and-Tiktok-OA-questions-)
    queue = deque()
    score = 0
    for i, a in enumerate(arr):
        if queue and i - queue[0] + 1 > k:
            queue.popleft()
        while queue and arr[queue[-1]] <= a:
            queue.pop()
        queue.append(i)
        if i + 1 >= k:
            score += len(queue)
    return score

def regular_expression_mating_with_brackets():
    # https://leetcode.com/company/amazon/discuss/5673621/AMAZON-OA-2024-REGEX
    def isMatchHelper(sIndex, pIndex):
        if (sIndex, pIndex) in memory.keys():
            return memory[(sIndex, pIndex)]
        elif pIndex == pLen:
            return sIndex == sLen
        else:
            firstmatch = sIndex < sLen and p[pIndex] in ['.', s[sIndex]]
            if pIndex + 1 < pLen and p[pIndex + 1] == '*':
                memory[(sIndex, pIndex)] = isMatchHelper(sIndex, pIndex + 2) or (
                            firstmatch and isMatchHelper(sIndex + 1, pIndex))
            elif p[pIndex] == '(':
                closing_bracket_index = p.find(')', pIndex, pLen)

                if closing_bracket_index == -1:
                    memory[(sIndex, pIndex)] = False
                    return memory[(sIndex, pIndex)]

                pattern = p[pIndex + 1: closing_bracket_index]
                firstmatch = (sIndex + len(pattern) <= sLen and s[sIndex: sIndex + len(pattern)] == pattern)

                next_p_index = closing_bracket_index + 2
                next_s_index = sIndex + len(pattern)

                if closing_bracket_index + 1 < pLen and p[closing_bracket_index + 1] == '*':
                    memory[(sIndex, pIndex)] = isMatchHelper(sIndex, next_p_index) or (
                                firstmatch and isMatchHelper(next_s_index, pIndex))
                else:
                    memory[(sIndex, pIndex)] = firstmatch and isMatchHelper(next_s_index, next_p_index)
            else:
                memory[(sIndex, pIndex)] = firstmatch and isMatchHelper(sIndex + 1, pIndex + 1)

        return memory[(sIndex, pIndex)]

    s, p = "pbsdf", "pa*sdf"
    memory, sLen, pLen = {}, len(s), len(p)
    print(isMatchHelper(0, 0))
print(regular_expression_mating_with_brackets())

def lexigraphically_greater_string(s):
    # https://leetcode.com/company/amazon/discuss/5700189/Amazon-Online-Assessment
    is_changed, stack = False, []
    for c in s:
        if not is_changed and stack and stack[-1] == c:
            if c == 'z':
                return "-1"
            else:
                stack.append(chr(ord(c) + 1 ))
                is_changed = True
        elif is_changed:
            if stack[-1] == 'a':
                stack.append('b')
            else:
                stack.append('a')
        else:
            stack.append(c)

    if not is_changed:
        for i in range(len(stack) - 1, -1, -1):
            if stack[i] == 'z':
                continue
            elif len(stack) == 1:
                is_changed = True
                stack[i] = chr(ord(stack[i]) + 1)
            elif i == 0:
                is_changed = True
                if ord(stack[i+1]) != ord(stack[i]) + 1:
                    stack[i] = chr(ord(stack[i]) + 1)
                elif ord(stack[i]) + 2 <= ord('z'):
                    stack[i] = chr(ord(stack[i]) + 2)
                else:
                    is_changed = False
                if is_changed:
                    break
            elif i == len(stack) - 1:
                is_changed = True
                if ord(stack[i-1]) != ord(stack[i]) + 1:
                    stack[i] = chr(ord(stack[i]) + 1)
                elif ord(stack[i]) + 2 <= ord('z'):
                    stack[i] = chr(ord(stack[i]) + 2)
                else:
                    is_changed = False
                if is_changed:
                    break
            elif ord(stack[i-1]) != ord(stack[i]) + 1 and ord(stack[i+1]) != ord(stack[i]) + 1:
                is_changed = True
                stack[i] = chr(ord(stack[i]) + 1)
                break
            elif ord(stack[i]) + 2 <= ord('z') and ord(stack[i-1]) != ord(stack[i]) + 2 and ord(stack[i+1]) != ord(stack[i]) + 2:
                is_changed = True
                stack[i] = chr(ord(stack[i]) + 2)
                break
            # elif ord(stack[i]) + 3 <= ord('z'):
            #     is_changed = True
            #     stack[i] = chr(ord(stack[i]) + 3)
            #     break

    return ''.join(stack) if is_changed else "-1"

def power_booster(a, b, c):
    player = [None] * len(a)
    max0 = max1 = 0
    for i in range(len(a)):
        player[i] = [a[i], b[i], c[i]]
        player[i].sort()
        max0 = max(max0, player[i][0])
        max1 = max(max1, player[i][1])
    count = 0
    for i in range(len(a)):
        if player[i][2] > max1 and player[i][1] > max0:
            count += 1
    return count

def circular_warehouse(arr):
    # https://leetcode.com/discuss/interview-question/5834906/amazon-new-grad-oa/2648819
    def helper(arr, target):
        cost = balance = min_balance = 0
        for a in arr:
            balance += (a - target)
            min_balance = min(min_balance, balance)
            cost += balance
        return cost - min_balance * len(arr)
    target = sum(arr) // len(arr)
    return min(helper(arr, target), helper(arr[::-1], target))

def max_rectangle_area(arr):
    arr.sort(reverse= True)
    max_area, pairs, r = 0, [], 1
    while r < len(arr):
        if arr[r - 1] == arr[r] or arr[r - 1] - 1 == arr[r]:
            pairs.append(arr[r])
            r += 2
        else:
            r += 1
        if len(pairs) == 2:
            max_area += (pairs[0] * pairs[1])
            pairs = []

    return max_area


def KMP(needle, haystack):
    if len(needle) == 0:
        return 0
    # pre-process needle
    lps = [0] * len(needle)
    previous_lps, right = 0, 1
    while right < len(needle):
        if needle[right] == needle[previous_lps] or not previous_lps:
            if needle[right] == needle[previous_lps]:
                previous_lps += 1
            lps[right] = previous_lps
            right += 1
        else:
            previous_lps = lps[previous_lps - 1]
    # KMP
    index_haystack, index_needle = 0, 0
    while index_haystack < len(haystack):
        if haystack[index_haystack] == needle[index_needle]:
            index_needle += 1
            index_haystack += 1
        else:
            if index_needle == 0:
                index_haystack += 1
            else:
                index_needle = lps[index_needle - 1]
        if index_needle == len(needle):
            return index_haystack - index_needle
    return -1



def ordered_configuration(configuration: str) -> list[str]:
    validator = re.compile(r'(\d{4})(\w{10})')
    split_configs = sorted(configuration.split("|"))
    result = {}
    last_val = None
    for val in split_configs:
        x = validator.fullmatch(val)
        if x and len(x.groups()) == 2:
            id, config = x.group(1), x.group(2)
            if last_val == None:
                last_val = int(id)
            elif id in result or last_val != int(id) - 1:
                return False
            last_val = int(id)
            result[id] = config
        else:
            print(f"Failed, {x}")
            return False
    return list(result.values())


def kth_smallest_in_window(arr, k, m):
    sorted_window, result = sortedcontainers.SortedList(), []
    for i in range(m):
        sorted_window.add(arr[ i])
    result.append(sorted_window[k-1])
    for i in range(m, len(arr)):
        sorted_window.add(arr[i])
        sorted_window.discard(arr[i-m])
        result.append(sorted_window[k-1])
    return result

def k_repetitive_score(s, k):
    n = len(s)
    ans = 0
    start = 0
    counts = [0] * 26

    for end in range(n):
        eint = ord(s[end]) - ord('a')
        counts[eint] += 1

        while counts[ord(s[start]) - ord('a')] > k or (s[start] != s[end] and counts[eint] >= k):
            counts[ord(s[start]) - ord('a')] -= 1
            start += 1

        if max(counts) >= k:
            ans += start + 1
    return ans

def k_consecutive_days(arr, k):
    left = left_exclude = 0
    max_point = current_total = current_total_days = 0

    for right, d in enumerate(arr):
        current_total_days += d
        current_total += (d * (d+1) / 2)

        while current_total_days > k:

            extra = current_total_days - k

            if extra >= arr[left] - left_exclude:
                current_total -= (arr[left] * (arr[left] + 1) / 2)
                current_total_days -= (arr[left] - left_exclude)
                left_exclude = 0
                left += 1
            else:
                left_exclude += extra
                current_total_days -= extra
        max_point = max(max_point, current_total - (left_exclude * (left_exclude + 1)/2))
    return max_point

# reverse binary string
def min_operation(string):
    i, r = 0, 0
    while i < len(string):
        if string[i] == string[len(string) - 1- r]:
            r += 1
        i += 1
    return len(string) - r

'''
The developers at Amazon want to perform a reliability drill on some servers. There are n servers where the ith server can serve request[i] number of requests and has an initial health of health[i] units. Each second, the developers send the maximum possible number of requests that can be served by all the available servers. With the request, the developers can also send a virus to one of the servers that can decrease the health of a t particular server by k units. The developers can choose the server where the virus should be sent. A server goes down when its health is less than or equal to 0. After all the servers are down, the developers must send one more request to conclude the failure of the application.
Find the minimum total number of requests that the developers must use to bring all the servers down.
Example
Consider n = 2, request = [3, 4], health = [4, 6], k =3
The minimum number of requests required is 21.
'''
def health(request, health, k):
    heap = []
    current_total_request, total = sum(request), 0
    for r, h in zip(request, health):
        heap.append((-( r / math.ceil(h / k) ), r, h))
    heapq.heapify(heap)
    while heap:
        total_down, r, h = heapq.heappop(heap)
        total += (current_total_request * math.ceil(h/k))
        current_total_request -= r
    return total + 1

def move_blocks(weights):
    mini = weights.index(min(weights))
    maxi = weights.index(max(weights))
    return mini + len(weights) - 1 - maxi - (mini > maxi)


def find_variability(input_string):
    counter = defaultdict(int)
    ans = 1

    for i in range(len(input_string) - 1, -1, -1):
        counter[input_string[i]] += 1
        ans += len(input_string) - i - counter[input_string[i]]

    return ans
# print("variability",find_variability('abc'))

def password_count_variability(password):
    n, c = len(password), Counter(password)
    total = 1
    for v in c.values():
        total+=v*(n-v)
        n-=v
    return total
#Q 9: largest lexiographic number after m state operation
def getNextState(state):
    return state | (state >> 1)  # Calculate the next state by shifting the current state right by 1 and OR'ing it with the current state

def getMaxAvailable(state, arr):
    # Determine the largest available element based on the current state
    N = len(arr)
    largest = float('-inf')
    position = 0
    while state > 0:
        if state & 1:
            # Calculate the offset from the rightmost bit
            # N-1-position gives the corresponding index in the array
            largest = max(largest, arr[N - 1 - position])
        state >>= 1  # Shift the state to the right to check the next bit
        position += 1  # Increment the position to track the offset
    return largest

def solution(arr, state, m):
    N = len(arr)
    state = int(state, 2)  # Convert state from binary string to integer
    res = []
    maxElement = max(arr)  # Find the maximum element in the array
    largest = getMaxAvailable(state, arr)  # Get the largest available element based on the initial state
    stateCanBeUpdated = maxElement != largest  # Flag to determine if state updates are necessary

    while m:
        # Append the largest available element to the result list
        res.append(largest)

        if stateCanBeUpdated:
            nextState = getNextState(state)  # Calculate the next state
            diff = nextState ^ state  # Determine the difference from the current state

            if not diff:
                stateCanBeUpdated = False  # If no new elements are available, stop updating the state
            else:
                largest = max(largest, getMaxAvailable(diff, arr))  # Update the largest available element
                state = nextState  # Update the state
                if largest == maxElement:
                    stateCanBeUpdated = False  # If the largest available element is the maximum element, stop updating the state
        m -= 1

    return res


def nondecreasing_server(power):
    # https://leetcode.com/company/amazon/discuss/6028612/Amazon-or-OA-or-07052024
    c_sum = 0
    max_element = power[0]
    for i in range(1, len(power)):
        current =  power[i] + c_sum
        diff = max_element - current
        if diff > 0:
            c_sum += diff
        max_element = max(max_element, current)
    return c_sum

def idle_robot_count(x, y):
    #https://leetcode.com/company/amazon/discuss/6028612/Amazon-or-OA-or-07052024
    rows = defaultdict(SortedList)
    cols = defaultdict(SortedList)
    for r, c in zip(x, y):
        rows[r].add(c)
        cols[c].add(r)
    idle = 0
    for r, c in zip(x, y):
        if not (r in rows and c in cols):
            continue
        left = bisect.bisect_left(rows[r], c)
        if left == 0:
            continue
        right = bisect.bisect_right(rows[r], c)
        if right == len(rows[r]):
            continue
        left = bisect.bisect_left(cols[c], r)
        if left == 0:
            continue
        right = bisect.bisect_right(cols[c], r)
        if right == len(cols[c]):
            continue
        idle += 1
    return idle


def get_min_connect_time(servers, n):
    servers.sort()
    diff = 0
    max_diff = float('-inf')
    for i in range(0, len(servers) - 1):
        diff += abs(servers[i + 1] - servers[i])
        max_diff = max(max_diff, abs(servers[i + 1] - servers[i]))

    a = n - servers[-1]
    b = servers[0] - 1
    diff += (a + b + 1)

    max_diff = max(max_diff, (a + b + 1))

    return diff - max_diff

def maxShip(weight):
    if max(weight) == weight[-1]:
        return 0
    res = 0
    curMax = 0
    max_right = [0] * len(weight)
    max_right[-1] = weight[-1]
    for i in range(len(weight) - 2, -1, -1):
        max_right[i] = max(max_right[i + 1], weight[i])

    for i in range(len(weight)):
        curMax = max(curMax, weight[i])
        if curMax != weight[i]:
            if (i <= len(weight) - 3 and max_right[i + 1] > weight[-1]) or i == len(weight) - 1:
                res += 1
                curMax = 0
    return res

def minimumAverageDifference(nums):
    c_sum, n = [0], len(nums)
    for num in nums:
        c_sum.append(c_sum[-1] + num)

    answer, index = c_sum[-1] // n, n - 1
    for i in range(1, len(c_sum) - 1):
        diff = abs(c_sum[i] // i - (c_sum[-1] - c_sum[i]) // (n - i))
        if diff < answer:
            index = i - 1
            answer = diff
        elif diff == answer and index > i - 1:
            index = i - 1
    return index

def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
    # wondow[l: r]: if forbidden word exists from k:r, k >= l <=r,
    # we can say no substring from l:r containing k:r is valid
    # so forward l to k + 1
    # search forbidden word in reverse order, so build trie also in reverse order
    def isExist(l, r, word, trie):
        while l <= r:
            if word[r] in trie:
                trie = trie[word[r]]
                if '#' in trie:
                    return r + 1
            else:
                return l
            r -= 1
        return l

    trie = {}
    for w in forbidden:
        current = trie
        for i in range(len(w) - 1, -1, -1):
            if w[i] not in current:
                current[w[i]] = {}
            current = current[w[i]]
        current['#'] = True

    l = max_len = 0
    for r, c in enumerate(word):
        l = isExist(l, r, word, trie)
        max_len = max(max_len, r-l+1)
    return max_len


def LCM_HCF(A):
    # Find the maximum value in the list
    n = max(A)

    # Initialize the divisors list
    divisors = [0] * (n + 1)

    # Count divisors for each number up to n
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            divisors[j] += 1

    # Calculate the result by summing up the divisor counts for each element in A
    res = 0
    for i in range(len(A)):
        res += divisors[A[i]]
    return res

def min_distance_to_friends_house(arr):
    result = []
    prefix_sum = [0] * (len(arr) +1)
    for i in range(len(arr)):
        prefix_sum[i + 1] = prefix_sum[i] + arr[i]
    for i, d in enumerate(arr):
        l, r = 0, len(arr) - 1
        if d - arr[0] < arr[-1] - d:
            r -= 1
        else:
            l = 1
        distance = 0
        if l < i:
            distance += (prefix_sum[i + 1] - prefix_sum[l + 1] - arr[l] * (i - l))
        if i < r:
            distance += (prefix_sum[r + 1] - prefix_sum[i + 1] - arr[i] * (r - i))
        result.append(distance)
    return result

def car_inbrige(U, weight):
    num_weights = len(weight)
    num_skips = 0
    left = 0
    right = 1

    while right < num_weights:
        if weight[left] + weight[right] > U:
            num_skips += 1
            if weight[left] > weight[right]:
                left = right
        else:
            left = right
        right += 1
    if weight[left] > U:
        num_skips += 1
    return num_skips


def smallest_palindrom_wildcard(s):
    letters, wildcard = [0] * 26, 0
    for c in s:
        if c == '?':
            wildcard += 1
        else:
            letters[ord(c) - ord('a')] += 1

    odd_count, last_odd = 0, -1
    for i in range(26):
        if letters[i] & 1:
            last_odd = i
            if wildcard > 0:
                letters[i] += 1
                wildcard -= 1
            else:
                odd_count += 1
    if odd_count > 1:
        return "-1"
    if wildcard & 1:
        letters[last_odd] -= 1
        wildcard += 1

    letters[0] += wildcard

    result, l, odd_character = list(s), 0, None
    for index in range(26):
        if letters[index] == 0:
            continue
        if letters[index] & 1:
            odd_character = chr(ord('a') + index)
        result[l: l + letters[index] // 2] = [chr(ord('a') + index)] * (letters[index] // 2)
        r = len(s) - 1 - l
        result[r - letters[index] // 2 + 1 : r + 1] = [chr(ord('a') + index)] * (letters[index] // 2)
        l += letters[index] // 2

    if odd_character is not None:
        result[l] = odd_character
    return ('').join(result)

def removeDuplicateLetters(self, s: str) -> str:
    occurence = {c: i for i, c in enumerate(s)}
    stack = []
    seen = set()

    for i, c in enumerate(s):
        if c not in seen:
            while stack and c < stack[-1] and occurence[stack[-1]] > i:
                seen.remove(stack.pop())
            stack.append(c)
            seen.add(c)
    return ''.join(stack)

# obfuscated message by rotation
def makeMinRotation(s: str) -> str:
    N = len(s)

    l, r = N-1, N-1
    m = N-1 # location of min character
    for i in reversed(range(N)):
        if s[i] < s[m]:
            m = i
        elif s[i] > s[m]:
            l=i
            r=m

    if l != r:
        return s[:l] + s[r] + s[l:r] + s[r+1:]
    else:
        return s


def max_equal_cost_packages_correct(cost):
    from collections import Counter

    # Count the frequency of each cost
    cost_count = Counter(cost)
    max_packages = 0

    # Iterate over possible target package costs
    for target_cost in range(min(cost), 2 * max(cost) + 1):  # The maximum possible sum of two costs
        current_packages = 0
        temp_count = cost_count.copy()
        pairs = 0

        for item_cost in temp_count:
            complement = target_cost - item_cost
            if item_cost == target_cost:
                current_packages += temp_count[item_cost]
                temp_count[item_cost] = 0
            elif complement in temp_count:
                if complement == item_cost:  # Special case for same-cost pairing
                    pairs = temp_count[item_cost] // 2
                else:  # Pair items with different costs
                    pairs = min(temp_count[item_cost], temp_count[complement])

                current_packages += pairs
                temp_count[item_cost] = 0
                temp_count[complement] = 0

        max_packages = max(max_packages, current_packages)

    return max_packages

def product_identifier(p_id):
    start = 0
    min_window = math.inf
    if p_id[0] == p_id[-1]:
        return len(p_id) - 1
    for i, c in enumerate(p_id):
        if c == p_id[0]:
            start = i
        elif c == p_id[-1]:
            min_window = min(min_window,  i - start + 1)
    return len(p_id) - min_window if min_window != math.inf else 0

def continuous_segment(capacity):
    count, tracker = 0, {}
    for i in range(2, len(capacity)):
        count += (capacity[i - 2] == capacity[i - 1] == capacity[i])
    for i, cap in enumerate(capacity):
        if cap in tracker and cap == (capacity[i - 1] - capacity[tracker[cap]]):
            count += 1
        tracker[cap] = i
        if i > 0:
            capacity[i] += capacity[i - 1]
    return count

def parenthesis_perfection_kit(s, parenthesis, ratings):
    stack, efficiency = [], 0
    opening, closing = [], []
    for i, p in enumerate(parenthesis):
        if p == '(':
            opening.append(-ratings[i])
        else:
            closing.append(-ratings[i])
    heapq.heapify(opening)
    heapq.heapify(closing)
    for p in s:
        if p == ')':
            if stack:
                stack.pop()
            else:
                if not opening:
                    return -1
                efficiency += (-heapq.heappop(opening))
        else:
            stack.append(p)
    while stack:
        if not closing:
            return -1
        efficiency += (-heapq.heappop(closing))
    while opening and closing:
        a = (-heapq.heappop(opening)) + (-heapq.heappop(closing))
        if a < 0:
            return efficiency
        efficiency += a
    return efficiency

def interesting_pair(arr, sum_v):
    # The keypoint here is |a - b| + |a + b| = 2 * max(abs(a), asb(b))
    # https://leetcode.com/company/amazon/discuss/6055780/Interesting-Pairs-or-FAANG-OA
    if sum_v & 1:
        return 0
    sum_v //= 2
    equal = smaller = 0
    for a in arr:
        equal += (abs(a) == sum_v)
        smaller += (abs(a) < sum_v)
    return (equal * (equal - 1)) // 2 + equal * smaller


def count_discount_pairs_optimized(prices):
    """Count pairs (i, j) where i < j and (prices[i] + prices[j]) is a power of three using a hashmap."""
    # https://leetcode.com/company/amazon/discuss/6032414/Amazon-or-OA-SDE-II-or-11082024
    count = 0
    freq = defaultdict(int)
    max_price = max(prices)

    # Precompute all possible powers of three within the sum range
    powers_of_three = []
    power = 1
    while power <= 2 * max_price:
        powers_of_three.append(power)
        power *= 3

    # Traverse prices and use hashmap to count valid pairs
    for price in prices:
        for power in powers_of_three:
            complement = power - price
            if complement in freq:
                count += freq[complement]
        freq[price] += 1

    return count

def dictinct_category_sum(nums):
    # https://leetcode.com/company/amazon/discuss/6048204/Amazon-Online-Assessment-Question-2024-November
    res = current = 0
    tracker = defaultdict(int)
    for i in range(1, len(nums)+ 1):
        current += (i - tracker[nums[i-1]])
        res += current
        tracker[nums[i-1]] = i
    return res

def overall_min_effort_storage_bin(arr):
    # https://leetcode.com/company/amazon/discuss/6044882/Amazon-Online-Assessment-Question-2024-November
    counter = Counter(arr)
    arr_min = min(arr)
    arr_max = max(arr)
    count = 0
    for num in range(arr_min, arr_max + 1):
        if num not in counter:
            continue
        for divisor in range(num, arr_max + 1, num):
            count += (num * counter[divisor])
            del counter[divisor]
    return count
def min_off_smart_bulb(arr):
    # https://leetcode.com/discuss/interview-question/5911956/Amazon-OA-SDE-2-or-Oct-24/2676896
    arr.sort()
    c_sum = off_count = 0
    for bulb in arr:
        if c_sum > bulb:
            off_count += 1
        else:
            c_sum += bulb
    return off_count

def amazon_parcel_efficiency(arr):
    # https://leetcode.com/discuss/interview-question/5966700/amazon-online-assessment-questions/
    # https://leetcode.com/company/amazon/discuss/5993397/Amazon-SDE-off-campus-OA
    heap, efficiency = [], sum(arr)
    n = len(arr) // 2
    for i in range(n):
        heapq.heappush(heap, arr[i])
        heapq.heappush(heap, arr[len(arr) - 1 - i])
        efficiency -= heapq.heappop(heap)
    return efficiency

# print("parcel", amazon_parcel_efficiency([4, 4, 8, 5, 3, 2]))
# print(min_off_smart_bulb([2, 1, 3, 4, 3]))
# print(bisect.bisect_left([1,2, 2, 2, 3, 5], 4), bisect.bisect_right([1,2, 2, 2, 3, 5], 4))
# print(idle_robot_count([0,0,1,1,1,2,2,4,4,0,2], [0,3,1,3,4,0,2,2,3,2,3]))
# print(overall_min_effort_storage_bin([3, 6, 2, 5, 25]))
# print(dictinct_category_sum([1, 1, 1]))
# print(min_errors_exclamation("101!1", 2, 3))
# print(count_discount_pairs_optimized([1, 2, 8]))
# print(interesting_pait([1,4,-1,2], 4))
# print(parenthesis_perfection_kit('()', '(())', [4,2,-3, -3]))
# print(continuous_segment([6, 1, 2, 3, 6]))
# print(product_identifier("babdcaac"))
# print(max_equal_cost_packages_correct([10,2,1]))
# print(makeMinRotation('aahhb'))
# print(smallest_palindrom_wildcard("a?rt???"))
# print(maxShip([4,3,6,5,3,4,7,1]))

# print(password_count_variability("aaaabcc"))
# print(find_variability("aaaabcc"))
# print(move_blocks([11,12,10,4]))
# print(max_rectangle_area( [2, 6, 6, 2, 3, 5]))

# print(circular_warehouse([6,6,6,3,4]))
# print(power_booster([9,4,2], [5,12,10], [11,3,13]))
# print(lexigraphically_greater_string("yyyyyy"))
# regular_expression_mating_with_brackets()
# print(beautiness_of_subarray( [5,4,3,2,1], 3))
# print(max_sumarray_greater_k_by_len([3, 1, 5, 6, 4, 3], 5))
# print(consecuative_k_block_flip("11111", 0))
# print(assign_request_to_server(5, [3,2,3,2,4]))
# print(getMinSizegamestoragependrive([9, 2, 4, 6], 3))
# print(missing_and_repeated_num([3, 1, 2, 5 ,3]))
# print(substring_count_vowel_less_then_threshold(["lciy","ttrd","aod"], 1))
# print(max_zero_by_decreasing_prefix([9, 6, 7, 2, 7, 2]))
# print(minimized_max_after_distribution_k([2, 3, 4, 5, 6], 10))
# print(getLargest_index_len_outliers([1,2,5,4,6], [5,6,7,3,4]))
# print(reduandant_string('abbacc', -1, 2))
# print(consecutive_racer([1, 4, 4, 2, 2, 4], 2))
# print(min_operation_to_make_palindrome("122527"))
# print(min_generation([1,1,2,4]))
# print(gas_station_query([[2, 4]], [3, 6, 10, 15, 20]))
# print(regex_with_single_star("hackerrank", "*"))
# print(max_zero_in_array([4,3,5,5,3]))
# print(findMaximumQuality([89, 48, 14], 1))
# print(wait_time([1,1,1, 5,6]))
# print(number_of_ware_hosue([ - 2, 1 , 0 ], 8)) #[-2, 1, 0], 8
# print(two_ware_house([1,2,3, 4, 5]))
# print(total_charge_after_merge([-1, 3, 2]))
# print(paint_canvas(2,3,2, [[1,2], [2,3], [2,1],[1,3],[2,2],[1,1]]))
# print(pnl([5,2,1,3]))
# print(getMaximumTasks(7, [4, 5, 2, 4], [5, 6, 3, 4]))
# print(getMaxAlternatingMusic('1011', 1))
# print(symmetrical_number("caabdbaac"))
# print(box_removed_less_capacity([2]  , 2))
# print(perfect_anagram("abcacb"))
# print(replacing_exclamation("!0!1", 3, 4))



# question lsit: https://leetcode.com/company/amazon/discuss/6075975/Amazon-SDE2-50-posts-in-1-DSA%2BLP%2BLLDHLD
