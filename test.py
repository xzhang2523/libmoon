from collections import Counter
import heapq


def reorganize_string(s: str) -> str:
    # 统计字符出现次数
    counter = Counter(s)
    max_heap = [(-freq, char) for char, freq in counter.items()]
    heapq.heapify(max_heap)
    prev_freq, prev_char = 0, ''
    result = []
    while max_heap:
        freq, char = heapq.heappop(max_heap)
        result.append(char)
        # 如果上一个字符还有剩余，将其重新推回堆中
        if prev_freq < 0:
            heapq.heappush(max_heap, (prev_freq, prev_char))
        # 更新当前字符的频率和字符
        prev_freq, prev_char = freq + 1, char
    result_str = ''.join(result)
    # 如果重新组织后的字符串与原长度相等，则返回结果，否则返回空字符串
    if len(result_str) == len(s):
        return result_str
    else:
        return "无法重新组织"



if __name__ == '__main__':
    # 示例使用
    s = "aaabbc"
    result = reorganize_string(s)
    print(result)
