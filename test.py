
def arraySign(nums):
    neg = 0

    for n in nums:
        if n == 0:
            return 0
        neg += (1 if n < 0 else 0)

    return -1 if neg % 2 else 1
