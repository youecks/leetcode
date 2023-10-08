# Depth First Iterative

class Node:
  def __init__(self, val):
    self.val = val
    self.left = None
    self.right = None

def tree_min_value(root):
  stack = [ root ]
  smallest = float("inf")
  while stack:
    current = stack.pop()
    if current.val < smallest:
      smallest = current.val

    if current.left is not None:
      stack.append(current.left)
    if current.right is not None:
      stack.append(current.right)

  return smallest
  