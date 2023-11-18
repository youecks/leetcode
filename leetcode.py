###########################
## Linked List Traversal ##
###########################

# Iterative
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

a = Node('A')
b = Node('B')
c = Node('C')
d = Node('D')

a.next = b
b.next = c
c.next = d

# Iterative Traversal
def print_list(head):
    current = head
    while current is not None:
        print(current.val)
        current = current.next

# Recursive Traversal
def print_list(head):
    if head is None:
        return 
    print(head.val)
    print_list(head.next)

print_list(b)


########################
## Linked List Values ##
########################

# Write a function, linked_list_values, that takes in the head of a linked list as an argument. The function should return a list containing all values of the nodes in the linked list.

# Iterative
def linked_list_values(head):
  values = []
  current = head
  while current is not None:
    values.append(current.val)
    current = current.next
  return values

# Recursive

def linked_list_values(head):
  values = []
  _linked_list_values(head, values)
  return values

def _linked_list_values(head, values):
  if head is None:
    return
  values.append(head.val)
  _linked_list_values(head.next, values)


##############
## Sum List ##
##############

# Iterative
class Node:
  def __init__(self, val):
    self.val = val
    self.next = None

def sum_list(head):
  
  current = head
  total_sum = 0
  
  while current is not None:
    total_sum += current.val
    current = current.next 
  
  
  return total_sum

# n = number of nodes
# Time: O(n)
# Space: O(1)


# Recursive
def sum_list(head):
  if head is None:
    return 0
  return head.val + sum_list(head.next)

# n = number of nodes
# Time: O(n)
# Space: O(n)


######################
## Linked List Find ##
######################

class Node:
  def __init__(self, val):
    self.val = val
    self.next = None

a = Node("a")
b = Node("b")
c = Node("c")
d = Node("d")

a.next = b
b.next = c
c.next = d

# Iterative
def linked_list_find(head, target):
  
    current = head
  
    while current is not None:
        if current.val == target:
            return True 
        current = current.next 
    return False

# n = number of nodes
# Time: O(n)
# Space: O(1)

# Recursive

def linked_list_find(head, target):
  if head is None:
    return False
  if head.val == target:
    return True
  return linked_list_find(head.next, target)

# n = number of nodes
# Time: O(n)
# Space: O(n)


####################
## Get Node Value ##
####################

# Recursive

class Node:
   def __init__(self, val):
      self.val = val
      self.next = None

def get_node_value(head, index):
  if head is None:
    return None
  if index == 0:
     return head.val
  
  return get_node_value(head.next, index - 1)

# Iterative

class Node:
  def __init__(self, val):
    self.val = val
    self.next = None
    
def get_node_value(head, index):
  current = head
  count = 0
  
  while current is not None:
    if count == index:
      return current.val
    count += 1
    current = current.next
  return None
    

##################
## Reverse List ##
##################

# Iterative
class Node:
  def __init__(self, val):
    self.val = val
    self.next = None
    
def reverse_list(head):
  prev = None 
  current = head
  while current is not None:
      next = current.next
      current.next = prev
      prev = current
      current = next
  return prev

# Recursive

def reverse_list(head, prev = None):
   if head is None:
      return prev
   next = head.next
   head.next = prev
   return reverse_list(next, head)


#################
## Zipper List ##
#################

# Iterative

def zipper_lists(head_1, head_2):
  tail = head_1
  current_1 = head_1.next
  current_2 = head_2
  count = 0
  while current_1 is not None and current_2 is not None:
    if count % 2 == 0:
      tail.next = current_2
      current_2 = current_2.next
    else:
      tail.next = current_1
      current_1 = current_1.next
    tail = tail.next
    count += 1
    
  if current_1 is not None:
    tail.next = current_1
  if current_2 is not None:
    tail.next = current_2
    
  return head_1

# n = length of list 1
# m = length of list 2
# Time: O(min(n, m))
# Space: O(1)

# Recursive

def zipper_lists(head_1, head_2):
  if head_1 is None and head_2 is None:
    return None
  if head_1 is None:
    return head_2
  if head_2 is None:
    return head_1
  next_1 = head_1.next
  next_2 = head_2.next
  head_1.next = head_2
  head_2.next = zipper_lists(next_1, next_2)
  return head_1  

# n = length of list 1
# m = length of list 2
# Time: O(min(n, m))
# Space: O(min(n, m))


#################
## Merge Lists ##
#################

# Iterative 

class Node: 
  def __init__(self, val):
    self.val = val
    self.next = None

def merge_lists(head_1, head_2):
  dummy_head = Node(None)
  tail = dummy_head
  current_1 = head_1
  current_2 = head_2
  
  while current_1 is not None and current_2 is not None:
    if current_1.val < current_2.val:
      tail.next = current_1
      current_1 = current_1.next
    else:
      tail.next = current_2
      current_2 = current_2.next
    tail = tail.next
    
  if current_1 is not None:
      tail.next = current_1
    
  if current_2 is not None:
      tail.next = current_2
  
  return dummy_head.next

# n = length of list 1
# m = length of list 2
# Time: O(min(n, m))
# Space: O(1)

# recursive

def merge_lists(head_1, head_2):
  if head_1 is None and head_2 is None:
    return None
  if head_1 is None:
    return head_2
  if head_2 is None:
    return head_1
  if head_1.val < head_2.val:
    next_1 = head_1.next
    head_1.next = merge_lists(next_1, head_2)
    return head_1
  else:
    next_2 = head_2.next
    head_2.next = merge_lists(head_1, next_2)
    return head_2
  
# n = length of list 1
# m = length of list 2
# Time: O(min(n, m))
# Space: O(min(n, m))


######################
## Is Univalue List ##
######################

# Iterative

class Node:
  def __init__(self, val):
    self.val = val
    self.next = None

def is_univalue_list(head):
  current = head
  while current is not None:
    if head.val != current.val:
      return False
    current = current.next 
  return True

# n = number of nodes
# Time: O(n)
# Space: O(1)

# Recursive 

def is_univalue_list(head, prev_val = None):
  if head is None:
    return True
  if prev_val is None or head.val == prev_val:
    return is_univalue_list(head.next, head.val)
  else:
    return False
  
# n = number of nodes
# Time: O(n)
# Space: O(n)


####################
## Longest Streak ##
####################

# Iterative

class Node:
  def __init__(self, val):
    self.val = val
    self.next = None

def longest_streak(head):
  max_streak = 0
  current_streak = 0
  prev_val = None
  
  current_node = head
  while current_node is not None:
    if current_node.val == prev_val:
      current_streak += 1
    else:
      current_streak = 1
  
    prev_val = current_node.val
    if current_streak > max_streak:
      max_streak = current_streak
    
    current_node = current_node.next
    
  return max_streak

# n = number of nodes
# Time: O(n)
# Space: O(1)


#################
## Remove Node ##
#################

# Iterative 

class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

def remove_node(head, target_val):
    if head.val == target_val:
        return head.next

    current = head
    prev = None
    while current is not None:
        if current.val == target_val:
            prev.next = current.next
            break

        prev = current
        current = current.next
    return head

# n = number of nodes
# Time: O(n)
# Space: O(1)

# Recursive

def remove_node(head, target_val):
  if head is None:
    return None

  if head.val == target_val:
    return head.next

  head.next = remove_node(head.next, target_val)
  return head

# n = number of nodes
# Time: O(n)
# Space: O(n)


#################
## Insert Node ##
#################

# Iterative

class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

def insert_node(head, value, index):
    if index == 0:
        new_head = Node(value)
        new_head.next = head
        return new_head

    count = 0
    current = head
    while current is not None:
        if count == index - 1:
            temp = current.next
            current.next = Node(value)
            current.next.next = temp

        count += 1
        current = current.next

    return head

# n = number of nodes
# Time: O(n)
# Space: O(1)

# Recursive

class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

def insert_node(head, value, index, count = 0):
  if index == 0:
    new_head = Node(value)
    new_head.next = head
    return new_head
  
  if head is None:
    return None
  
  if count == index - 1:
      temp = head.next
      head.next = Node(value)
      head.next.next = temp
      return 
  
  insert_node(head.next, value, index, count + 1)
  return head

# n = number of nodes
# Time: O(n)
# Space: O(n)


########################
## Create Linked List ##
########################

# Iterative 

class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

def create_linked_list(values):
    dummy_head = Node(None)
    tail = dummy_head
    
    for val in values:
        tail.next = Node(val)
        tail = tail.next
    
    return dummy_head.next

# n = length of values
# Time: O(n)
# Space: O(n)

# Recursive

def create_linked_list(values, i = 0):
  if i == len(values):
    return None
  head = Node(values[i])
  head.next = create_linked_list(values, i + 1)
  return head

# n = length of values
# Time: O(n)
# Space: O(n)


###############
## Add Lists ##
###############

# Recursive

class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

def add_lists(head_1, head_2, carry = 0):
  if head_1 is None and head_2 is None and carry == 0:
    return None
  
  val_1 = 0 if head_1 is None else head_1.val
  val_2 = 0 if head_2 is None else head_2.val  
  sum = val_1 + val_2 + carry
  next_carry = 1 if sum > 9 else 0
  digit = sum % 10
  
  result = Node(digit)
  
  next_1 = None if head_1 is None else head_1.next
  next_2 = None if head_2 is None else head_2.next
  
  result.next = add_lists(next_1, next_2, next_carry)
  return result

# n = length of list 1
# m = length of list 2
# Time: O(max(n, m))
# Space: O(max(n, m))

# Iterative

class Node:
    def __init__(self, val):
        self.val = val
        self.next = None
        
def add_lists(head_1, head_2):
  dummy_head = Node(None)
  tail = dummy_head
  
  carry = 0
  current_1 = head_1
  current_2 = head_2
  while current_1 is not None or current_2 is not None or carry == 1:
    val_1 = 0 if current_1 is None else current_1.val
    val_2 = 0 if current_2 is None else current_2.val
    sum = val_1 + val_2 + carry
    carry = 1 if sum > 9 else 0
    digit = sum % 10
    
    tail.next = Node(digit)
    tail = tail.next
    
    if current_1 is not None:
      current_1 = current_1.next
      
    if current_2 is not None:
      current_2 = current_2.next
      
  return dummy_head.next

# n = length of list 1
# m = length of list 2
# Time: O(max(n, m))
# Space: O(max(n, m))


########################
## Depth First Values ##
########################

# Iterative

class Node:
  def __init__(self, val):
    self.val = val
    self.left = None
    self.right = None

def depth_first_values(root):
    if root is None:
       return []
    
    values = []
    stack = [ root ]
    while stack:
        current = stack.pop()
        values.append(current.val)

        if current.right is not None:
            stack.append(current.right)
        if current.left is not None:
            stack.append(current.left)
    return values

# n = number of nodes
# Time: O(n)
# Space: O(n)

# Recursive 

def depth_first_values(root):
  if not root:
    return []
  
  left_values = depth_first_values(root.left)
  right_values = depth_first_values(root.right)
  return [ root.val, *left_values, *right_values ]

# n = number of nodes
# Time: O(n)
# Space: O(n)


##########################
## Breadth First Values ##
##########################

# Iterative - Recursive doesnt work with breadth first

# class Node:
#   def __init__(self, val):
#     self.val = val
#     self.left = None
#     self.right = None

def breadth_first_values(root):
    if root is None:
        return []
    values = []
    queue = [ root ]
    while queue:
        current = queue.pop(0)
        values.append(current.val)

        if current.left:
            queue.append(current.left)
        if current.right:
            queue.append(current.right)
    return values


##############
## Tree Sum ##
##############

# Recursive

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def tree_sum(root):
    if root is None:
        return 0
    return root.val + tree_sum(root.left) + tree_sum(root.right)

# n = number of nodes
# Time: O(n)
# Space: O(n)

# Iterative

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

from collections import deque

def tree_sum(root):
  if not root:
    return 0

  queue = deque([ root ])
  total_sum = 0
  while queue:
    node = queue.popleft()

    total_sum += node.val

    if node.left:
      queue.append(node.left)

    if node.right:
      queue.append(node.right)

  return total_sum

# n = number of nodes
# Time: O(n)
# Space: O(n)


###################
## Tree Includes ##
###################

# Breadth First Search

from collections import deque

def tree_includes(root, target):
  if not root:
    return False
  
  queue = deque([ root ])
  
  while queue:
    node = queue.popleft()
    
    if node.val == target:
      return True
    
    if node.left:
      queue.append(node.left)
      
    if node.right:
      queue.append(node.right)
      
  return False

# n = number of nodes
# Time: O(n)
# Space: O(n)

# Depth first Search

def tree_includes(root, target):
  if not root:
    return False
  
  if root.val == target:
    return True
  
  return tree_includes(root.left, target) or tree_includes(root.right, target)

# n = number of nodes
# Time: O(n)
# Space: O(n)


####################
## Tree Min Value ##
####################

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

# n = number of nodes
# Time: O(n)
# Space: O(n)


###############################
## Max Root To Leaf Path Sum ##
###############################

# Depth First (Recursive)

class Node:
  def __init__(self, val):
    self.val = val
    self.left = None
    self.right = None

def max_path_sum(root):
  if root is None:
    return float("-inf")

  if root.left is None and root.right is None:
    return root.val

  return root.val + max(max_path_sum(root.left), max_path_sum(root.right))

# n = number of nodes
# Time: O(n)
# Space: O(n)


######################
## Tree Path Finder ##
######################

class Node:
  def __init__(self, val):
    self.val = val
    self.left = None
    self.right = None

def path_finder(root, target):
  result = _path_finder(root, target)
  if result is None:
    return None
  else:
    return result[::-1]

def _path_finder(root, target):
  if root is None:
    return None
  
  if root.val == target:
    return [ root.val ]
  
  left_path = _path_finder(root.left, target)
  if left_path is not None:
    left_path.append(root.val)
    return left_path
  
  right_path = _path_finder(root.right, target)
  if right_path is not None:
    right_path.append(root.val)
    return right_path
  
  return None

# n = number of nodes
# Time: O(n)
# Space: O(n)


######################
## Tree Value Count ##
######################

class Node:
  def __init__(self, val):
    self.val = val
    self.left = None
    self.right = None

def tree_value_count(root, target):
  if root is None:
    return 0

  count = 0
  stack = [ root ]
  while stack:
    current = stack.pop()
    if current.val == target:
      count += 1

    if current.left is not None:
      stack.append(current.left)
    if current.right is not None:
      stack.append(current.right)

  return count

# n = number of nodes
# Time: O(n)
# Space: O(n)


##############
## How High ##
##############

# class Node:
#   def __init__(self, val):
#     self.val = val
#     self.left = None
#     self.right = None

def how_high(node):
  if node is None:
    return -1

  left_height = how_high(node.left)
  right_height = how_high(node.right)
  return 1 + max(left_height, right_height)

# n = number of nodes
# Time: O(n)
# Space: O(n)


########################
## Bottom Right Value ##
########################

class Node:
  def __init__(self, val):
    self.val = val
    self.left = None
    self.right = None
  
from collections import deque

def bottom_right_value(root):
  queue = deque([root])

  while queue:
    current = queue.popleft()

    if current.left is not None:
      queue.append(current.left)
    
    if current.right is not None:
      queue.append(current.right)

  return current.val

# n = number of nodes
# Time: O(n)
# Space: O(n)


####################
## All Tree Paths ##
####################

# Recursive

class Node:
  def __init__(self, val):
    self.val = val
    self.left = None
    self.right = None

def all_tree_paths(root):
  if root is None:
    return []
  
  if root.left is None and root.right is None:
    return [ [root.val] ]
  paths = []

  left_sub_paths = all_tree_paths(root.left)
  for sub_path in left_sub_paths:
    paths.append([ root.val, *sub_path ])

  right_sub_paths = all_tree_paths(root.right)
  
  for sub_path in right_sub_paths:
    paths.append([ root.val, *sub_path ])
  return paths

# n = number of nodes
# Time: O(n)
# Space: O(n)


#################
## Tree Levels ##
#################

class Node:
  def __init__(self, val):
    self.val = val
    self.left = None
    self.right = None

def tree_levels(root):
  if root is None:
    return []
  
  levels = []
  stack = [ (root, 0) ]

  while stack:
    node, level_num = stack.pop()

    if len(levels) == level_num:
      levels.append([ node.val ])
    else:
      levels[level_num].append(node.val)

    if node.right is not None:
      stack.append((node.right, level_num + 1))
    if node.left is not None:
      stack.append((node.left, level_num + 1))
    
    
  return levels

# n = number of nodes
# Time: O(n)
# Space: O(n)


#####################
## Levels Averages ##
#####################

# class Node:
#   def __init__(self, val):
#     self.val = val
#     self.left = None
#     self.right = None

from statistics import mean

def level_averages(root):
  levels = []
  fill_levels(root, levels, 0)
  avgs = []
  for level in levels:
    avgs.append(mean(level))
  return avgs

def fill_levels(root, levels, level_num):
  if root is None:
    return

  if len(levels) == level_num:
    levels.append([ root.val ])
  else:
    levels[level_num].append(root.val)

  fill_levels(root.left, levels, level_num + 1)
  fill_levels(root.right, levels, level_num + 1)

# n = number of nodes
# Time: O(n)
# Space: O(n)


###############
## Leaf List ##
###############

# Depth First Iterative

# class Node:
#   def __init__(self, val):
#     self.val = val
#     self.left = None
#     self.right = None

def leaf_list(root):
  if root is None:
    return [ ]
  
  leaves = []
  stack = [ root ]
  while stack:
    current = stack.pop()

    if current.left is None and current.right is None:
      leaves.append(current.val)

    if current.right is not None:
      stack.append(current.right)

    if current.left is not None:
      stack.append(current.left)

  return leaves

# n = number of nodes
# Time: O(n)
# Space: O(n)


##############
## Has Path ##
##############

# Depth First

def has_path(graph, src, dst):
  if src == dst:
    return True
  
  for neighbor in graph[src]:
    if has_path(graph, neighbor, dst) == True:
      return True
    
  return False

# n = number of nodes
# e = number edges
# Time: O(e)
# Space: O(n)


#####################
## Undirected Path ##
#####################

# Depth First

def undirected_path(edges, node_A, node_B):
  graph = build_graph(edges)
  return has_path(graph, node_A, node_B, set())

def build_graph(edges):
  graph = {}
  
  for edge in edges:
    a, b = edge
    
    if a not in graph:
      graph[a] = []
    if b not in graph:
      graph[b] = []
      
    graph[a].append(b)
    graph[b].append(a)
    
  return graph
    
def has_path(graph, src, dst, visited):
  if src == dst:
    return True
  
  if src in visited:
    return False
  
  visited.add(src)
  
  for neighbor in graph[src]:
    if has_path(graph, neighbor, dst, visited) == True:
      return True
    
  return False

# n = number of nodes
# e = number edges
# Time: O(e)
# Space: O(e)


##############
## Has Path ##
##############

# Breadth First

from collections import deque

def has_path(graph, src, dst):
  queue = deque([ src ])
  
  while queue:
    current = queue.popleft()
    
    if current == dst:
      return True
    
    for neighbor in graph[current]:
      queue.append(neighbor)
    
  return False

# n = number of nodes
# e = number edges
# Time: O(e)
# Space: O(n)


################################
## Connected Components Count ##
################################

# Depth First

def connected_components_count(graph):
  visited = set()
  count = 0
  
  for node in graph:
    if explore(graph, node, visited) == True:
      count += 1
      
  return count

def explore(graph, current, visited):
  if current in visited:
    return False
  
  visited.add(current)
  
  for neighbor in graph[current]:
    explore(graph, neighbor, visited)
  
  return True

# n = number of nodes
# e = number edges
# Time: O(e)
# Space: O(n)


#######################
## Largest Component ##
#######################

def largest_component(graph):
  visited = set()
  
  largest = 0
  for node in graph:
    size = explore_size(graph, node, visited)
    if size > largest:
      largest = size
  
  return largest

def explore_size(graph, node, visited):
  if node in visited:
    return 0
  
  visited.add(node)
  
  size = 1
  for neighbor in graph[node]:
    size += explore_size(graph, neighbor, visited)
    
  return size

# n = number of nodes
# e = number edges
# Time: O(e)
# Space: O(n)


################
## Uncompress ##
################

def uncompress(s):
  numbers = '0123456789'
  result = []
  i = 0
  j = 0
  while j < len(s):
    if s[j] in numbers:
      j += 1
    else:      
      num = int(s[i:j])
      result.append(s[j] * num)
      j += 1
      i = j
      
  return ''.join(result)

# n = number of groups
# m = max num found in any group
# Time: O(n*m)
# Space: O(n*m)


##############
## Compress ##
##############

def compress(s):
  s += '!'
  result = []
  i = 0
  j = 0
  while j < len(s):
    if s[i] == s[j]:
      j += 1  
    else:
      num = j - i
      if num == 1:
        result.append(s[i])
      else:
        result.append(str(num)) 
        result.append(s[i])
      i = j
    
  return ''.join(result)

# n = length of string
# Time: O(n)
# Space: O(n)


##############
## Anagrams ##
##############

def anagrams(s1, s2):
  return char_count(s1) == char_count(s2)

def char_count(s):
  count = {}
  
  for char in s:
    if char not in count:
      count[char] = 0
    count[char] += 1
  
  return count

# n = length of string 1
# m = length of string 2
# Time: O(n + m)
# Space: O(n + m)


########################
## Most Frequent Char ##
########################

def most_frequent_char(s):
  count = {}
  for char in s:
    if char not in count:
      count[char] = 0    
    count[char] += 1
    
  best = None
  for char in s:
    if best is None or count[char] > count[best]:
      best = char
  return best

# n = length of string
# Time: O(n)
# Space: O(n)


##############
## Pair Sum ##
##############

def pair_sum(numbers, target_sum):
  previous_numbers = {}
  
  for index, num in enumerate(numbers):
    complement = target_sum - num
    
    if complement in previous_numbers:
      return (index, previous_numbers[complement])
    
    previous_numbers[num] = index

# n = length of numbers list
# Time: O(n)
# Space: O(n)


###################
## Shortest Path ##
###################

# Breadth First

from collections import deque

def shortest_path(edges, node_A, node_B):
  graph = build_graph(edges)
  visited = set([ node_A])
  queue = deque([ ( node_A, 0 ) ])

  while queue:
    node, distance = queue.popleft()

    if node == node_B:
      return distance
    
    for neighbor in graph[node]:
      if neighbor not in visited:
        visited.add(neighbor)
        queue.append((neighbor, distance + 1))

  return -1

def build_graph(edges):
  graph = {}

  for edge in edges:
    a, b = edge
    if a not in graph:
      graph[a] = []
    if b not in graph:
      graph[b] = []
    graph[a].append(b)
    graph[b].append(a)

  return graph

# e = number edges
# Time: O(e)
# Space: O(e)


##################
## Island Count ##
##################

def island_count(grid):
    visited = set()
    count = 0

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if explore(grid, r, c, visited) == True:
                count += 1
    return count

def explore(grid, r, c, visited):
    row_inbounds = 0 <= r < len(grid)
    col_inbounds = 0 <= c < len(grid[0])
    if not row_inbounds or not col_inbounds:
        return False
    
    if grid[r][c] == 'W':
        return False
    
    pos = (r, c)
    if pos in visited:
        return False
    visited.add(pos)

    explore(grid, r - 1, c, visited)
    explore(grid, r + 1, c, visited)
    explore(grid, r, c - 1, visited)
    explore(grid, r, c + 1, visited)

    return True

# r = number of rows
# c = number of columns
# Time: O(rc)
# Space: O(rc)


####################
## Minimum Island ##
####################

def minimum_island(grid):
  visited = set()
  min_size = float("inf")
  for r in range(len(grid)):
    for c in range(len(grid[0])):
      size = explore_size(grid, r, c, visited)
      if size > 0 and size < min_size:
        min_size = size
  return min_size

def explore_size(grid, r, c, visited):
  row_inbounds = 0 <= r < len(grid)
  col_inbounds = 0 <= c < len(grid[0])
  if not row_inbounds or not col_inbounds:
    return 0
  
  if grid[r][c] == 'W':
    return 0
  
  pos = (r, c)
  if pos in visited:
    return 0
  visited.add(pos)
  
  size = 1
  size += explore_size(grid, r - 1, c, visited)
  size += explore_size(grid, r + 1, c, visited)  
  size += explore_size(grid, r, c - 1, visited)
  size += explore_size(grid, r, c + 1, visited)
  return size

# r = number of rows
# c = number of columns
# Time: O(rc)
# Space: O(rc)


####################
## Closest Carrot ##
####################

from collections import deque

def closest_carrot(grid, starting_row, starting_col):
  visited = set([ (starting_row, starting_col) ])
  queue = deque([ (starting_row, starting_col, 0) ])
  while queue:
    row, col, distance = queue.popleft()
    
    if grid[row][col] == 'C':
      return distance
    
    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for delta in deltas:
      delta_row, delta_col = delta
      neighbor_row = row + delta_row
      neighbor_col = col + delta_col      
      pos = (neighbor_row, neighbor_col)
      row_inbounds = 0 <= neighbor_row < len(grid)
      col_inbounds = 0 <= neighbor_col < len(grid[0])
      if row_inbounds and col_inbounds and pos not in visited and grid[neighbor_row][neighbor_col] != 'X':
        visited.add(pos)
        queue.append((neighbor_row, neighbor_col, distance + 1))
        
  return -1

# r = number of rows
# c = number of columns
# Time: O(rc)
# Space: O(rc)


##################
## Longest Path ##
##################

def longest_path(graph):
  distance = {}
  for node in graph:
    if len(graph[node]) == 0:
      distance[node] = 0
      
  for node in graph:
    traverse_distance(graph, node, distance)
    
  return max(distance.values())

def traverse_distance(graph, node, distance):
  if node in distance:
    return distance[node]
  
  largest = 0
  for neighbor in graph[node]:
    attempt = traverse_distance(graph, neighbor, distance)
    if attempt > largest:
      largest = attempt
  
  distance[node] = 1 + largest
  return distance[node]

# e = # edges
# n = # nodes
# Time: O(e)
# Space: O(n)


########################
## Semesters Required ##
########################

def semesters_required(num_courses, prereqs):
  graph = build_graph(num_courses, prereqs)
  distance = {}
  for course in range(num_courses):
    if len(graph[course]) == 0:
      distance[course] = 1
  
  for course in range(num_courses):
    traverse_distance(graph, course, distance)
    
  return max(distance.values())

def traverse_distance(graph, node, distance):
  if node in distance:
    return distance[node]
  
  max_distance = 0
  for neighbor in graph[node]:
    neighbor_distance = traverse_distance(graph, neighbor, distance)
    if neighbor_distance > max_distance:
      max_distance = neighbor_distance
    
  distance[node] = 1 + max_distance
  return distance[node]

def build_graph(num_courses, prereqs):
  graph = {}

  for course in range(num_courses):
    graph[course] = []
  
  for prereq in prereqs:
    a, b = prereq
    graph[a].append(b)
  
  return graph

# p = # prereqs
# c = # courses
# Time: O(p)
# Space: O(c)


#################
## Best Bridge ##
#################

from collections import deque

def best_bridge(grid):
  main_island = None
  for r in range(len(grid)):
    for c in range(len(grid[0])):
      potential_island = traverse_island(grid, r, c, set())
      if len(potential_island) > 0:
        main_island = potential_island
  
  visited = set(main_island)
  queue = deque([ ])
  for pos in main_island:
    r, c = pos
    queue.append((r, c, 0))
  
  while queue:
    row, col, distance = queue.popleft()
    if grid[row][col] == 'L' and (row, col) not in main_island:    
      return distance - 1
    
    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for delta in deltas:
      delta_row, delta_col = delta
      neighbor_row = row + delta_row
      neighbor_col = col + delta_col
      neighbor_pos = (neighbor_row, neighbor_col)
      if inbounds(grid, neighbor_row, neighbor_col) and neighbor_pos not in visited:
        visited.add(neighbor_pos)
        queue.append((neighbor_row, neighbor_col, distance + 1))
  
def inbounds(grid, row, col):
  row_inbounds = 0 <= row < len(grid)
  col_inbounds = 0 <= col < len(grid[0])
  return row_inbounds and col_inbounds

def traverse_island(grid, row, col, visited):
  if not inbounds(grid, row, col) or grid[row][col] == 'W':
    return visited
  
  pos = (row, col)
  if pos in visited:
    return visited
  
  visited.add(pos)
  
  traverse_island(grid, row - 1, col, visited)
  traverse_island(grid, row + 1, col, visited)
  traverse_island(grid, row, col - 1, visited)
  traverse_island(grid, row, col + 1, visited)
  return visited

# r = number of rows
# c = number of columns
# Time: O(rc)
# Space: O(rc)


###############
## Has Cycle ##
###############

# White Grey Black Algo

# def has_cycle(graph):
#   visiting = set()
#   visited = set()

#   for node in graph:
#   	if cycle_detect(graph, node, visiting, visited) == True:
#   		return True
    
#   return False

# def cycle_detect(graph, node, visiting, visited):
#   if node in visited:
#   	return False

#   if node in visiting:
#   	return True

#   visiting.add(node)

#   for neighbor in graph[node]:
#     if cycle_detect(graph, neighbor, visiting, visited) == True:
#     	return True

#   visiting.remove(node)
#   visited.add(node)

#   return False

# n = number of nodes
# Time: O(n^2)
# Space: O(n)


######################
## Prereqs Possible ##
######################

def prereqs_possible(num_courses, prereqs):
  graph = build_graph(num_courses, prereqs)
  visiting = set()
  visited = set()
  
  for node in range(0, num_courses):
    if has_cycle(graph, node, visiting, visited):
      return False
    
  return True
  
def has_cycle(graph, node, visiting, visited):
  if node in visited:
    return False
  
  if node in visiting:
    return True
  
  visiting.add(node)
  
  for neighbor in graph[node]:
    if has_cycle(graph, neighbor, visiting, visited):
      return True
  
  visiting.remove(node)
  visited.add(node)
  
  return False


def build_graph(num_courses, prereqs):
  graph = {}
  
  for i in range(0, num_courses):
    graph[i] = []
    
  for prereq in prereqs:
    a, b = prereq
    graph[a].append(b)
    
  return graph

# p = # prereqs
# n = # courses
# Time: O(n + p)
# Space: O(n)


###################
## Knight Attack ##
###################

from collections import deque

def knight_attack(n, kr, kc, pr, pc):
  visited = set();
  visited.add((kr, kc))
  queue = deque([ (kr, kc, 0) ])
  while len(queue) > 0:
    r, c, step = queue.popleft();
    if (r, c) == (pr, pc):
      return step
    neighbors = get_knight_moves(n, r, c)
    for neighbor in neighbors:
      neighbor_row, neighbor_col = neighbor
      if neighbor not in visited:
        visited.add(neighbor)
        queue.append((neighbor_row, neighbor_col, step + 1))
  return None

def get_knight_moves(n, r, c):
  positions = [
    ( r + 2, c + 1 ),
    ( r - 2, c + 1 ),
    ( r + 2, c - 1 ),
    ( r - 2, c - 1 ),
    ( r + 1, c + 2 ),
    ( r - 1, c + 2 ),
    ( r + 1, c - 2 ),
    ( r - 1, c - 2 ),
  ]
  inbounds_positions = [];
  for pos in positions:
    new_row, new_col = pos
    if 0 <= new_row < n and 0 <= new_col < n:
      inbounds_positions.append(pos)
  return inbounds_positions


##################
## Pair Product ##
##################

def pair_product(numbers, target_product):
  previous_nums = {}
  
  for index, num in enumerate(numbers):
    complement = target_product / num
    
    if complement in previous_nums:
      return (index, previous_nums[complement])
    
    previous_nums[num] = index

# n = length of numbers list
# Time: O(n)
# Space: O(n)


##################
## Intersection ##
##################

def intersection(a, b):
  set_a = set(a)
  return [ item for item in b if item in set_a ]

# n = length of array a, m = length of array b
# Time: O(n+m)
# Space: O(n)


###############
## Five Sort ##
###############

def five_sort(nums):
 i = 0
 j = len(nums) - 1
 while i < j:
  if nums[j] == 5:
   j -= 1
  elif nums[i] == 5:
   nums[i], nums[j] = nums[j], nums[i]
   i += 1
  else:
   i += 1
 return nums

# n = array size
# Time: O(n)
# Space: O(1)


##############
## Is Prime ##
##############

from math import sqrt, floor

def is_prime(n):
  if n < 2:
    return False
  
  for i in range(2, floor(sqrt(n)) + 1):
    if n % i == 0:
      return False
    
  return True

# n = input number
# Time: O(square_root(n))
# Space: O(1)

########################
## Contains Duplicate ##
########################

def containsDuplicate(nums):
    hashset = set()
    
    for n in nums:
        if n in hashset:
            return True
        hashset.add(n)
    return False

containsDuplicate([1,2,3,1])

# n = input array
# Time: O(n)
# Space: O(1)


########################
## Linked List Values ##
########################

# class Node:
#   def __init__(self, val):
#     self.val = val
#     self.next = None

def linked_list_values(head):
  current = head
  values = []
  
  if head is None:
    return
  values.append(current.val)
  linked_list_values(current.next)
  
  return values

# n = number of nodes
# Time: O(n)
# Space: O(n)


###################
## Valid Anagram ##
###################

class Solution:
  def isAnagram(self, s: str, t: str) -> bool:
    if len(s) != len(t):
      return False

    # initialize hashmap
    hmap_s = {}
    hmap_t = {}

    # add string characters to hashmap
    for i in range(len(s)):
      hmap_s[s[i]] = 1 + hmap_s.get(s[i], 0)
      hmap_t[t[i]] = 1 + hmap_t.get(t[i], 0)

    if hmap_s == hmap_t:
      return True

# n = number of char 
# Time: O(n)
# Space: O(n)


#############
## Two Sum ##
#############

def twoSum(nums, target):
  prev_map = {}

  for i, num in enumerate(nums, 0):
    pair = target - num
    print(pair)

    if pair in prev_map:
      return (prev_map[pair], i)
    
    prev_map[num] = i

# n = number of elements 
# Time: O(n)
# Space: O(n)


############################
## Concatenation of Array ##
############################

def getConcatenation(nums):
  newArr = []
  
  for i in range(2):
    for num in nums:
        newArr.append(num)
  return newArr

# n = length of array
# Time: O(n)
# Space: O(n)


####################
## Is Subsequence ##
####################

class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
      i = 0
      j = 0

      if len(s) == i:
          return True

      while i < len(s) and j < len(t):
          if s[i] == t[j]:
              i += 1
          j += 1
          
          if i == len(s):
              return True
          else:
              if j == len(t):
                  return False

# n = length s + length t
# Time: O(n)
# Space: O(1)


#########################
## Length of Last Word ##
#########################

class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        i = len(s) - 1
        l = 0
        
        while s[i] == ' ':
            i -= 1
        while i >= 0 and s[i] != ' ':
            l += 1
            i -= 1
        return l

# n = length input string
# Time: O(n)
# Space: O(1)


############################################
## Replace Elements with Greatest Element ##
############################################

def replaceElements(arr):
    
    currMax = -1

    for i in range(len(arr) -1, -1, -1):
        newMax = max(currMax, arr[i])
        arr[i] = currMax
        currMax = newMax
    return arr

# n = length of array
# Time: O(n)
# Space: O(1)


###########################
## Longest Common Prefix ##
###########################

class Solution:
    def longestCommonPrefix(strs):
        res = ""

        for i in range(len(strs[0])):
            for s in strs:
                if i == len(s) or s[i] != strs[0][i]:
                    return res
            res += strs[0][i]
        return res
    
# n = length of array
# Time: O(n)
# Space: O(1)


####################
## Remove Element ##
####################

def removeElement(nums, val):
    k = 0 

    for i in range(len(nums)):
        if nums[i] != val:
            nums[k] = nums[i]
            k += 1
    return k


# Time: O(n)
# Space: O(1)



########################
## Isomorphic Strings ##
########################

def isIsomorphic(s, t):
    mapS = {}
    mapT = {}

    for i in range(len(s)):
        c1 = s[i]
        c2 = t[i]

        if (c1 in mapS and mapS[c1] != c2) or (c2 in mapT and mapT[c2] != c1):
            return False
        
        mapS[c1] = c2
        mapT[c2] = c1
    return True

# n = length of strings
# Time: O(n)
# Space: O(1)


######################
## Majority Element ##
######################

def majorityElement(nums):
  count = {}
  res = 0
  max_count = 0

  for n in nums:
    count[n] = 1 + count.get(n, 0)
    res = n if count[n] > max_count else res
    max_count = max(count[n], max_count)
  return res

# n = length of strings
# Time: O(n)
# Space: O(n)


#######################################
## Find the Difference of Two Arrays ##
#######################################

def findDifference(nums1, nums2):
    nums_set1 = set(nums1)
    nums_set2 = set(nums2)
    result1 = set()
    result2 = set()

    for n in nums1:
        if n not in nums_set2:
            result1.add(n)

    for n in nums2:
        if n not in nums_set1:
            result2.add(n)
    
    return [list(result1), list(result2)]

# n = length of array
# Time: O(n + m)
# Space: O(n + m)

####################
## Weekend Review ##
####################

def twoSum(nums, t):
    hashmap = {}

    for i, num in enumerate(nums, 0):
        comp = t - num 
    
        if comp in hashmap:
            # return index of both nums in hashmap
            return (hashmap[comp], i)
        # add to hashmap
        hashmap[num] = i
        
print(twoSum([2,7,11,15], 9))

def longest_common_prefix(arr):
    tracker = ''

    # for the length of flower do this 
    for i in range(len(arr[0])):
        # for each index/string in the array do this
        for eachstring in arr:
            # if the char in each string is not equal to the char in flower return tracker
            if i == len(eachstring) or eachstring[i] != arr[0][i]:
                return tracker
        tracker += arr[0][i]

    return tracker
        

######################
## Pascals Triangle ##
######################

def generate(numRows):
    res = []

    for i in range(numRows - 1):
        temp = [0] + res[-1] + [0]
        row = []
        for j in range(len(res[-1]) + 1):
            row.append(temp[j] + temp[j + 1])
        res.append(row)
    return res

# n = number of rows
# Time: O(n^2)
# Space: O(n^2)


######################
## Find Pivot Index ##
######################

def pivotIndex(nums):
    total = sum(nums)

    leftSum = 0
    for i in range(len(nums)):
        rightSum = total - nums[i] - leftSum
        if leftSum == rightSum:
            return i
        leftSum += nums[i]
    return -1


#####################################
## Sign of the Product of an Array ##
#####################################

def arraySign(nums):
    neg = 0

    for n in nums:
        if n == 0:
            return 0
        neg += (1 if n < 0 else 0)

    return -1 if neg % 2 else 1

# n = length of array
# Time: O(n)
# Space: O(n)


####################
## Design HashSet ##
####################

class ListNode:
    def __init__(self, key):
        self.key = key
        self.next = None

class MyHashSet:
    def __init__(self):
        self.set = [ListNode(0) for i in range(10**4)]
    
    def add(self, key: int) -> None:
        cur = self.set[key % len(self.set)]
        while cur.next:
            if cur.next.key == key:
                return
            cur = cur.next
        cur.next = ListNode(key)

    def remove(self, key: int) -> None:
        cur = self.set[key % len(self.set)]
        while cur.next:
            if cur.next.key == key:
                cur.next = cur.next.next
                return
            cur = cur.next
    
    def contains(self, key: int) -> bool:
        cur = self.set[key % len(self.set)]
        while cur.next:
            if cur.next.key == key:
                return True
            cur = cur.next
        return False
    

###########################
## Design Parking System ##
###########################

class ParkingSystem:

  def __init__(self, big: int, medium: int, small: int):
      self.spaces = [big, medium, small]

  def addCar(self, carType: int) -> bool:
      if self.spaces[carType - 1] > 0:
          self.spaces[carType - 1] -= 1
          return True
      return False
  

##################
## Word Pattern ##
##################

  class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        words = s.split(" ")
        if len(pattern) != len(words):
            return False
        charToWord = {}
        wordToChar = {}
        
        for c, w in zip(pattern, words):
            if c in charToWord and charToWord[c] != w:
                return False
            if w in wordToChar and wordToChar[w] != c:
                return False
            charToWord[c] = w
            wordToChar[w] = c
        return True

# n + m = length of string
# Time: O(n + m)
# Space: O(n + m)


############################
## Unique Email Addresses ##
############################

class Solution:
    def numUniqueEmails(self, emails: list[str]) -> int:
        unique_emails: set[str] = set()
        for email in emails:
            local_name, domain_name = email.split('@')
            local_name = local_name.split('+')[0]
            local_name = local_name.replace('.', '')
            email = local_name + '@' + domain_name
            unique_emails.add(email)
        return len(unique_emails)
    
  
####################
## Design HashMap ##
####################

class ListNode:
    def __init__(self, key=-1, val=-1, next=None):
        self.key = key
        self.val = val
        self.next = next

class MyHashMap:
    def __init__(self):
        self.map = [ListNode() for i in range(1000)]
        
    def hashcode(self, key):
        return key % len(self.map)

    def put(self, key: int, value: int) -> None:
        cur = self.map[self.hashcode(key)]
        while cur.next:
            if cur.next.key == key:
                cur.next.val = value
                return
            cur = cur.next
        cur.next = ListNode(key, value)
         
    def get(self, key: int) -> int:
        cur = self.map[self.hashcode(key)].next
        while cur and cur.key != key:
            cur = cur.next
        if cur:
            return cur.val
        return -1

    def remove(self, key: int) -> None:
        cur = self.map[self.hashcode(key)]
        while cur.next and cur.next.key != key:
            cur = cur.next
        if cur and cur.next:
            cur.next = cur.next.next


####################
## Group Anagrams ##
####################

# class Solution:
#     def groupAnagrams(strs):
#         # res = defaultdict(list)

#         for s in strs: 
#             count = [0] * 26 # a ... z

#             for c in s:
#                 count[ord(c) - ord("a")] += 1

#             res[tuple(count)].append(s)
        
#         return res.values()

# m = number strings given n = length of each string
# Time: O(n * m)
# Space: O(n * m)

#############################
##  Next Greater Element I ##
#############################

class Solution:
    def nextGreaterElement(nums1, nums2):

        nums1Idx = { n:i for i, n in enumerate(nums1) }
        res = [-1] * len(nums1)

        stack = []
        for i in range(len(nums2)):
            cur = nums2[i]

            while stack and cur > stack[-1]:
                val = stack.pop() 
                idx = nums1Idx[val]
                res[idx] = cur

            if cur in nums1Idx:
                stack.append(cur)
        
        return res


# m = size of array
# Time: O(m + n)
# Space: O(m)


################################################
##  Find All Numbers Disappeared in an Array  ##
################################################

class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        for n in nums:
            i = abs(n) - 1
            nums[i] = -1 * abs(nums[i])

        res = []
        for i, n in enumerate(nums):
            if n > 0:
                res.append(i + 1)
        return res


###################################
##  Range Sum Query - Immutable  ##
###################################



############
## Review ##
############

def twoSum(nums, target):
    hashmap = {}
    for idx, num in enumerate(nums, 0):
        complement =  target - num
        if complement in hashmap:
            return (hashmap[complement], idx)
        hashmap[num] = idx
        

print(twoSum([3,3], 6))

def longestCommonPrefix(arr):
    pref = ''

    for idx in range(len(arr[0])):
        for el in arr:
        #      num         num
            if idx == len(el):
                return pref
            if el[idx] != arr[0][idx]:
                return pref
        pref += el[idx]

print(longestCommonPrefix(["dog","dacecar","dar"]))

class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        k = 0 

        for i in range(len(nums)):
            if nums[i] != val:
                nums[k] = nums[i]
                k += 1
        return k