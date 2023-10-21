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