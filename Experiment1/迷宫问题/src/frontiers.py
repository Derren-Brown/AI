"""
该文件定义了不同算法所需的前沿集合，包括前沿集合基类（BaseFrontier）、
栈（StackFrontier）、队列（QueueFrontier）和优先队列（PriorityFrontier）
"""


class BaseFrontier:
    """前沿集合基类，定义了初始化、判断是否为空、判断是否含有节点等通用操作"""
    def __init__(self, init_node):
        """
        使用初始节点来初始化前沿集合，前沿集合由列表数据类型实现
        :param init_node: 初始节点
        """
        self.frontier = [init_node]

    def empty(self):
        """判断前沿集合是否为空"""
        return len(self.frontier) == 0

    def contains_node(self, node):
        """判断某个节点是否在前沿集合中"""
        return node in self.frontier

    def push(self, *args):
        """将节点放入前沿集合，不同搜索算法的实现方式不同"""
        raise NotImplementedError

    def pop(self):
        """从前沿集合中取出节点，不同搜索算法的实现方式不同"""
        raise NotImplementedError


class StackFrontier(BaseFrontier):
    """由栈构成的前沿集合"""
    def push(self, node):
        """
        将新节点添加到栈顶（列表末端）
        :param node: 需要添加进前沿集合的新节点
        """
        self.frontier.append(node)

    def pop(self):
        """从栈顶（列表末端）取出节点，并从前沿集合中删除该节点"""
        node = self.frontier[-1]
        self.frontier = self.frontier[:-1]
        return node


class QueueFrontier(BaseFrontier):
    """由队列构成的前沿集合"""
    def push(self, node):
        """
        将新节点添加到队尾（列表末端）
        :param node: 需要添加进前沿集合的新节点
        """
        self.frontier.append(node)

    def pop(self):
        """从队首（列表前端）取出节点，并从前沿集合中删除该节点"""
        node = self.frontier[0]
        self.frontier = self.frontier[1:]
        return node


class PriorityQueueFrontier(BaseFrontier):
    """优先队列构成的前沿集合"""
    def __init__(self, init_node):
        """
        和BaseFrontier基类相比还需要一个列表来维护每个节点的路径代价，因此需要重新定义初始化操作
        :param init_node: 初始节点
        """
        super(PriorityQueueFrontier, self).__init__(init_node)
        self.frontier_cost = [0]  # 从初始节点到初始节点的路径代价为0

    def swap(self, idx1, idx2):
        """
        同步交换节点列表（self.frontier）和路径代价列表（self.frontier_cost）中的两个元素
        :param idx1: 元素1的下标
        :param idx2: 元素2的下标
        """
        tmp = self.frontier[idx1]
        self.frontier[idx1] = self.frontier[idx2]
        self.frontier[idx2] = tmp

        tmp = self.frontier_cost[idx1]
        self.frontier_cost[idx1] = self.frontier_cost[idx2]
        self.frontier_cost[idx2] = tmp

    def push(self, node, cost):
        """
        将新节点添加到优先队列中（自底向上构造小顶堆）
        :param node: 需要添加进前沿集合的新节点
        :param cost: 该节点对应的路径代价
        """
        if not self.contains_node(node):  # 如果该节点不在前沿集合中，则添加到列表末端。并作为该节点的起始位置
            self.frontier.append(node)
            self.frontier_cost.append(cost)
            child_idx = len(self.frontier) - 1
        else:  # 否则找到该节点的位置
            for child_idx in range(len(self.frontier)):
                if self.frontier[child_idx] == node:
                    break
        parent_idx = int((child_idx - 1) / 2)  # 使用parent和child命名下标为二叉树中父子节点的概念，在小顶堆中父节点的值必须小于子节点的值
        while self.frontier_cost[parent_idx] > cost:  # 将新节点自底向上移动直到父节点的路径代价比该节点路径代价小
            self.swap(parent_idx, child_idx)
            if parent_idx == 0:  # 说明已经移动到二叉树最顶端了
                break
            child_idx = parent_idx
            parent_idx = int((parent_idx - 1) / 2)

    def pop(self):
        """从队首中取出节点，并从前沿集合中删除该节点"""
        self.swap(0, len(self.frontier) - 1)  # 交换队首和队尾节点，这样前沿集合中路径代价最小的节点移动至队尾，是我们需要取出的节点
        node = self.frontier[-1]  # 从队尾取出上述节点
        self.frontier = self.frontier[:-1]  # 从节点列表中删除该节点
        self.frontier_cost = self.frontier_cost[:-1]  # 从路径代价列表中删除该节点的代价
        # 接下来需要把刚刚交换至队首的节点自顶向下调整到合适位置
        parent_idx = 0
        child_idx = 2 * parent_idx + 1
        while child_idx < len(self.frontier):
            if child_idx + 1 < len(self.frontier) \
                    and self.frontier_cost[child_idx + 1] < self.frontier_cost[child_idx]:  # 找到孩子节点中值更小的
                child_idx += 1
            if self.frontier_cost[child_idx] < self.frontier_cost[parent_idx]:  # 交换节点直到孩子节点的值都大于该节点
                self.swap(parent_idx, child_idx)
                parent_idx = child_idx
                child_idx = 2 * parent_idx + 1
            else:
                break
        return node
