"""
该文件定义了不同搜索算法的具体实现，包括搜索算法基类（BaseSearch）、广度优先搜索（BreadthFirstSearch）、
深度优先搜索（DepthFirstSearch）、迭代加深搜索（IterativeDeepeningDepthFirstSearch）、
一致代价搜索（UniformCostSearch）和A*搜索（AStarSearch）
"""
from frontiers import BaseFrontier, StackFrontier, QueueFrontier, PriorityQueueFrontier


class BaseSearch:
    """搜索算法基类，定义了初始化、算法框架、结果输出等通用操作"""
    def __init__(self, maze_file):
        """
        初始化搜索类所需的各个变量，并抽象迷宫问题为Python可处理的数据类型
        :param maze_file: 迷宫问题的文件路径
        """
        self.construct_maze(maze_file)  # 建立迷宫，并设置初始状态和目标状态
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 按照上下左右的顺序搜索
        self.actions_costs = [2, 1, 3, 5]  # 上下左右移动一步分别对应的代价
        self.frontier = BaseFrontier(self.init_node)  # 前沿集合中仅含有初始节点
        self.explored = []  # 已探索集合为空
        self.came_from = {self.init_node: None}  # 记录父节点
        self.path_cost = {self.init_node: 0}  # 记录路径代价
        self.path = []  # 在找到解后用于存储具体路径
        self.find_solution = False  # 记录是否找到迷宫问题的解

    def construct_maze(self, maze_file):
        """
        根据字符串建立迷宫，例如：
        输入的迷宫文本文件中包含
        B10B
        0A10
        B00B
        则建立迷宫的列表表示为
        self.maze = [[0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 0]]
        self.maze_row = 3
        self.maze_col = 4
        self.init_node = (0, 0)
        self.goal_node = [(0, 0), (0, 3), (2, 0), (2, 3)]
        :param maze_file: 迷宫问题的文件路径
        """
        with open(maze_file) as f:
            text_maze = f.read()
        text_maze = text_maze.splitlines()
        self.maze_row = len(text_maze)  # 迷宫的行数
        self.maze_col = len(text_maze[0])  # 迷宫的列数
        self.maze = []
        self.goal_node = []
        for i in range(self.maze_row):  # 按行构建迷宫
            row = []
            for j in range(self.maze_col):
                if text_maze[i][j] == 'A':  # A的位置为初始节点
                    self.init_node = (i, j)
                    row.append(0)
                elif text_maze[i][j] == 'B':  # B的位置为目标节点
                    self.goal_node.append((i, j))
                    row.append(0)
                elif text_maze[i][j] == '0':  # 0代表可通行方格
                    row.append(0)
                else:  # 否则为障碍物
                    row.append(1)
            self.maze.append(row)

    def goal_test(self, node):
        """
        测试是否到达目标节点
        :param node: 需要进行测试的节点
        """
        return node in self.goal_node

    def solve(self):
        """使用具体的搜索算法解决迷宫问题"""
        while not self.frontier.empty():  # 若前沿集合为空，搜索失败，终止算法
            node = self.frontier.pop()  # 从前沿集合中取出一个节点n
            if self.goal_test(node):  # 若节点n包含目标状态，搜索成功，返回方案
                self.find_solution = True
                self.final_goal_node = node
                return
            else:
                self.explored.append(node)
                new_nodes, new_nodes_costs = self.expand(node)
                for new_node, new_node_cost in zip(new_nodes, new_nodes_costs):
                    self.process_new_node(node, new_node, new_node_cost)

    def expand(self, node):
        """
        扩展当前节点，得到其可达节点集合
        :param node: 当前节点
        """
        new_nodes = []
        new_nodes_costs = []  # 代表从node移动一步到new_node的代价
        for action, action_cost in zip(self.actions, self.actions_costs):  # 按顺序执行动作，获取新节点的位置
            new_row = node[0] + action[0]
            new_col = node[1] + action[1]
            new_node_cost = action_cost
            if self.check_row_col(new_row, new_col):  # 如果位置合法则加入新节点集合
                new_node = (new_row, new_col)
                new_nodes.append(new_node)
                new_nodes_costs.append(action_cost)

        return new_nodes, new_nodes_costs  # 返回可达新节点与从node到这些节点对应的一步代价

    def check_row_col(self, new_row, new_col):
        """
        检查位置合法性
        :param new_row: 行坐标
        :param new_col: 列坐标
        """
        if new_row < 0 or new_row >= self.maze_row:  # 检查行坐标合法性
            return False
        if new_col < 0 or new_col >= self.maze_col:  # 检查列坐标合法性
            return False
        if self.maze[new_row][new_col] == 1:  # 检查该位置是否为障碍物
            return False
        return True

    def process_new_node(self, node, new_node, new_node_cost):
        """
        处理新节点，包括判断是否放入前沿集合、记录其父节点和路径代价
        :param node: 当前节点
        :param new_node: 新节点
        :param new_node_cost: 从当前节点移动到新节点的代价
        """
        if new_node not in self.explored and not self.frontier.contains_node(new_node):
            self.frontier.push(new_node)
            self.came_from[new_node] = node
            self.path_cost[new_node] = self.path_cost[node] + new_node_cost

    def output(self):
        """输出求解结果"""
        if self.find_solution:
            print('Find solution! Total cost: {}.'.format(self.path_cost[self.final_goal_node]))  # 找到可行路径并输出其路径代价
            self.backtrack()  # 从目标节点开始回溯路径上的节点
            self.print_solution()  # 输出具体路径
            self.draw_solution()  # 可视化迷宫问题的解
        else:
            print('No solution!')  # 未找到可行路径

    def backtrack(self):
        """从目标节点开始回溯路径上的节点"""
        n = self.final_goal_node
        while n is not None:  # 注意初始节点的父节点定义为None
            self.path.append(n)  # 将节点加入路径中
            n = self.came_from[n]  # 向父节点回溯
        self.path.reverse()  # 反转路径得到从初始节点到目标节点的顺序

    def print_solution(self):
        """输出具体路径"""
        for i in range(len(self.path) - 1):
            print('({}, {}) -> '.format(self.path[i][0], self.path[i][1]), end='')
        print('({}, {})'.format(self.path[-1][0], self.path[-1][1]))

    def draw_solution(self):
        """可视化迷宫问题的解"""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(self.maze_row, self.maze_col))
        for i in range(self.maze_row):
            for j in range(self.maze_col):
                # 用不同颜色代表不同的迷宫方块含义
                if (i, j) in self.path:
                    color = (0.886, 0.941, 0.851)  # 绿色
                elif self.maze[i][j] == 0:
                    color = 'white'
                else:
                    color = 'black'
                if (i, j) == self.init_node:
                    color = (0.957, 0.694, 0.514)  # 橙色
                elif (i, j) in self.goal_node:
                    color = (0.561, 0.667, 0.863)  # 蓝色
                # 画出方块
                rec = plt.Rectangle((j, self.maze_row - i), width=1, height=1, facecolor=color, edgecolor='black')
                ax.add_patch(rec)
        # 添加A、B注释
        plt.text(x=self.init_node[1] + 0.3, y=self.maze_row - self.init_node[0] + 0.3, s='A', fontsize=35)
        for goal_node in self.goal_node:
            plt.text(x=goal_node[1] + 0.3, y=self.maze_row - goal_node[0] + 0.3, s='B', fontsize=35)
        plt.axis('scaled')
        plt.axis('off')
        plt.show()


class BreadthFirstSearch(BaseSearch):
    """广度优先搜索"""
    def __init__(self, maze_file):
        """
        和BaseSearch基类相比使用队列作为前沿集合，因此需要重新定义初始化操作
        :param maze_file: 迷宫问题的文件路径
        """
        super(BreadthFirstSearch, self).__init__(maze_file)
        self.frontier = QueueFrontier(self.init_node)


class DepthFirstSearch(BaseSearch):
    """深度优先搜索"""
    def __init__(self, maze_file):
        """
        和BaseSearch基类相比使用栈作为前沿集合，因此需要重新定义初始化操作
        :param maze_file: 迷宫问题的文件路径
        """
        super(DepthFirstSearch, self).__init__(maze_file)
        self.frontier = StackFrontier(self.init_node)


class IterativeDeepeningDepthFirstSearch(BaseSearch):
    """迭代加深搜索"""
    def reset(self):
        """由于每次增加深度限制后都要重新进行深度优先搜索，所以要把一些变量重新初始化"""
        self.frontier = StackFrontier(self.init_node)  # 重置前沿集合，仅含初始节点
        self.explored = []  # 重置已探索集合，为空
        self.came_from = {self.init_node: None}  # 重置记录父节点的字典
        self.path_cost = {self.init_node: 0}  # 重置记录路径代价的字典
        self.depth = {self.init_node: 0}  # 记录当前节点的层级（深度）

    def process_new_node(self, node, new_node, new_node_cost):
        """
        处理新节点，包括判断是否放入前沿集合、记录其父节点和路径代价
        :param node: 当前节点
        :param new_node: 新节点
        :param new_node_cost: 从当前节点移动到新节点的代价
        """
        if new_node not in self.explored and not self.frontier.contains_node(new_node):
            self.frontier.push(new_node)
            self.came_from[new_node] = node
            self.path_cost[new_node] = self.path_cost[node] + new_node_cost
            self.depth[new_node] = self.depth[node] + 1

    def plain_solve(self, depth_limit):
        """
        内层循环：有深度限制的深度优先搜索
        :param depth_limit: 深度限制
        """
        while not self.frontier.empty():
            node = self.frontier.pop()
            if self.goal_test(node):
                self.find_solution = True
                self.final_goal_node = node
                return
            elif self.depth[node] < depth_limit:  # 达到深度限制后不再扩展节点
                self.explored.append(node)
                new_nodes, new_nodes_costs = self.expand(node)
                for new_node, new_node_cost in zip(new_nodes, new_nodes_costs):
                    self.process_new_node(node, new_node, new_node_cost)

    def solve(self):
        """外层循环：逐渐增大深度限制搜索"""
        for depth_limit in range(self.maze_row * self.maze_col):
            self.reset()  # 重置初始条件
            self.plain_solve(depth_limit)  # 给定深度限制下的深度优先搜索
            if self.find_solution:  # 找到可行路径后及时退出循环
                return


class UniformCostSearch(BaseSearch):
    """一致代价搜索"""
    def __init__(self, maze_file):
        """
        和BaseSearch基类相比使用优先队列作为前沿集合，因此需要重新定义初始化操作
        :param maze_file: 迷宫问题的文件路径
        """
        super(UniformCostSearch, self).__init__(maze_file)
        self.frontier = PriorityQueueFrontier(self.init_node)

    def process_new_node(self, node, new_node, new_node_cost):
        """
        处理新节点，一致代价搜索的处理逻辑和通用的搜索流程不同，还需考虑路径代价，因此需要重新定义函数
        :param node: 当前节点
        :param new_node: 新节点
        :param new_node_cost: 从当前节点移动到新节点的代价
        """
        if new_node not in self.explored:
            if not self.frontier.contains_node(new_node):
                self.frontier.push(new_node, self.path_cost[node] + new_node_cost)
                self.came_from[new_node] = node
                self.path_cost[new_node] = self.path_cost[node] + new_node_cost
            elif self.path_cost[node] + new_node_cost < self.path_cost[new_node]:
                self.frontier.update(new_node, self.path_cost[node] + new_node_cost)
                self.came_from[new_node] = node
                self.path_cost[new_node] = self.path_cost[node] + new_node_cost


class AStarSearch(BaseSearch):
    """A*搜索"""
    def __init__(self, maze_file):
        """
        和BaseSearch基类相比使用优先队列作为前沿集合，因此需要重新定义初始化操作
        :param maze_file: 迷宫问题的文件路径
        """
        super(AStarSearch, self).__init__(maze_file)
        self.frontier = PriorityQueueFrontier(self.init_node)
        self.astar_cost = {self.init_node: self.heuristic_func(self.init_node)}  # 即课件中的f(n)=g(n)+h(n)

    def process_new_node(self, node, new_node, new_node_cost):
        """
        处理新节点，一致代价搜索的处理逻辑和通用的搜索流程不同，还需考虑路径代价，因此需要重新定义函数
        :param node: 当前节点
        :param new_node: 新节点
        :param new_node_cost: 从当前节点移动到新节点的代价
        """
        if new_node not in self.explored:
            new_astar_cost = self.path_cost[node] + new_node_cost + self.heuristic_func(new_node)
            if not self.frontier.contains_node(new_node):
                self.frontier.push(new_node, new_astar_cost)
                self.came_from[new_node] = node
                self.path_cost[new_node] = self.path_cost[node] + new_node_cost
                self.astar_cost[new_node] = new_astar_cost
            elif new_astar_cost < self.astar_cost[new_node]:
                self.frontier.update(new_node, new_astar_cost)
                self.came_from[new_node] = node
                self.path_cost[new_node] = self.path_cost[node] + new_node_cost
                self.astar_cost[new_node] = new_astar_cost

    def heuristic_func(self, node):
        """
        :param node: 当前节点
        :return: 使用启发函数计算从当前节点到目标节点的代价
        """
        min_distance = float('inf')
        for goal in self.goal_node:
            distance = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
            if distance < min_distance:
                min_distance = distance
        return min_distance
