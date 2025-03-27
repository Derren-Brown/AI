"""
该文件为负责运行具体求解程序的代码，在命令行执行python main.py可得迷宫问题求解结果
"""
import argparse
from search import BreadthFirstSearch, DepthFirstSearch, IterativeDeepeningDepthFirstSearch, UniformCostSearch, AStarSearch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='search for maze problem')
    parser.add_argument('--search_type', default='astar', type=str, help='bfs/dfs/iddfs/ucs/astar')
    parser.add_argument('--maze_file', default='maze.txt', type=str, help='file path of the maze')
    args = parser.parse_args()
    if args.search_type == 'bfs':
        solver = BreadthFirstSearch(args.maze_file)  # 选择一种具体的搜索算法并实例化
    elif args.search_type == 'dfs':
        solver = DepthFirstSearch(args.maze_file)
    elif args.search_type == 'iddfs':
        solver = IterativeDeepeningDepthFirstSearch(args.maze_file)
    elif args.search_type == 'ucs':
        solver = UniformCostSearch(args.maze_file)
    elif args.search_type == 'astar':
        solver = AStarSearch(args.maze_file)
    solver.solve()  # 使用上述搜索算法求解迷宫问题
    solver.output()  # 如果有解，输出迷宫的具体路径和花费代价，并进行可视化

