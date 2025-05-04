import heapq
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np


class Node:
     
    def __init__(self, state, parent=None, action=None, cost=0):
       
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def __lt__(self, other):
        # Required for heap comparison (we only care about priority, not node content)
        return True


class Maze:
     

    def __init__(self, filename):
        
        self.grid = []
        self.start = None
        self.goal = None
        self.walls = []
        self.height = 0
        self.width = 0

        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                row = []
                for j, char in enumerate(line.strip()):
                    if char == 'A':
                        self.start = (i, j)
                        row.append(0)
                    elif char == 'B':
                        self.goal = (i, j)
                        row.append(0)
                    elif char == '#':
                        self.walls.append((i, j))
                        row.append(1)
                    else:
                        row.append(0)
                self.grid.append(row)

        if not self.start or not self.goal:
            raise ValueError("Maze must have both start (A) and goal (B) positions")

        self.height = len(self.grid)
        self.width = len(self.grid[0]) if self.height > 0 else 0

    def neighbors(self, state):
        
        row, col = state
        candidates = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]

        result = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and self.grid[r][c] != 1:
                result.append((action, (r, c)))
        return result

    def manhattan_distance(self, a, b):
        
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def solve(self, algorithm="astar"):
     
        if algorithm not in ["greedy", "astar"]:
            raise ValueError("Algorithm must be 'greedy' or 'astar'")

        frontier = []
        heapq.heappush(frontier, (0, Node(self.start)))
        explored = set()
        explored_states = []

        while frontier:
            _, node = heapq.heappop(frontier)

            if node.state == self.goal:
                path = []
                while node.parent is not None:
                    path.append(node.state)
                    node = node.parent
                path.reverse()
                return path, explored_states

            if node.state in explored:
                continue

            explored.add(node.state)
            explored_states.append(node.state)

            for action, state in self.neighbors(node.state):
                if state not in explored:
                    if algorithm == "greedy":
                        # Greedy Best-First uses only heuristic
                        priority = self.manhattan_distance(state, self.goal)
                    else:
                        # A* uses cost + heuristic
                        cost = node.cost + 1
                        priority = cost + self.manhattan_distance(state, self.goal)

                    heapq.heappush(frontier,
                                   (priority,
                                    Node(state, node, action, node.cost + 1)))

        raise ValueError("No path exists")


def visualize_maze(maze, path=None, explored=None, filename="maze_solution.png"):
    
    # Create blank image
    img = Image.new("RGB", (maze.width * 50, maze.height * 50), "white")
    draw = ImageDraw.Draw(img)

    # Draw walls
    for i in range(maze.height):
        for j in range(maze.width):
            if maze.grid[i][j] == 1:
                draw.rectangle([j * 50, i * 50, (j + 1) * 50, (i + 1) * 50], fill="black")

    # Draw explored states (if any)
    if explored:
        for state in explored:
            i, j = state
            draw.rectangle([j * 50, i * 50, (j + 1) * 50, (i + 1) * 50], fill="lightblue")

    # Draw solution path (if any)
    if path:
        for state in path:
            i, j = state
            draw.rectangle([j * 50, i * 50, (j + 1) * 50, (i + 1) * 50], fill="green")

    # Draw start and goal positions
    start_i, start_j = maze.start
    goal_i, goal_j = maze.goal
    draw.rectangle([start_j * 50, start_i * 50, (start_j + 1) * 50, (start_i + 1) * 50], fill="red")
    draw.rectangle([goal_j * 50, goal_i * 50, (goal_j + 1) * 50, (goal_i + 1) * 50], fill="blue")

    # Add grid lines
    for i in range(maze.height + 1):
        draw.line([(0, i * 50), (maze.width * 50, i * 50)], fill="gray", width=1)
    for j in range(maze.width + 1):
        draw.line([(j * 50, 0), (j * 50, maze.height * 50)], fill="gray", width=1)

    img.save(filename)
    return img


def create_sample_maze(filename="maze.txt"):
    
    maze_content = """###########
#A...#...B#
#.#.#.#.#.#
#.#...#...#
#.#####.#.#
#.......#.#
###########"""

    with open(filename, 'w') as f:
        f.write(maze_content)
    print(f"Sample maze created at {filename}")


def compare_algorithms(maze):
     
    print("\n=== Solving with Greedy Best-First Search ===")
    try:
        path_greedy, explored_greedy = maze.solve(algorithm="greedy")
        print(f"Path found with {len(path_greedy)} steps")
        print(f"Explored {len(explored_greedy)} states")
        visualize_maze(maze, path_greedy, explored_greedy, "maze_greedy.png")
    except ValueError as e:
        print(f"Greedy search failed: {e}")

    print("\n=== Solving with A* Search ===")
    try:
        path_astar, explored_astar = maze.solve(algorithm="astar")
        print(f"Path found with {len(path_astar)} steps")
        print(f"Explored {len(explored_astar)} states")
        visualize_maze(maze, path_astar, explored_astar, "maze_astar.png")
    except ValueError as e:
        print(f"A* search failed: {e}")

    # Display statistics
    if 'path_greedy' in locals() and 'path_astar' in locals():
        print("\n=== Comparison ===")
        print(f"Greedy path length: {len(path_greedy)}")
        print(f"A* path length: {len(path_astar)}")
        print(f"Greedy explored states: {len(explored_greedy)}")
        print(f"A* explored states: {len(explored_astar)}")


def main():
    # Create a sample maze if one doesn't exist
    maze_file = "maze.txt"
    create_sample_maze(maze_file)

    # Load and solve the maze
    try:
        maze = Maze(maze_file)
        print(f"Maze loaded: {maze.height}x{maze.width}")
        print(f"Start: {maze.start}, Goal: {maze.goal}")

        # Visualize the original maze
        visualize_maze(maze, filename="maze_original.png")
        print("Original maze saved as maze_original.png")

        # Compare both algorithms
        compare_algorithms(maze)

        # Display the images
        try:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(Image.open("maze_original.png"))
            plt.title("Original Maze")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(Image.open("maze_greedy.png"))
            plt.title("Greedy Best-First Solution")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(Image.open("maze_astar.png"))
            plt.title("A* Solution")
            plt.axis('off')

            plt.tight_layout()
            plt.show()
        except FileNotFoundError:
            print("Could not display images - check the output files directly")

    except FileNotFoundError:
        print(f"Error: Could not find maze file {maze_file}")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()