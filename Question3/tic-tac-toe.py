import math
import copy
import pygame
import sys
from typing import List, Tuple, Optional, Set

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
LINE_WIDTH = 15
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4

# Colors
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)
TEXT_COLOR = (255, 255, 255)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe - Minimax AI')
screen.fill(BG_COLOR)

# Game board
board = [[None for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]


def initial_state() -> List[List[Optional[str]]]:
    """Return starting state of the board"""
    return [[None, None, None],
            [None, None, None],
            [None, None, None]]


def player(board: List[List[Optional[str]]]) -> str:
    """Returns which player's turn it is (X or O)"""
    x_count = sum(row.count('X') for row in board)
    o_count = sum(row.count('O') for row in board)
    return 'O' if x_count > o_count else 'X'


def actions(board: List[List[Optional[str]]]) -> Set[Tuple[int, int]]:
    """Returns set of all possible actions (i, j) available on the board"""
    possible_actions = set()
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if board[i][j] is None:
                possible_actions.add((i, j))
    return possible_actions


def result(board: List[List[Optional[str]]], action: Tuple[int, int]) -> List[List[Optional[str]]]:
    """Returns the board that results from making move (i, j) on the board"""
    if action not in actions(board):
        raise ValueError("Invalid action")

    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(board)
    return new_board


def winner(board: List[List[Optional[str]]]) -> Optional[str]:
    """Returns the winner of the game, if there is one"""
    # Check rows
    for row in board:
        if row.count('X') == 3:
            return 'X'
        if row.count('O') == 3:
            return 'O'

    # Check columns
    for col in range(BOARD_COLS):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] is not None:
            return board[0][col]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
        return board[0][2]

    return None


def terminal(board: List[List[Optional[str]]]) -> bool:
    """Returns True if game is over, False otherwise"""
    if winner(board) is not None:
        return True
    return all(cell is not None for row in board for cell in row)


def utility(board: List[List[Optional[str]]]) -> int:
    """Returns 1 if X has won, -1 if O has won, 0 otherwise"""
    result = winner(board)
    if result == 'X':
        return 1
    elif result == 'O':
        return -1
    else:
        return 0


def minimax(board: List[List[Optional[str]]]) -> Optional[Tuple[int, int]]:
    """Returns the optimal action for the current player on the board"""
    if terminal(board):
        return None

    current_player = player(board)

    if current_player == 'X':
        value, move = max_value(board)
    else:
        value, move = min_value(board)

    return move


def max_value(board: List[List[Optional[str]]]) -> Tuple[float, Optional[Tuple[int, int]]]:
    """Helper function for minimax - maximizes the value"""
    if terminal(board):
        return utility(board), None

    v = -math.inf
    best_move = None

    for action in actions(board):
        new_value, _ = min_value(result(board, action))
        if new_value > v:
            v = new_value
            best_move = action
            if v == 1:  # Early exit if winning move found
                break

    return v, best_move


def min_value(board: List[List[Optional[str]]]) -> Tuple[float, Optional[Tuple[int, int]]]:
    """Helper function for minimax - minimizes the value"""
    if terminal(board):
        return utility(board), None

    v = math.inf
    best_move = None

    for action in actions(board):
        new_value, _ = max_value(result(board, action))
        if new_value < v:
            v = new_value
            best_move = action
            if v == -1:  # Early exit if winning move found
                break

    return v, best_move


def draw_lines():
    """Draw the Tic Tac Toe grid lines"""
    # Horizontal lines
    pygame.draw.line(screen, LINE_COLOR, (0, SQUARE_SIZE), (WIDTH, SQUARE_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (0, 2 * SQUARE_SIZE), (WIDTH, 2 * SQUARE_SIZE), LINE_WIDTH)
    # Vertical lines
    pygame.draw.line(screen, LINE_COLOR, (SQUARE_SIZE, 0), (SQUARE_SIZE, HEIGHT), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (2 * SQUARE_SIZE, 0), (2 * SQUARE_SIZE, HEIGHT), LINE_WIDTH)


def draw_figures():
    """Draw X's and O's on the board"""
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == 'O':
                pygame.draw.circle(screen, CIRCLE_COLOR,
                                   (int(col * SQUARE_SIZE + SQUARE_SIZE // 2),
                                    int(row * SQUARE_SIZE + SQUARE_SIZE // 2)),
                                   CIRCLE_RADIUS, CIRCLE_WIDTH)
            elif board[row][col] == 'X':
                pygame.draw.line(screen, CROSS_COLOR,
                                 (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE),
                                 (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE),
                                 CROSS_WIDTH)
                pygame.draw.line(screen, CROSS_COLOR,
                                 (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE),
                                 (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE),
                                 CROSS_WIDTH)


def display_game_over(winner: Optional[str]):
    """Display the game over message"""
    font = pygame.font.SysFont('comicsans', 40)
    if winner:
        text = f"{winner} wins!"
    else:
        text = "It's a tie!"

    text_surface = font.render(text, True, TEXT_COLOR)
    text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))

    # Create a semi-transparent overlay
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 128))
    screen.blit(overlay, (0, 0))

    screen.blit(text_surface, text_rect)

    # Add play again prompt
    again_font = pygame.font.SysFont('comicsans', 30)
    again_text = again_font.render("Press R to play again or Q to quit", True, TEXT_COLOR)
    again_rect = again_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
    screen.blit(again_text, again_rect)

    pygame.display.update()


def reset_game():
    """Reset the game state"""
    global board
    board = initial_state()
    screen.fill(BG_COLOR)
    draw_lines()
    pygame.display.update()


def main():
    """Main game loop"""
    global board

    # Draw the initial board
    draw_lines()
    pygame.display.update()

    # Game variables
    game_over = False
    ai_turn = False  # AI goes second as 'O'

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game_over:
                    reset_game()
                    game_over = False
                    ai_turn = False
                elif event.key == pygame.K_q and game_over:
                    pygame.quit()
                    sys.exit()

            if not game_over and not ai_turn and event.type == pygame.MOUSEBUTTONDOWN:
                # Human player's turn
                mouseX, mouseY = event.pos
                clicked_row = mouseY // SQUARE_SIZE
                clicked_col = mouseX // SQUARE_SIZE

                if board[clicked_row][clicked_col] is None:
                    board[clicked_row][clicked_col] = player(board)
                    draw_figures()
                    pygame.display.update()

                    if terminal(board):
                        game_over = True
                        display_game_over(winner(board))
                    else:
                        ai_turn = True

        # AI's turn
        if not game_over and ai_turn:
            pygame.time.delay(500)  # Small delay so AI doesn't feel instantaneous

            move = minimax(board)
            if move:
                row, col = move
                board[row][col] = player(board)
                draw_figures()
                pygame.display.update()

                if terminal(board):
                    game_over = True
                    display_game_over(winner(board))

            ai_turn = False


if __name__ == "__main__":
    main()