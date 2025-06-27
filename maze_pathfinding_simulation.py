# filename: maze_pathfinding_simulation.py
import sys
import os
import time
import math
import random
import argparse
import collections
import numpy as np
import pygame

# ----------------------------- Configuration Constants -----------------------------

# Maze sizes and cell sizes for display
MIN_GRID_SIZE = 50
MAX_GRID_SIZE = 100
DEFAULT_GRID_SIZE = 61  # Changed to an odd number for maze generation

CELL_SIZE_DEFAULT = 10  # pixels per cell, zoom will scale this

# Colors for visualization (R,G,B)
COLOR_BG = (10, 10, 10)
COLOR_WALL = (40, 40, 40)
COLOR_PATH_BFS = (50, 150, 255)
COLOR_PATH_DFS = (255, 100, 100)
COLOR_PATH_ASTAR = (100, 255, 100)
COLOR_FRONTIER_BFS = (100, 200, 255)
COLOR_FRONTIER_DFS = (255, 150, 150)
COLOR_FRONTIER_ASTAR = (150, 255, 150)
COLOR_EXPLORED_BFS = (30, 100, 200)
COLOR_EXPLORED_DFS = (200, 60, 60)
COLOR_EXPLORED_ASTAR = (60, 200, 60)
COLOR_START = (255, 255, 0)
COLOR_GOAL = (255, 0, 255)
COLOR_TEXT = (230, 230, 230)
COLOR_SUCCESS_BG = (20, 50, 20)

# Visualization parameters
FPS = 15
SPEED_STEPS = [10, 5, 4, 2, 1]  # Adjusted so index 0 is fastest
DEFAULT_SPEED_INDEX = 1  # index in SPEED_STEPS

# Sound parameters
SAMPLE_RATE = 44100
AUDIO_BUFFER_SIZE = 512

# Frequencies per algorithm for base pitch
FREQ_RANGE_BFS = (250, 600)
FREQ_RANGE_DFS = (700, 1200)
FREQ_RANGE_ASTAR = (1300, 1800)

# Success sound frequency and duration
SUCCESS_FREQS = {
    'BFS': 900,
    'DFS': 1200,
    'ASTAR': 1500
}
SUCCESS_SOUND_DURATION = 0.6

# Victory chord frequencies (harmonized)
VICTORY_FREQS = [440, 660, 880]

# ----------------------------- Maze Generation: DFS backtracker -----------------------------

class Maze:
    def __init__(self, width, height, complexity=0.75):
        # Maze dimensions in cells (must be odd to ensure walls)
        self.width = width if width % 2 == 1 else width - 1
        self.height = height if height % 2 == 1 else height - 1
        self.complexity = complexity  # complexity factor (0 to 1)
        self.grid = np.ones((self.height, self.width), dtype=np.uint8)  # 1 = wall, 0 = path
        self._generate()

    def _generate(self):
        # Initialize grid: start with all walls
        self.grid.fill(1)
        # Start position for maze carving
        start_y = random.randrange(1, self.height, 2)
        start_x = random.randrange(1, self.width, 2)
        self.grid[start_y, start_x] = 0

        walls = []
        # Add surrounding walls of the start cell
        if start_y - 2 > 0:
            walls.append((start_y - 2, start_x, start_y - 1, start_x))
        if start_y + 2 < self.height:
            walls.append((start_y + 2, start_x, start_y + 1, start_x))
        if start_x - 2 > 0:
            walls.append((start_y, start_x - 2, start_y, start_x - 1))
        if start_x + 2 < self.width:
            walls.append((start_y, start_x + 2, start_y, start_x + 1))

        while walls:
            idx = random.randint(0, len(walls) - 1)
            wy, wx, py, px = walls.pop(idx)
            if self.grid[wy, wx] == 1:
                # Check neighbors with path cells
                neighbors = 0
                for ny, nx in [(wy - 2, wx), (wy + 2, wx), (wy, wx - 2), (wy, wx + 2)]:
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        if self.grid[ny, nx] == 0:
                            neighbors += 1
                if neighbors == 1:
                    # Carve path
                    self.grid[wy, wx] = 0
                    self.grid[py, px] = 0
                    # Add walls around new cell
                    for ny, nx, nwy, nwx in [(wy - 2, wx, wy - 1, wx), (wy + 2, wx, wy + 1, wx),
                                             (wy, wx - 2, wy, wx - 1), (wy, wx + 2, wy, wx + 1)]:
                        if 0 <= ny < self.height and 0 <= nx < self.width:
                            if self.grid[ny, nx] == 1:
                                walls.append((ny, nx, nwy, nwx))

        # Add complexity by random wall openings based on complexity factor
        openings = int(self.complexity * (self.width * self.height) * 0.05)
        for _ in range(openings):
            # Random wall cell surrounded by two paths in opposite directions can be opened
            y = random.randrange(1, self.height - 1)
            x = random.randrange(1, self.width - 1)
            if self.grid[y, x] == 1:
                neighbors = 0
                # Check if two neighbors opposite directions are paths to avoid breaking maze too much
                pairs = [((y - 1, x), (y + 1, x)), ((y, x - 1), (y, x + 1))]
                for (ny1, nx1), (ny2, nx2) in pairs:
                    if self.grid[ny1, nx1] == 0 and self.grid[ny2, nx2] == 0:
                        neighbors = 2
                        break
                if neighbors == 2:
                    self.grid[y, x] = 0

    def is_path(self, y, x):
        return self.grid[y, x] == 0

# ----------------------------- Pathfinding Algorithms -----------------------------

class PathfindingAlgorithm:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.width = maze.width
        self.height = maze.height

        self.frontier = None
        self.explored = set()
        self.parent = dict()
        self.finished = False
        self.found_path = False
        self.path = []

    def neighbors(self, y, x):
        for ny, nx in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
            if 0 <= ny < self.height and 0 <= nx < self.width and self.maze.is_path(ny, nx):
                yield ny, nx

    def step(self):
        raise NotImplementedError

    def reconstruct_path(self):
        path = []
        current = self.goal
        while current != self.start:
            path.append(current)
            current = self.parent.get(current)
            if current is None:
                # No path found
                return []
        path.append(self.start)
        path.reverse()
        return path

    def progress(self):
        # Returns progress as float [0,1] estimating progress towards goal
        # Use heuristic distance for A*, and explored size for BFS/DFS
        if self.finished and self.found_path:
            return 1.0
        return 0.0

# BFS Implementation
class BFS(PathfindingAlgorithm):
    def __init__(self, maze, start, goal):
        super().__init__(maze, start, goal)
        self.frontier = collections.deque()
        self.frontier.append(start)
        self.explored = set()
        self.explored.add(start)

    def step(self):
        if self.finished:
            return
        if not self.frontier:
            self.finished = True
            self.found_path = False
            return
        current = self.frontier.popleft()
        if current == self.goal:
            self.finished = True
            self.found_path = True
            self.path = self.reconstruct_path()
            return
        for ny, nx in self.neighbors(*current):
            if (ny, nx) not in self.explored and (ny, nx) not in self.frontier:
                self.frontier.append((ny, nx))
                self.parent[(ny, nx)] = current
        self.explored.add(current)

    def progress(self):
        if self.finished and self.found_path:
            return 1.0
        if not self.frontier:
            return 0.0
        try:
            min_dist = min(abs(ny - self.goal[0]) + abs(nx - self.goal[1]) for ny, nx in self.frontier)
        except ValueError:
            return 0.0
        max_dist = self.width + self.height
        prog = 1.0 - (min_dist / max_dist)
        return max(0.0, min(prog, 1.0))

# DFS Implementation (iterative)
class DFS(PathfindingAlgorithm):
    def __init__(self, maze, start, goal):
        super().__init__(maze, start, goal)
        self.frontier = []
        self.frontier.append(start)
        self.explored = set()
        self.explored.add(start)

    def step(self):
        if self.finished:
            return
        if not self.frontier:
            self.finished = True
            self.found_path = False
            return
        current = self.frontier.pop()
        if current == self.goal:
            self.finished = True
            self.found_path = True
            self.path = self.reconstruct_path()
            return
        for ny, nx in self.neighbors(*current):
            if (ny, nx) not in self.explored and (ny, nx) not in self.frontier:
                self.frontier.append((ny, nx))
                self.parent[(ny, nx)] = current
        self.explored.add(current)

    def progress(self):
        if self.finished and self.found_path:
            return 1.0
        if not self.frontier:
            return 0.0
        try:
            min_dist = min(abs(ny - self.goal[0]) + abs(nx - self.goal[1]) for ny, nx in self.frontier)
        except ValueError:
            return 0.0
        max_dist = self.width + self.height
        prog = 1.0 - (min_dist / max_dist)
        return max(0.0, min(prog, 1.0))

# A* Implementation
class AStar(PathfindingAlgorithm):
    def __init__(self, maze, start, goal):
        super().__init__(maze, start, goal)
        self.frontier = []
        self.g_score = {start: 0}
        self.f_score = {start: self.heuristic(start, goal)}
        import heapq
        self.heapq = heapq
        self.count = 0
        self.heapq.heappush(self.frontier, (self.f_score[start], self.count, start))
        self.explored = set()

    def heuristic(self, a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def step(self):
        if self.finished:
            return

        # Pop next node that is not already explored
        while self.frontier:
            _, _, current = self.heapq.heappop(self.frontier)
            if current not in self.explored:
                break
        else:
            self.finished = True
            self.found_path = False
            return

        if current == self.goal:
            self.finished = True
            self.found_path = True
            self.path = self.reconstruct_path()
            return

        self.explored.add(current)
        for ny, nx in self.neighbors(*current):
            tentative_g = self.g_score.get(current, math.inf) + 1
            if tentative_g < self.g_score.get((ny, nx), math.inf):
                self.parent[(ny, nx)] = current
                self.g_score[(ny, nx)] = tentative_g
                f = tentative_g + self.heuristic((ny, nx), self.goal)
                self.f_score[(ny, nx)] = f
                self.count += 1
                self.heapq.heappush(self.frontier, (f, self.count, (ny, nx)))

    def progress(self):
        if self.finished and self.found_path:
            return 1.0
        if not self.frontier:
            return 0.0
        try:
            min_f = min(f for f, _, _ in self.frontier)
        except ValueError:
            return 0.0
        # Heuristic from start to goal (max possible f)
        f_goal = self.heuristic(self.start, self.goal)
        # Normalize progress between 0 and 1
        prog = 1.0 - min(min_f / (f_goal + self.width + self.height), 1.0)
        return max(0.0, min(prog, 1.0))

# ----------------------------- Audio Synthesis Utilities -----------------------------

def generate_sine_wave(frequency, duration=0.1, volume=1.0, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(2 * np.pi * frequency * t) * volume
    # Convert to 16-bit signed integers
    audio = np.int16(wave * 32767)
    return audio

def generate_success_sound(frequency, duration=SUCCESS_SOUND_DURATION):
    # Simple ascending sine wave burst with exponential decay
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.sin(2 * np.pi * frequency * t) * np.exp(-5 * t)
    audio = np.int16(wave * 32767)
    return audio

def generate_victory_chord(frequencies, duration=1.5):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = sum(np.sin(2 * np.pi * f * t) for f in frequencies)
    wave /= len(frequencies)
    audio = np.int16(wave * 32767 * 0.7)
    return audio

# ----------------------------- Audio Channel and Mixer -----------------------------

class AudioSynthesizer:
    def __init__(self):
        # Initialize mixer with stereo channels for better sound
        pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, AUDIO_BUFFER_SIZE)
        pygame.mixer.init()
        self.channels = {}
        self.muted = False
        self.volume = 0.6
        self.sounds_cache = {}

    def play_continuous_tone(self, name, frequency, volume):
        if self.muted:
            volume = 0.0
        key = (name, int(frequency * 10))
        sound = self.sounds_cache.get(key)
        if sound is None:
            # Generate a short sine wave buffer looped for continuous sound
            duration = 0.1
            audio_mono = generate_sine_wave(frequency, duration=duration, volume=volume)
            # Convert mono to stereo by duplicating channels
            audio_stereo = np.column_stack((audio_mono, audio_mono)).astype(np.int16)
            sound = pygame.sndarray.make_sound(audio_stereo)
            sound.set_volume(volume)
            self.sounds_cache[key] = sound
        channel = self.channels.get(name)
        if channel is None or not channel.get_busy():
            channel = sound.play(-1)
            self.channels[name] = channel
        else:
            channel.set_volume(volume)

    def play_success_sound(self, name, frequency):
        if self.muted:
            return
        audio_mono = generate_success_sound(frequency)
        audio_stereo = np.column_stack((audio_mono, audio_mono)).astype(np.int16)
        sound = pygame.sndarray.make_sound(audio_stereo)
        sound.set_volume(0.8)
        sound.play()

    def play_victory_chord(self):
        if self.muted:
            return
        audio_mono = generate_victory_chord(VICTORY_FREQS)
        audio_stereo = np.column_stack((audio_mono, audio_mono)).astype(np.int16)
        sound = pygame.sndarray.make_sound(audio_stereo)
        sound.set_volume(0.9)
        sound.play()

    def stop_all(self):
        for ch in self.channels.values():
            if ch:
                ch.stop()

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            self.stop_all()

# ----------------------------- Visualization and Simulation -----------------------------

class Simulation:
    def __init__(self, grid_size=DEFAULT_GRID_SIZE, cell_size=CELL_SIZE_DEFAULT,
                 speed_index=DEFAULT_SPEED_INDEX, complexity=0.75,
                 auto_test=False, strict_completion=False):
        self.grid_size = max(MIN_GRID_SIZE, min(MAX_GRID_SIZE, grid_size))
        if self.grid_size % 2 == 0:
            self.grid_size -= 1
        self.cell_size = cell_size
        self.speed_index = speed_index
        if self.speed_index < 0:
            self.speed_index = 0
        elif self.speed_index >= len(SPEED_STEPS):
            self.speed_index = DEFAULT_SPEED_INDEX
        self.speed = SPEED_STEPS[self.speed_index]
        self.complexity = complexity
        self.auto_test = auto_test
        self.strict_completion = strict_completion

        # Initialize pygame and screen
        pygame.init()
        self.window_width = self.grid_size * self.cell_size * 3 + 80
        self.window_height = self.grid_size * self.cell_size + 120
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        pygame.display.set_caption("Pathfinding Algorithms Maze Simulation - BFS vs DFS vs A*")

        self.font = pygame.font.SysFont('Consolas', 18)
        self.large_font = pygame.font.SysFont('Consolas', 36, bold=True)

        self.clock = pygame.time.Clock()

        # Generate maze (iteration 11: fresh maze with new random seed)
        # Use a new seed based on time and randomness to ensure fresh maze each run
        random.seed(time.time() + os.getpid())
        self.maze = Maze(self.grid_size, self.grid_size, complexity=self.complexity)

        # Define start and goal points: top-left and bottom-right open cells
        self.start = self._find_open_cell_near(1, 1)
        self.goal = self._find_open_cell_near(self.grid_size - 2, self.grid_size - 2)

        # Ensure start and goal are different and path accessible
        if self.start == self.goal:
            # Try to find another goal if same as start
            found_goal = False
            for y in range(self.grid_size - 2, 0, -1):
                for x in range(self.grid_size - 2, 0, -1):
                    if self.maze.is_path(y, x) and (y, x) != self.start:
                        self.goal = (y, x)
                        found_goal = True
                        break
                if found_goal:
                    break

        # Initialize algorithms
        self.algorithms = {
            'BFS': BFS(self.maze, self.start, self.goal),
            'DFS': DFS(self.maze, self.start, self.goal),
            'ASTAR': AStar(self.maze, self.start, self.goal)
        }

        self.colors = {
            'BFS': {
                'frontier': COLOR_FRONTIER_BFS,
                'explored': COLOR_EXPLORED_BFS,
                'path': COLOR_PATH_BFS
            },
            'DFS': {
                'frontier': COLOR_FRONTIER_DFS,
                'explored': COLOR_EXPLORED_DFS,
                'path': COLOR_PATH_DFS
            },
            'ASTAR': {
                'frontier': COLOR_FRONTIER_ASTAR,
                'explored': COLOR_EXPLORED_ASTAR,
                'path': COLOR_PATH_ASTAR
            }
        }

        self.audio = AudioSynthesizer()

        self.finished = False
        self.success_messages_shown = set()

        # Camera and zoom management
        self.camera_offset_x = 20
        self.camera_offset_y = 80
        self.zoom = 1.0
        self.max_zoom = 2.5
        self.min_zoom = 0.5
        self._auto_zoom_and_pan()

        # For auto-test mode exit handling
        self.exit_after_delay = False
        self.exit_delay_start = None
        self.exit_delay_seconds = 3

        # Safety counter for max steps in auto-test mode to avoid infinite loops
        self.max_total_steps = self.grid_size * self.grid_size * 30  # heuristic upper bound
        self.total_steps_taken = 0

    def _auto_zoom_and_pan(self):
        # Adjust zoom to fit maze in window height max, then scale for 3 columns
        max_height = self.window_height - 140
        max_width = (self.window_width - 80) // 3
        zoom_y = max_height / (self.cell_size * self.grid_size)
        zoom_x = max_width / (self.cell_size * self.grid_size)
        self.zoom = max(self.min_zoom, min(self.max_zoom, min(zoom_x, zoom_y)))

        # Calculate camera offset to center each maze section in its column
        self.cell_draw_size = max(1, int(self.cell_size * self.zoom))
        self.camera_offset_y = 60 + (max_height - self.cell_draw_size * self.grid_size) // 2
        self.camera_offset_x = 20

    def _find_open_cell_near(self, y, x):
        # Find nearest open cell to (y,x)
        if self.maze.is_path(y, x):
            return (y, x)
        queue = collections.deque()
        queue.append((y, x))
        visited = set()
        visited.add((y, x))
        while queue:
            cy, cx = queue.popleft()
            for ny, nx in [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]:
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size and (ny, nx) not in visited:
                    if self.maze.is_path(ny, nx):
                        return (ny, nx)
                    visited.add((ny, nx))
                    queue.append((ny, nx))
        # Fallback
        return (y, x)

    def draw_text(self, surf, text, pos, color=COLOR_TEXT, font=None):
        if font is None:
            font = self.font
        text_surf = font.render(text, True, color)
        surf.blit(text_surf, pos)

    def draw_maze_section(self, surf, maze, alg, rect, alg_name):
        # rect = pygame.Rect for drawing area
        cs = self.cell_draw_size
        # Draw walls and paths
        for y in range(maze.height):
            for x in range(maze.width):
                cell_rect = pygame.Rect(rect.left + x * cs, rect.top + y * cs, cs, cs)
                if maze.grid[y, x] == 1:
                    pygame.draw.rect(surf, COLOR_WALL, cell_rect)
                else:
                    pygame.draw.rect(surf, COLOR_BG, cell_rect)
        # Draw explored nodes
        if hasattr(alg, 'explored') and alg.explored:
            for y, x in alg.explored:
                if 0 <= y < maze.height and 0 <= x < maze.width:
                    cell_rect = pygame.Rect(rect.left + x * cs, rect.top + y * cs, cs, cs)
                    pygame.draw.rect(surf, self.colors[alg_name]['explored'], cell_rect)
        # Draw frontier nodes
        if alg.frontier:
            if isinstance(alg.frontier, collections.deque) or isinstance(alg.frontier, list):
                if alg_name != 'ASTAR':
                    for pos in alg.frontier:
                        y, x = pos
                        if 0 <= y < maze.height and 0 <= x < maze.width:
                            cell_rect = pygame.Rect(rect.left + x * cs, rect.top + y * cs, cs, cs)
                            pygame.draw.rect(surf, self.colors[alg_name]['frontier'], cell_rect)
                else:
                    # For A*, frontier is a list of tuples (f_score, count, (y,x))
                    for _, _, pos in alg.frontier:
                        y, x = pos
                        if 0 <= y < maze.height and 0 <= x < maze.width:
                            cell_rect = pygame.Rect(rect.left + x * cs, rect.top + y * cs, cs, cs)
                            pygame.draw.rect(surf, self.colors[alg_name]['frontier'], cell_rect)
        # Draw path last so it is visible
        if alg.finished and alg.found_path and alg.path:
            for y, x in alg.path:
                if 0 <= y < maze.height and 0 <= x < maze.width:
                    cell_rect = pygame.Rect(rect.left + x * cs, rect.top + y * cs, cs, cs)
                    pygame.draw.rect(surf, self.colors[alg_name]['path'], cell_rect)
        # Draw start and goal
        sy, sx = alg.start
        gy, gx = alg.goal
        if 0 <= sy < maze.height and 0 <= sx < maze.width:
            start_rect = pygame.Rect(rect.left + sx * cs, rect.top + sy * cs, cs, cs)
            pygame.draw.rect(surf, COLOR_START, start_rect)
        if 0 <= gy < maze.height and 0 <= gx < maze.width:
            goal_rect = pygame.Rect(rect.left + gx * cs, rect.top + gy * cs, cs, cs)
            pygame.draw.rect(surf, COLOR_GOAL, goal_rect)

    def draw(self):
        self.screen.fill(COLOR_BG)
        width = self.window_width
        height = self.window_height

        # Draw title
        self.draw_text(self.screen, "Maze Pathfinding Algorithms Comparison (BFS, DFS, A*)",
                       (20, 10), color=(220, 220, 220), font=self.large_font)

        # Draw maze sections for each algorithm side by side
        margin = 20
        section_width = (width - 4 * margin) // 3
        section_height = self.cell_draw_size * self.grid_size

        for i, alg_name in enumerate(['BFS', 'DFS', 'ASTAR']):
            rect = pygame.Rect(margin + i * (section_width + margin),
                               self.camera_offset_y, section_width, section_height)
            # Draw border and label
            pygame.draw.rect(self.screen, COLOR_WALL, rect, 2)
            self.draw_text(self.screen, alg_name, (rect.left + 10, rect.top - 30),
                           color=self.colors[alg_name]['path'], font=self.font)

            # Draw maze and algorithm state inside rect
            self.draw_maze_section(self.screen, self.maze, self.algorithms[alg_name], rect, alg_name)

        # Draw instructions and status at bottom
        instr_y = self.camera_offset_y + section_height + 10
        self.draw_text(self.screen, "Speed: {} updates/frame (fixed) | Press 'M' to mute/unmute audio".format(self.speed),
                       (20, instr_y))

        # If finished show success screen/message
        if self.finished:
            overlay = pygame.Surface((width, height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            self.draw_text(self.screen, "ALL ALGORITHMS COMPLETED!", (width // 2 - 150, height // 2 - 20),
                           color=(0, 255, 0), font=self.large_font)
            self.draw_text(self.screen, "Simulation will exit automatically." if self.auto_test else
                           "Press ESC to exit.", (width // 2 - 150, height // 2 + 40),
                           color=(255, 255, 255), font=self.font)

        pygame.display.flip()

    def update_algorithms(self):
        # Step all algorithms multiple times per frame according to speed setting
        # To avoid infinite loops in auto-test, limit max steps per frame to prevent blocking
        max_steps_per_frame = 1000  # safety cap to prevent infinite loops
        step_count = 0
        for _ in range(self.speed):
            any_pending = False
            for alg_name, alg in self.algorithms.items():
                if not alg.finished:
                    alg.step()
                    step_count += 1
                    self.total_steps_taken += 1
                    any_pending = True
                    if step_count >= max_steps_per_frame:
                        # Break early to avoid infinite processing in one frame
                        return
                    # Safety check: if total steps exceed max allowed, terminate simulation forcibly
                    if self.auto_test and self.total_steps_taken > self.max_total_steps:
                        # Print error and exit with failure
                        sys.stderr.write("ERROR: Maximum step count exceeded, possible infinite loop.\n")
                        sys.stderr.flush()
                        pygame.quit()
                        sys.exit(1)
            if not any_pending:
                # All finished, no need to continue stepping more times this frame
                break

    def update_audio(self):
        # Update continuous tones for each algorithm with pitch and volume according to progress
        if self.audio.muted:
            return
        for alg_name, alg in self.algorithms.items():
            prog = alg.progress()
            freq_range = {
                'BFS': FREQ_RANGE_BFS,
                'DFS': FREQ_RANGE_DFS,
                'ASTAR': FREQ_RANGE_ASTAR
            }[alg_name]
            frequency = freq_range[0] + (freq_range[1] - freq_range[0]) * prog
            volume = 0.2 + 0.8 * prog
            self.audio.play_continuous_tone(alg_name, frequency, volume)

    def handle_success_sounds(self):
        # Play success sound once per algorithm when it finishes pathfinding with path
        for alg_name, alg in self.algorithms.items():
            if alg.finished and alg.found_path and alg_name not in self.success_messages_shown:
                self.audio.play_success_sound(alg_name, SUCCESS_FREQS[alg_name])
                self.success_messages_shown.add(alg_name)
                if self.auto_test:
                    print(f"{alg_name}_ALGORITHM_COMPLETE", flush=True)

    def check_all_finished(self):
        return all(alg.finished and alg.found_path for alg in self.algorithms.values())

    def check_any_finished_without_path(self):
        # Detect if any algorithm finished but did not find a path
        return any(alg.finished and not alg.found_path for alg in self.algorithms.values())

    def run(self):
        running = True

        # For auto-test strict completion timeout to avoid infinite runs
        auto_test_timeout_seconds = 300  # 5 minutes max
        start_time = time.time()

        while running:
            # Automatic termination on timeout in auto-test mode to prevent infinite loops
            if self.auto_test and (time.time() - start_time) > auto_test_timeout_seconds:
                # If strict completion required but not all finished or any no path, exit with error
                if self.strict_completion:
                    if not self.check_all_finished() or self.check_any_finished_without_path():
                        pygame.quit()
                        sys.exit(1)
                running = False
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.window_width, self.window_height = event.size
                    self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
                    self._auto_zoom_and_pan()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_m:
                        self.audio.toggle_mute()
                    elif event.key == pygame.K_SPACE:
                        # Optional pause/resume toggle could be implemented here if needed
                        pass

            if not self.finished:
                self.update_algorithms()
                self.handle_success_sounds()
                self.update_audio()

                # Check if all algorithms finished and found path
                if self.check_all_finished():
                    if not self.finished:
                        self.finished = True
                        self.audio.stop_all()
                        self.audio.play_victory_chord()
                        if self.auto_test:
                            print("SIMULATION_COMPLETE_SUCCESS", flush=True)
                        self.exit_after_delay = True
                        self.exit_delay_start = time.time()
                else:
                    # Defensive: if all algorithms finished but any no path found, handle accordingly
                    all_finished = all(alg.finished for alg in self.algorithms.values())
                    any_no_path = self.check_any_finished_without_path()
                    if all_finished and any_no_path:
                        # Respect strict completion flag: fail if strict, else finish gracefully
                        if self.strict_completion:
                            pygame.quit()
                            sys.exit(1)
                        else:
                            if not self.finished:
                                self.finished = True
                                self.exit_after_delay = True
                                self.exit_delay_start = time.time()
                                if self.auto_test:
                                    print("SIMULATION_COMPLETE_SUCCESS", flush=True)
            else:
                # In auto_test mode, exit after delay
                if self.auto_test and self.exit_after_delay and time.time() - self.exit_delay_start > self.exit_delay_seconds:
                    running = False

            self.draw()
            self.clock.tick(FPS)

        pygame.quit()

        # Auto test mode exit handling
        if self.auto_test:
            all_ok = self.check_all_finished()
            any_no_path = self.check_any_finished_without_path()
            if not all_ok or any_no_path:
                if self.strict_completion:
                    sys.exit(1)
            sys.exit(0)

# ----------------------------- Command-line and Entrypoint -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Maze Pathfinding Algorithms Visualization")
    parser.add_argument('--grid-size', type=int, default=DEFAULT_GRID_SIZE,
                        help='Grid size (odd number between 50 and 100)')
    parser.add_argument('--speed-index', type=int, default=DEFAULT_SPEED_INDEX,
                        help='Speed index (0-fastest to 4-slowest)')
    parser.add_argument('--complexity', type=float, default=0.75,
                        help='Maze complexity parameter (0 to 1)')
    parser.add_argument('--auto-test', action='store_true',
                        help='Run in auto-test mode (no user input, auto exit)')
    parser.add_argument('--strict-completion', action='store_true',
                        help='Require all algorithms to complete before exit')
    return parser.parse_args()

def main():
    args = parse_args()

    # Clamp grid size to odd number between MIN and MAX
    grid_size = args.grid_size
    if grid_size < MIN_GRID_SIZE:
        grid_size = MIN_GRID_SIZE
    elif grid_size > MAX_GRID_SIZE:
        grid_size = MAX_GRID_SIZE
    if grid_size % 2 == 0:
        grid_size -= 1

    speed_index = args.speed_index
    if speed_index < 0 or speed_index >= len(SPEED_STEPS):
        speed_index = DEFAULT_SPEED_INDEX

    complexity = args.complexity
    if complexity < 0.0:
        complexity = 0.0
    elif complexity > 1.0:
        complexity = 1.0

    sim = Simulation(grid_size=grid_size, speed_index=speed_index,
                     complexity=complexity, auto_test=args.auto_test,
                     strict_completion=args.strict_completion)
    sim.run()


if __name__ == "__main__":
    main()