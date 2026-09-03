import random
import pygame
import sys

# =============================================================================
# TICK CONFIGURATION
# =============================================================================
TICK_DURATION        = 0.1   # 1 tick = 0.1 simulated seconds
YELLOW_TIME          = 4.0   # yellow phase length (simulated seconds)

# Spawn attempt frequency.  At 2.2 px/tick a right/left vehicle needs ~268
# ticks (~26.8 s) to reach the stop line, so sim_duration must be ≥ 120 s

# during optimisation so vehicles actually arrive, queue, and clear.
TRAFFIC_MODES = {
    "low": 40,      # slow spawning
    "medium": 15,   # your current behavior
    "high": 5      # fast spawning
}

# =============================================================================
# SPEED & SPACING
# =============================================================================
# Pixels-per-tick (equal to the original pixels-per-frame at 60 FPS).
speeds = {'car': 2.2, 'bus': 1.8, 'truck': 1.8, 'bike': 2.5}

# Minimum gap between the rear of the vehicle ahead and the nose of the follower.
SAFE_GAP = 15   # px


# =============================================================================
# INTERSECTION OCCUPANCY SYSTEM
# =============================================================================
# The junction box is the rectangle between the four stop lines.
#   x: stopLines['right'] → stopLines['left']   = 590 → 800
#   y: stopLines['down']  → stopLines['up']     = 330 → 535
JUNCTION = pygame.Rect(590, 330, 210, 205)   # (left, top, width, height)

# Conflict table: directions that CANNOT share the junction simultaneously.
# Physics rationale:
#   right & down  – perpendicular crossing paths collide at junction centre
#   right & up    – perpendicular crossing paths collide at junction centre
#   left  & down  – perpendicular crossing paths collide at junction centre
#   left  & up    – perpendicular crossing paths collide at junction centre
#   right & left  – same axis, opposite lanes → no spatial conflict
#   up    & down  – same axis, opposite lanes → no spatial conflict
CONFLICTS = {
    'right': {'down', 'up'},
    'left':  {'down', 'up'},
    'down':  {'right', 'left'},
    'up':    {'right', 'left'},
}


class IntersectionManager:
    """
    Tick-accurate intersection reservation system.

    At the start of each tick the manager computes which directions currently
    have vehicles INSIDE the junction box.  A vehicle approaching the stop line
    is blocked if any conflicting direction is already occupying the junction.

    This is intentionally simple and deterministic:
      • No per-vehicle reservation IDs.
      • No lookahead or prediction.
      • O(n) where n = vehicles in the junction (typically 1–3).
      • Evaluated fresh every tick from ground-truth vehicle positions.
    """

    def __init__(self):
        # Directions that currently have at least one vehicle inside JUNCTION.
        self._occupied_directions: set = set()

    def update(self, simulation):
        """
        Recompute which directions are occupying the junction this tick.
        Must be called BEFORE any vehicle.move() so the constraint is current.
        """
        self._occupied_directions.clear()
        for v in simulation:
            vr = pygame.Rect(int(v.x), int(v.y),
                             v.image.get_width(), v.image.get_height())
            if JUNCTION.colliderect(vr):
                self._occupied_directions.add(v.direction)

    def can_enter(self, direction: str) -> bool:
        """
        Return True if a vehicle from `direction` is allowed to enter the
        junction this tick, i.e. no conflicting direction is currently inside.
        """
        return self._occupied_directions.isdisjoint(CONFLICTS.get(direction, set()))

    @property
    def occupied_directions(self) -> set:
        return frozenset(self._occupied_directions)


# =============================================================================
# SPAWN-POINT POSITIONS (one entry per lane 0-2)
# =============================================================================
x = {
    'right': [0, 0, 0],
    'down':  [755, 727, 697],
    'left':  [1400, 1400, 1400],
    'up':    [602, 627, 657],
}
y = {
    'right': [348, 370, 398],
    'down':  [0, 0, 0],
    'left':  [498, 466, 436],
    'up':    [800, 800, 800],
}

vehicleTypes     = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
stopLines   = {'right': 590, 'down': 330, 'left': 800, 'up': 535}


# =============================================================================
# VEHICLE CLASS
# =============================================================================
class Vehicle(pygame.sprite.Sprite):
    """
    Single vehicle in the simulation.

    Timing model (tick-based, no wall-clock):
      spawn_tick      – tick at which this vehicle was created.
      stop_line_tick  – tick of first contact with the stop line.
                        Only this interval is counted as waiting time;
                        free-travel time is excluded from fitness.

    Movement:
      speed is in px/tick.  displacement per tick = self.speed (no multiplier).

    Intersection constraint:
      Before crossing the stop line the vehicle checks IntersectionManager.
      If a conflicting direction occupies the junction the vehicle stays at
      the stop line even when the signal is green.
    """

    def __init__(self, lane, vtype, dir_num, direction,
                 vehicles_dict, simulation_group, images_cache, spawn_tick):
        super().__init__()

        self.lane           = lane
        self.vtype          = vtype
        self.speed          = speeds[vtype]
        self.dir_num        = dir_num
        self.direction      = direction
        self.x              = float(x[direction][lane])
        self.y              = float(y[direction][lane])
        self.crossed        = False
        self.spawn_tick     = spawn_tick
        self.stop_line_tick = None   
        self.was_stopped = False 

        self._vehicles   = vehicles_dict
        self._simulation = simulation_group

        self._vehicles[direction][lane].append(self)
        self.index = len(self._vehicles[direction][lane]) - 1

        self.image = images_cache[direction][vtype]
        self._simulation.add(self)

    # ------------------------------------------------------------------
    # Car-following: gap to the vehicle directly ahead in the same lane
    # ------------------------------------------------------------------
    def _can_move(self) -> bool:
        """
        True if there is at least SAFE_GAP pixels between this vehicle's
        front face and the rear face of the vehicle ahead.
        Lead vehicles (index 0) are never blocked by following distance.
        """
        if self.index == 0:
            return True

        front = self._vehicles[self.direction][self.lane][self.index - 1]

        if self.direction == 'right':
            return (self.x + self.image.get_width() + SAFE_GAP) < front.x
        elif self.direction == 'left':
            return (self.x - SAFE_GAP) > (front.x + front.image.get_width())
        elif self.direction == 'down':
            return (self.y + self.image.get_height() + SAFE_GAP) < front.y
        elif self.direction == 'up':
            return (self.y - SAFE_GAP) > (front.y + front.image.get_height())

        return True

    # ------------------------------------------------------------------
    # Hard-clamp: push vehicle back to the stop line if it overshot
    # ------------------------------------------------------------------
    def _enforce_stop_line(self):
        if self.direction == 'right':
            limit = stopLines['right'] - self.image.get_width()
            if self.x > limit:
                self.x = limit
        elif self.direction == 'left':
            limit = stopLines['left']
            if self.x < limit:
                self.x = limit
        elif self.direction == 'down':
            limit = stopLines['down'] - self.image.get_height()
            if self.y > limit:
                self.y = limit
        elif self.direction == 'up':
            limit = stopLines['up']
            if self.y < limit:
                self.y = limit

    # ------------------------------------------------------------------
    # Per-tick movement
    # ------------------------------------------------------------------
    def move(self, current_green, current_yellow, metrics, sim_tick,
             intersection: IntersectionManager):
        """
        Attempt to advance this vehicle by self.speed pixels.

        Gate order (all must pass to allow movement):
          1. Car-following gap check (_can_move).
          2. Signal state  – must be green for this direction.
          3. Intersection reservation – no conflicting direction in junction.
        Gates 2 and 3 only apply while the vehicle has not yet crossed.
        """
        # ── Gate 1: car-following ──────────────────────────────────────
        if not self._can_move():
            return

        signal_allows      = True
        junction_allows    = True
        at_stop_line       = False

        if not self.crossed:
            buf = 1   # 1-px look-ahead so we detect the line before crossing
            at_stop_line = False
            if self.direction == 'right':
                at_stop_line = (self.x + self.image.get_width() >= stopLines['right'])
            elif self.direction == 'left':
                at_stop_line = (self.x <= stopLines['left'])
            elif self.direction == 'down':
                at_stop_line = (self.y + self.image.get_height() >= stopLines['down'])
            elif self.direction == 'up':
                at_stop_line = (self.y <= stopLines['up'])

            if at_stop_line:
                # ── Gate 2: traffic signal ─────────────────────────────
                if not (current_green == self.dir_num and current_yellow == 0):
                    signal_allows = False

                # ── Gate 3: intersection reservation ──────────────────
                # Even on green, hold at the stop line if a conflicting
                # direction already occupies the junction box.
                if signal_allows and not intersection.can_enter(self.direction):
                    junction_allows = False

                # Start the waiting clock on first stop-line contact
                if self.stop_line_tick is None:
                    self.stop_line_tick = sim_tick

        # ── Movement or clamp ─────────────────────────────────────────

        can_move_now = signal_allows and junction_allows

        if can_move_now:
            # reset stop flag when moving
            self.was_stopped = False

            if   self.direction == 'right': self.x += self.speed
            elif self.direction == 'left':  self.x -= self.speed
            elif self.direction == 'down':  self.y += self.speed
            elif self.direction == 'up':    self.y -= self.speed

        else:
            if at_stop_line:
                self._enforce_stop_line()

                # count stop ONLY once when first blocked
                if not self.was_stopped:
                    metrics["total_stops"] += 1
                    self.was_stopped = True

        # ── Crossing detection ────────────────────────────────────────
        if not self.crossed:
            crossed_now = (
                (self.direction == 'right' and self.x > stopLines['right']) or
                (self.direction == 'left'  and self.x + self.image.get_width() < stopLines['left']) or
                (self.direction == 'down'  and self.y > stopLines['down']) or
                (self.direction == 'up'    and self.y + self.image.get_height() < stopLines['up'])
            )
            if crossed_now:
                self.crossed = True
                metrics['total_vehicles_passed'] += 1
                if self.stop_line_tick is not None:
                    wait_ticks = sim_tick - self.stop_line_tick
                    metrics['total_waiting_time'] += wait_ticks * TICK_DURATION


# =============================================================================
# HELPER UTILITIES
# =============================================================================

def _build_vehicles_dict():
    """Return a fresh per-direction, per-lane vehicle registry."""
    return {
        'right': {0: [], 1: [], 2: []},
        'down':  {0: [], 1: [], 2: []},
        'left':  {0: [], 1: [], 2: []},
        'up':    {0: [], 1: [], 2: []},
    }


def _load_images():
    """Load all vehicle and signal images; return (vehicle_cache, signal_cache)."""
    vehicle_cache = {}
    for d in directionNumbers.values():
        vehicle_cache[d] = {}
        for vt in vehicleTypes.values():
            vehicle_cache[d][vt] = pygame.image.load(f"images/{d}/{vt}.png")

    signal_cache = {
        'red':    pygame.image.load("images/signals/red.png"),
        'yellow': pygame.image.load("images/signals/yellow.png"),
        'green':  pygame.image.load("images/signals/green.png"),
    }
    return vehicle_cache, signal_cache


def _get_queue(simulation):
    """Count vehicles that have not yet crossed the stop line."""
    return sum(1 for v in simulation if not v.crossed)


# =============================================================================
# FITNESS FUNCTION  ← DO NOT MODIFY
# =============================================================================
def _fitness(metrics, simulation):
    """
    Fitness = avg_stop_line_wait  +  queue_fraction

    avg_stop_line_wait
        Mean simulated seconds spent waiting AT the stop line for vehicles
        that have already crossed.  Free-travel time is excluded.
        Vehicles that hit a green on arrival contribute 0.

    queue_fraction
        (blocked vehicles) / (total spawned so far), capped at 1.
        Normalised so it stays in the same 0–60 s range as avg_wait,
        preventing a flooded screen from swamping the signal entirely.
        Scaled by MAX_EXPECTED_WAIT (= longest possible single red cycle)
        so the two terms are always comparable in magnitude.

    Lower is better.
    """
    passed = metrics['total_vehicles_passed']
    queue = _get_queue(simulation)
    total_spawned = passed + queue

    if passed == 0:
        return 1000.0 + queue

    avg_wait = metrics['total_waiting_time'] / passed

    MAX_EXPECTED_WAIT = 60.0
    queue_penalty = (queue / max(total_spawned, 1)) * MAX_EXPECTED_WAIT

    stop_penalty = metrics["total_stops"] * 0.5   # 👈 ADD THIS

    return avg_wait + queue_penalty + stop_penalty

# =============================================================================
# SPAWN-CLEARANCE CHECK
# =============================================================================
def _lane_has_space(vehicles_dict, direction, lane, image) -> bool:
    """
    Return True only if the most-recently spawned vehicle in this lane has
    moved far enough from the spawn edge to leave room for a new one.
    Empty lanes always allow a spawn.
    """
    lane_list = vehicles_dict[direction][lane]
    if not lane_list:
        return True

    last          = lane_list[-1]
    min_clearance = SAFE_GAP + max(image.get_width(), image.get_height())

    if   direction == 'right': return last.x >= min_clearance
    elif direction == 'left':  return last.x <= (1400 - min_clearance)
    elif direction == 'down':  return last.y >= min_clearance
    elif direction == 'up':    return last.y <= (800  - min_clearance)

    return True


# =============================================================================
# DETERMINISTIC VEHICLE SPAWNER
# =============================================================================
def _maybe_spawn_vehicle(sim_tick, rng, vehicles_dict, simulation, vehicle_images, spawn_interval):
    """
    Attempt to spawn one vehicle every SPAWN_INTERVAL_TICKS ticks.

    The RNG is ALWAYS consumed on a spawn tick (even when the spawn is blocked
    by lane congestion) so the traffic pattern is identical for every call with
    the same seed, regardless of congestion outcomes.

    Direction distribution:
        right : 40 %
        down  : 20 %
        left  : 20 %
        up    : 20 %
    """
    if sim_tick % spawn_interval != 0:
        return

    # Consume RNG deterministically whether or not the spawn proceeds
    vtype_idx = rng.randint(0, 3)
    lane      = rng.randint(0, 2)
    r         = rng.randint(0, 99)

    if   r < 40: d = 0   # right  40 %
    elif r < 60: d = 1   # down   20 %
    elif r < 80: d = 2   # left   20 %
    else:        d = 3   # up     20 %

    direction = directionNumbers[d]
    vtype     = vehicleTypes[vtype_idx]
    image     = vehicle_images[direction][vtype]

    if not _lane_has_space(vehicles_dict, direction, lane, image):
        return   # congested — RNG already consumed, sequence stays intact

    Vehicle(
        lane, vtype, d, direction,
        vehicles_dict, simulation, vehicle_images,
        spawn_tick=sim_tick,
    )


# =============================================================================
# CLEANUP
# =============================================================================
def _cleanup_out_of_bounds(simulation, vehicles_dict):
    """
    Remove vehicles once they are fully off the visible canvas.

    Crossed vehicles  – culled when their trailing edge exits the 1200×800 window.
    Uncrossed vehicles – culled only with a large safety margin (shouldn't happen).
    """
    for v in list(simulation):
        remove = False

        if v.crossed:
            if   v.direction == 'right' and v.x > 1200:
                remove = True
            elif v.direction == 'left'  and v.x + v.image.get_width() < 0:
                remove = True
            elif v.direction == 'down'  and v.y > 800:
                remove = True
            elif v.direction == 'up'    and v.y + v.image.get_height() < 0:
                remove = True
        else:
            if v.x < -300 or v.x > 1800 or v.y < -300 or v.y > 1200:
                remove = True

        if remove:
            lane_list = vehicles_dict[v.direction][v.lane]
            if v in lane_list:
                lane_list.remove(v)
                for i, veh in enumerate(lane_list):
                    veh.index = i
            v.kill()


# =============================================================================
# SIGNAL CONTROLLER  (tick-based state machine, no threads)
# =============================================================================
class SignalController:
    """
    Cycle: GREEN for green_ticks[current] → YELLOW for YELLOW_TICKS
           → next direction → repeat.
    Purely tick-driven; no wall-clock or threading.
    """

    YELLOW_TICKS = round(YELLOW_TIME / TICK_DURATION)   # 40 ticks = 4 s

    def __init__(self, green_times):
        self.green_ticks   = [round(g / TICK_DURATION) for g in green_times]
        self.current_green = 0
        self.is_yellow     = False
        self._phase_tick   = 0

    def step(self):
        self._phase_tick += 1
        if not self.is_yellow:
            if self._phase_tick >= self.green_ticks[self.current_green]:
                self.is_yellow   = True
                self._phase_tick = 0
        else:
            if self._phase_tick >= self.YELLOW_TICKS:
                self.is_yellow     = False
                self.current_green = (self.current_green + 1) % 4
                self._phase_tick   = 0

    @property
    def yellow_flag(self):
        return 1 if self.is_yellow else 0


# =============================================================================
# EVALUATE  (called by PSO / GA on every fitness query)
# =============================================================================
def evaluate(green_times, sim_duration=30, headless=False, traffic_mode="medium"):
    """
    Run the deterministic tick-based traffic simulation and return a fitness
    score (lower = better).

    Parameters
    ----------
    green_times  : list[int|float] – [G0, G1, G2, G3] green phase durations (s)
    sim_duration : float           – simulated duration in seconds
    headless     : bool            – True → skip all pygame rendering

    Performance
    -----------
    180-second sim = 1 800 ticks.  Headless typically < 1 s on modern hardware.
    """
    spawn_interval = TRAFFIC_MODES[traffic_mode]
    total_ticks = round(sim_duration / TICK_DURATION)

    rng = random.Random(42)   # seeded → identical traffic for every candidate

    metrics = {
        'total_waiting_time':    0.0,
        'total_vehicles_passed': 0,
        'total_stops': 0
    }
    vehicles_dict = _build_vehicles_dict()
    intersection  = IntersectionManager()

    if not pygame.get_init():
        pygame.init()

    vehicle_images, signal_images = _load_images()

    if headless:
        screen = clock = font = bg = None
        red_img = yellow_img = green_img = None
    else:
        screen     = pygame.display.set_mode((1200, 800))
        pygame.display.set_caption("Traffic Simulation – Evaluating …")
        bg         = pygame.image.load("images/intersection.png")
        red_img    = signal_images['red']
        yellow_img = signal_images['yellow']
        green_img  = signal_images['green']
        font       = pygame.font.Font(None, 30)
        clock      = pygame.time.Clock()

    simulation = pygame.sprite.Group()
    signals    = SignalController(green_times)

    for sim_tick in range(total_ticks):

        # ① Window events
        if not headless:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        # ② Signal phase
        signals.step()
        cg = signals.current_green
        cy = signals.yellow_flag

        # ③ Spawn attempt (RNG always consumed)
        _maybe_spawn_vehicle(
            sim_tick, rng, vehicles_dict,
            simulation, vehicle_images,
            spawn_interval
        )
        
        # ④ Update intersection occupancy BEFORE moving any vehicle
        #    so every vehicle sees a consistent junction state this tick.
        intersection.update(simulation)

        # ⑤ Move all vehicles (passing intersection manager)
        for v in list(simulation):
            v.move(cg, cy, metrics, sim_tick, intersection)

        # ⑥ Cull off-screen vehicles
        _cleanup_out_of_bounds(simulation, vehicles_dict)

        # ⑦ Render (visual mode only)
        if not headless:
            screen.blit(bg, (0, 0))

            # Signals
            for i in range(4):
                if i == cg:
                    screen.blit(yellow_img if cy else green_img, signalCoods[i])
                else:
                    screen.blit(red_img, signalCoods[i])

            # Vehicles
            for v in simulation:
                screen.blit(v.image, (int(v.x), int(v.y)))

            # Optional: draw junction box for debugging
            # pygame.draw.rect(screen, (255, 0, 0), JUNCTION, 2)

            # HUD
            passed  = metrics['total_vehicles_passed']
            avg     = (metrics['total_waiting_time'] / passed) if passed else 0.0
            q       = _get_queue(simulation)
            fit     = _fitness(metrics, simulation)
            elapsed = sim_tick * TICK_DURATION

            hud = [
                (f"Avg Wait:  {avg:.2f} s",                               (255, 255, 255)),
                (f"Queue:     {q}",                                        (255, 255, 255)),
                (f"Fitness:   {fit:.2f}",                                  (255, 255, 255)),
                (f"Elapsed:   {elapsed:.1f} s / {sim_duration:.0f} s",    (200, 200, 100)),
                (f"Green:     Dir {cg} ({'YELLOW' if cy else 'GREEN'})",  (200, 200, 100)),
                (f"Tick:      {sim_tick} / {total_ticks}",                (150, 200, 255)),
                (f"Junction:  {', '.join(intersection.occupied_directions) or '—'}",
                                                                           (255, 180,  80)),
            ]
            for row, (text, colour) in enumerate(hud):
                screen.blit(font.render(text, True, colour), (20, 20 + row * 30))

            pygame.display.update()
            clock.tick(60)


    passed = metrics['total_vehicles_passed']
    avg = (metrics['total_waiting_time'] / passed) if passed else 0.0
    q = _get_queue(simulation)
    fit = _fitness(metrics, simulation)

    return {
        "fitness": fit,
        "avg_wait": avg,
        "queue": q,
        "passed": passed,
        "stops": metrics["total_stops"]
    }


# =============================================================================
# INTERACTIVE MAIN
# =============================================================================
def main():
    """
    Open the simulation window and run until the user closes it.
    Uses a non-seeded RNG for visual variety.
    """
    DEFAULT_GREEN_TIMES  = [20, 20, 20, 20]
    SIM_DURATION_SECONDS = 7200

    total_ticks   = round(SIM_DURATION_SECONDS / TICK_DURATION)
    rng           = random.Random()
    metrics       = {'total_waiting_time': 0.0, 'total_vehicles_passed': 0}
    vehicles_dict = _build_vehicles_dict()
    intersection  = IntersectionManager()

    pygame.init()
    simulation                    = pygame.sprite.Group()
    vehicle_images, signal_images = _load_images()

    screen     = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Traffic Simulation")
    bg         = pygame.image.load("images/intersection.png")
    red_img    = signal_images['red']
    yellow_img = signal_images['yellow']
    green_img  = signal_images['green']
    font       = pygame.font.Font(None, 30)
    clock      = pygame.time.Clock()
    signals    = SignalController(DEFAULT_GREEN_TIMES)

    for sim_tick in range(total_ticks):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        signals.step()
        cg = signals.current_green
        cy = signals.yellow_flag

        _maybe_spawn_vehicle(sim_tick, rng, vehicles_dict, simulation, vehicle_images)

        intersection.update(simulation)

        screen.blit(bg, (0, 0))

        for i in range(4):
            if i == cg:
                screen.blit(yellow_img if cy else green_img, signalCoods[i])
            else:
                screen.blit(red_img, signalCoods[i])

        for v in list(simulation):
            v.move(cg, cy, metrics, sim_tick, intersection)
            screen.blit(v.image, (int(v.x), int(v.y)))

        _cleanup_out_of_bounds(simulation, vehicles_dict)

        passed = metrics['total_vehicles_passed']
        avg    = (metrics['total_waiting_time'] / passed) if passed else 0.0
        screen.blit(font.render(f"Avg Wait: {avg:.2f} s",                      True, (255, 255, 255)), (20, 20))
        screen.blit(font.render(f"Queue:    {_get_queue(simulation)}",          True, (255, 255, 255)), (20, 50))
        screen.blit(font.render(f"Fitness:  {_fitness(metrics, simulation):.2f}", True, (255, 255, 255)), (20, 80))
        screen.blit(font.render(f"Junction: {', '.join(intersection.occupied_directions) or '—'}",
                                True, (255, 180, 80)), (20, 110))

        pygame.display.update()
        clock.tick(60)

