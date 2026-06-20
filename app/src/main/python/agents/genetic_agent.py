from typing import Optional, List, Dict, Tuple
from agents.planet_wars_agent import PlanetWarsPlayer
from core.game_state import GameState, Action, Player, GameParams, Planet
from core.game_state_factory import GameStateFactory

class PlanetForecaster:
    def __init__(self, game_state: GameState, transporter_speed: float, max_ticks: int):
        self.planets = game_state.planets
        self.now = game_state.game_tick
        self.max_ticks = max_ticks
        self.speed = transporter_speed

        # Para cada planeta: lista de llegadas (tick, owner, n_ships), ordenada.
        self.arrivals: Dict[int, List[Tuple[int, Player, float]]] = {
            p.id: [] for p in self.planets
        }
        for p in self.planets:
            t = p.transporter
            if t is None:
                continue
            dest = self.planets[t.destination_index]
            arrival_tick = self.now + self._ticks_to_arrive(t.s, dest)
            self.arrivals[t.destination_index].append((arrival_tick, t.owner, t.n_ships))
        for pid in self.arrivals:
            self.arrivals[pid].sort(key=lambda a: a[0])

        # Cache de trayectorias ya calculadas.
        self._traj_cache: Dict[int, List[Tuple[int, float, Player]]] = {}

    # distancia -> nº de ticks hasta entrar en el radio del destino
    def _ticks_to_arrive(self, pos, dest: Planet) -> int:
        d = pos.distance(dest.position)
        effective = max(0.0, d - dest.radius)
        return int(effective / self.speed) + 1

    # construye la trayectoria del planeta (hitos en cada llegada
    def _trajectory(self, planet_id: int) -> List[Tuple[int, float, Player]]:
        if planet_id in self._traj_cache:
            return self._traj_cache[planet_id]

        planet = self.planets[planet_id]
        owner = planet.owner
        ships = planet.n_ships
        last_t = self.now
        traj: List[Tuple[int, float, Player]] = [(last_t, ships, owner)]

        for tick, fleet_owner, fleet_ships in self.arrivals[planet_id]:
            if tick > self.max_ticks:
                break
            if owner != Player.Neutral:
                ships += planet.growth_rate * (tick - last_t)
            last_t = tick

            if owner == Player.Neutral:
                # En neutral la llegada reduce las naves; si pasa de 0 cambia de dueño.
                ships -= abs(fleet_ships)
                if ships < 0:
                    owner = fleet_owner
                    ships = -ships
            else:
                if fleet_owner == owner:
                    ships += fleet_ships
                else:
                    ships -= fleet_ships
                    if ships < 0:
                        owner = owner.opponent()
                        ships = -ships
            traj.append((tick, ships, owner))

        if owner != Player.Neutral and last_t < self.max_ticks:
            ships += planet.growth_rate * (self.max_ticks - last_t)
        traj.append((self.max_ticks, ships, owner))

        self._traj_cache[planet_id] = traj
        return traj

    # consultas públicas
    def state_at(self, planet_id: int, tick: int) -> Tuple[float, Player]:
        """Naves y dueño del planeta en el tick dado."""
        traj = self._trajectory(planet_id)
        planet = self.planets[planet_id]
        ships, owner, prev_t = traj[0][1], traj[0][2], traj[0][0]
        for t, s, o in traj:
            if t > tick:
                if owner != Player.Neutral:
                    ships += planet.growth_rate * (tick - prev_t)
                return (ships, owner)
            ships, owner, prev_t = s, o, t
        if owner != Player.Neutral:
            ships += planet.growth_rate * (tick - prev_t)
        return (ships, owner)

    def final_state(self, planet_id: int) -> Tuple[float, Player]:
        """Naves y dueño al final de la simulación, sin nuevas órdenes."""
        traj = self._trajectory(planet_id)
        return (traj[-1][1], traj[-1][2])


#  AGENTE
class GeneticAgent(PlanetWarsPlayer):

    def get_action(self, game_state: GameState) -> Action:
        self.PARAMS = {
        "growth_value":      25.0,  # peso del crecimiento de un planeta
        "distance_penalty":  0.05,  # penalización por distancia
        "attack_margin":     1.15,  # naves extra sobre lo justo al atacar
        "defense_margin":    1.05,  # naves extra al defender
        "enemy_bonus":       1.6,   # valor relativo de capturar enemigo
        "neutral_bonus":     1.0,   # valor relativo de capturar neutral
        "defense_value":     1.5,   # cuánto vale defender un planeta propio
        "exposure_penalty":  0.02,  # riesgo por nave que sale del origen
        "keep_garrison":     0.0,   # naves mínimas que conserva un planeta
        "max_send_fraction": 0.95,  # fracción máxima enviable por jugada
        }

        p = self.PARAMS

        my_planets = [pl for pl in game_state.planets
                      if pl.owner == self.player and pl.transporter is None and pl.n_ships > 0]
        if not my_planets:
            return Action.do_nothing()

        max_ticks = self.params.max_ticks
        forecaster = PlanetForecaster(game_state, self.params.transporter_speed, max_ticks)

        best_action: Optional[Action] = None
        best_value = 0.0

        for source in my_planets:
            spendable = source.n_ships - p["keep_garrison"]
            if spendable <= 0:
                continue
            max_send = min(spendable, source.n_ships * p["max_send_fraction"])

            for dest in game_state.planets:
                if dest.id == source.id:
                    continue
                send, value = self._evaluate_move(source, dest, max_send, forecaster, game_state)
                if send > 0 and value > best_value:
                    best_value = value
                    best_action = Action(
                        player_id=self.player,
                        source_planet_id=source.id,
                        destination_planet_id=dest.id,
                        num_ships=send,
                    )

        return best_action if best_action is not None else Action.do_nothing()

    def _evaluate_move(self, source: Planet, dest: Planet, max_send: float,
                       forecaster: PlanetForecaster, game_state: GameState):
        p = self.PARAMS
        distance = source.position.distance(dest.position)
        n_ticks = int(max(0.0, distance - dest.radius) / self.params.transporter_speed) + 1
        arrival_tick = game_state.game_tick + n_ticks
        dist_cost = p["distance_penalty"] * n_ticks

        max_ticks = self.params.max_ticks
        if arrival_tick >= max_ticks:
            return (0.0, 0.0)

        final_ships, final_owner = forecaster.final_state(dest.id)

        if dest.owner == self.player:
            # DEFENSA 
            # Solo si el planeta termina la simulación en manos ajenas.
            if final_owner == self.player:
                return (0.0, 0.0)
            # 'final_ships' son las naves con que el rival se queda el planeta:
            # es el déficit que debemos cubrir para que no caiga.
            deficit = final_ships
            send = min(max_send, deficit * p["defense_margin"])
            if send <= 0:
                return (0.0, 0.0)
            saved = dest.growth_rate * p["growth_value"] * p["defense_value"]
            value = saved - dist_cost - p["exposure_penalty"] * send
            return (send, value)

        # ATAQUE
        if final_owner == self.player:
            return (0.0, 0.0)
        ships_on_arrival, owner_on_arrival = forecaster.state_at(dest.id, arrival_tick)
        if owner_on_arrival == self.player:
            return (0.0, 0.0)
        else:
            needed = ships_on_arrival * p["attack_margin"]
        if needed > max_send:
            return (0.0, 0.0)
        send = needed
        bonus = p["enemy_bonus"] if dest.owner == self.player.opponent() else p["neutral_bonus"]
        gain = dest.growth_rate * p["growth_value"] * bonus
        value = gain - dist_cost - p["exposure_penalty"] * send
        return (send, value)

    def get_agent_type(self) -> str:
        return "Genetic_V1"