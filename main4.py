import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigs, ArpackNoConvergence
from collections import defaultdict
import zlib
import random
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML, display, clear_output
import warnings
warnings.filterwarnings('ignore')

# Установка стиля
plt.style.use('dark_background')

# ============================================================
# СТРУКТУРА ДАННЫХ
# ============================================================

@dataclass
class RealityState:
    graph: nx.Graph = field(default_factory=nx.Graph)
    triples: Set[Tuple[int, int, int]] = field(default_factory=set)
    curvature_field: Dict[int, float] = field(default_factory=dict)
    time: int = 0
    _next_id: int = 0
    
    def add_node(self) -> int:
        node_id = self._next_id
        self._next_id += 1
        self.graph.add_node(node_id, created_at=self.time)
        self.curvature_field[node_id] = 0.0
        return node_id
    
    def add_edge(self, u: int, v: int, weight: float = 1.0):
        if u not in self.graph or v not in self.graph:
            return
        self.graph.add_edge(u, v, weight=weight)
        common = set(self.graph.neighbors(u)) & set(self.graph.neighbors(v))
        for w in common:
            if w not in (u, v):
                self.triples.add(tuple(sorted([u, v, w])))
    
    def remove_node(self, node: int):
        if node in self.graph:
            self.graph.remove_node(node)
            self.curvature_field.pop(node, None)
            self.triples = {t for t in self.triples if node not in t}
    
    @property
    def node_count(self) -> int:
        return len(self.graph.nodes)
    
    @property
    def edge_count(self) -> int:
        return len(self.graph.edges)
    
    @property
    def triple_count(self) -> int:
        return len(self.triples)
    
    def largest_component_size(self) -> int:
        if self.node_count == 0:
            return 0
        return len(max(nx.connected_components(self.graph), key=len))

# ============================================================
# ВЫЧИСЛЕНИЯ
# ============================================================

def compute_local_curvature_field(state: RealityState) -> Dict[int, float]:
    local_curvature = defaultdict(float)
    
    for u, v, n_w in state.triples:
        w_uv = state.graph[u][v].get('weight', 1.0)
        w_vw = state.graph[v][n_w].get('weight', 1.0)
        w_wu = state.graph[n_w][u].get('weight', 1.0)
        
        eps = 1e-6
        a = 1.0 / (w_vw + eps)
        b = 1.0 / (w_wu + eps)
        c = 1.0 / (w_uv + eps)
        
        try:
            cos_A = np.clip((b**2 + c**2 - a**2) / (2*b*c + eps), -1+eps, 1-eps)
            cos_B = np.clip((a**2 + c**2 - b**2) / (2*a*c + eps), -1+eps, 1-eps)
            cos_C = np.clip((a**2 + b**2 - c**2) / (2*a*b + eps), -1+eps, 1-eps)
            
            A = np.arccos(cos_A)
            B = np.arccos(cos_B)
            C = np.arccos(cos_C)
            
            deficit = A + B + C - np.pi
            local_curvature[u] += deficit / 3.0
            local_curvature[v] += deficit / 3.0
            local_curvature[n_w] += deficit / 3.0
        except (ValueError, KeyError, ZeroDivisionError):
            continue
    
    return dict(local_curvature)

def compute_curvature(state: RealityState) -> float:
    if not state.curvature_field:
        return 0.0
    return sum(c**2 for c in state.curvature_field.values())

def compute_integrated_info(state: RealityState) -> float:
    if state.node_count < 2:
        return 0.0
    
    try:
        if nx.is_connected(state.graph):
            L = nx.normalized_laplacian_matrix(state.graph)
            n = state.node_count
        else:
            largest_cc = max(nx.connected_components(state.graph), key=len)
            subgraph = state.graph.subgraph(largest_cc).copy()
            if len(subgraph) < 2:
                return 0.0
            L = nx.normalized_laplacian_matrix(subgraph)
            n = len(subgraph)
        
        if n <= 10:
            eigenvalues = np.linalg.eigvalsh(L.toarray())
        else:
            k = min(3, n-1)
            eigenvalues = eigs(L, k=k, which='SM', return_eigenvectors=False)
            eigenvalues = np.sort(np.abs(eigenvalues))
        
        lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        
        # Нормализованная энтропия спектра как мера сложности
        if len(eigenvalues) > 0:
            probs = np.abs(eigenvalues) / (np.sum(np.abs(eigenvalues)) + 1e-10)
            spectral_entropy = -np.sum(probs * np.log(probs + 1e-10)) / np.log(len(eigenvalues) + 1)
        else:
            spectral_entropy = 0.0
        
        # Комбинированная мера: алгебраическая связность + спектральная сложность
        return lambda_2 * (1.0 + spectral_entropy)
    except (np.linalg.LinAlgError, ValueError, ArpackNoConvergence):
        return 0.0

def serialize_state(state: RealityState) -> bytes:
    if state.node_count == 0:
        return b'empty'
    edges_list = []
    for u, v in state.graph.edges():
        weight = state.graph[u][v].get('weight', 1.0)
        edges_list.append(f"{min(u,v)},{max(u,v)},{weight:.3f}")
    return ";".join(sorted(edges_list)).encode('utf-8')

def compute_accumulated_complexity(state: RealityState, initial_serialized: bytes) -> float:
    if state.node_count == 0:
        return 0.0
    current = serialize_state(state)
    return max(0, len(zlib.compress(current, level=9)) - len(initial_serialized))

# ============================================================
# ДИНАМИКА
# ============================================================

def generate_moves(state: RealityState) -> List[Tuple[str, Tuple, float]]:
    moves = []
    
    # EXPAND: создать узел на ребре
    if state.edge_count > 0:
        edges = list(state.graph.edges())
        n_sample = min(15, len(edges))
        for u, v in random.sample(edges, n_sample) if len(edges) > n_sample else edges:
            moves.append(('expand', (u, v), 1.0))
    
    # INTEGRATE: схлопнуть тройку
    if state.triple_count > 0:
        triples = list(state.triples)
        n_sample = min(10, len(triples))
        for t in random.sample(triples, n_sample) if len(triples) > n_sample else triples:
            moves.append(('integrate', t, 1.0))
    
    # DEFLATE: удалить изолированный узел
    for node in state.graph.nodes():
        if state.graph.degree(node) == 0:
            moves.append(('deflate', (node,), 1.0))
    
    # SEED: вакуумная флуктуация
    moves.append(('seed', (), 1.0))
    
    # CONNECT: соединить компоненты (с повышенным приоритетом)
    if not nx.is_connected(state.graph) and state.node_count >= 4:
        components = sorted(nx.connected_components(state.graph), key=len, reverse=True)
        if len(components) >= 2:
            c1 = list(components[0])
            c2 = list(components[1])
            if c1 and c2:
                moves.append(('connect', (random.choice(c1), random.choice(c2)), 3.0))
    
    return moves

def apply_move(state: RealityState, move: Tuple) -> RealityState:
    import copy
    ns = RealityState(
        graph=copy.deepcopy(state.graph),
        time=state.time + 1
    )
    ns._next_id = state._next_id
    ns.triples = state.triples.copy()
    ns.curvature_field = state.curvature_field.copy()
    
    move_type, args = move[0], move[1]
    
    if move_type == 'expand':
        u, v = args
        if u in ns.graph and v in ns.graph:
            w = ns.add_node()
            ns.add_edge(w, u, weight=1.0)
            ns.add_edge(w, v, weight=1.0)
            if ns.graph.has_edge(u, v):
                ns.graph[u][v]['weight'] = min(3.0, ns.graph[u][v].get('weight', 1.0) * 1.05)
    
    elif move_type == 'integrate':
        u, v, w = args
        if all(n in ns.graph for n in [u, v, w]):
            particle = ns.add_node()
            ext = defaultdict(float)
            for node in [u, v, w]:
                for nb in state.graph.neighbors(node):
                    if nb not in [u, v, w]:
                        ext[nb] += state.graph[node][nb].get('weight', 1.0)
            for nb, tw in ext.items():
                if nb in ns.graph:
                    ns.add_edge(particle, nb, weight=tw / 3.0)
            ns.graph.remove_nodes_from([u, v, w])
            for n in [u, v, w]:
                ns.curvature_field.pop(n, None)
            ns.triples = {t for t in ns.triples if not any(n in [u, v, w] for n in t)}
    
    elif move_type == 'deflate':
        (node,) = args
        if node in ns.graph and ns.graph.degree(node) == 0:
            ns.remove_node(node)
    
    elif move_type == 'seed':
        a = ns.add_node()
        b = ns.add_node()
        ns.add_edge(a, b, weight=1.0)
    
    elif move_type == 'connect':
        u, v = args
        if u in ns.graph and v in ns.graph:
            ns.add_edge(u, v, weight=0.3)  # Слабая связь
    
    ns.curvature_field = compute_local_curvature_field(ns)
    return ns


def score_move_cheap(state: RealityState, move: Tuple,
                     E: float, alpha: float, beta: float, gamma: float,
                     vacuum: float, connectivity: float,
                     C: float, lcs: int, U: float) -> Tuple[float, object]:
    """
    Cheap approximate ΔE score for a candidate move.

    - deflate / seed / connect: fully analytical — no graph copy, no eigenvalues.
    - expand / integrate: one deepcopy + curvature recompute, but eigenvalue
      decomposition and zlib compression are skipped. I is proxied by lcs/N,
      U is held constant (γ=0.02 makes its per-step variation negligible).

    Returns (score, candidate_state_or_None). For expand/integrate the computed
    state is returned so the caller can reuse it if this move is chosen.
    """
    move_type, args = move[0], move[1]
    priority = move[2] if len(move) > 2 else 1.0
    N = state.node_count

    if move_type == 'deflate':
        # Isolated node contributes 0 curvature and is outside the largest component.
        ns_N = N - 1
        ns_lcs = lcs
        ns_I_proxy = ns_lcs / max(1, ns_N)
        ns_E = (alpha * C - beta * ns_I_proxy + gamma * U
                - vacuum * ns_N - connectivity * (ns_lcs / max(1, ns_N)))
        return priority * np.exp(-(ns_E - E)), None

    elif move_type == 'seed':
        # Two fresh nodes + one edge between them: no new triples, C unchanged.
        ns_N = N + 2
        ns_lcs = max(lcs, 2)
        ns_I_proxy = ns_lcs / max(1, ns_N)
        ns_E = (alpha * C - beta * ns_I_proxy + gamma * U
                - vacuum * ns_N - connectivity * (ns_lcs / max(1, ns_N)))
        return priority * np.exp(-(ns_E - E)), None

    elif move_type == 'connect':
        # Merges two components. Cross-component edge creates no new triples
        # (the two endpoints share no neighbors before the connection).
        u, v = args
        comp_u = nx.node_connected_component(state.graph, u)
        comp_v = nx.node_connected_component(state.graph, v)
        ns_N = N
        ns_lcs = len(comp_u) + len(comp_v)
        ns_I_proxy = ns_lcs / max(1, ns_N)
        ns_E = (alpha * C - beta * ns_I_proxy + gamma * U
                - vacuum * ns_N - connectivity * (ns_lcs / max(1, ns_N)))
        return priority * np.exp(-(ns_E - E)), None

    else:  # expand or integrate: topology changes require a graph copy
        ns = apply_move(state, move)
        ns_C = compute_curvature(ns)
        ns_N = ns.node_count
        ns_lcs = ns.largest_component_size()
        ns_I_proxy = ns_lcs / max(1, ns_N)
        ns_E = (alpha * ns_C - beta * ns_I_proxy + gamma * U
                - vacuum * ns_N - connectivity * (ns_lcs / max(1, ns_N)))
        return priority * np.exp(-(ns_E - E)), ns


# ============================================================
# СИМУЛЯТОР
# ============================================================

class RealitySimulation:
    def __init__(self, alpha=0.3, beta=3.0, gamma=0.02, vacuum=0.8, connectivity=3.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.vacuum = vacuum
        self.connectivity = connectivity
        
        self.state = RealityState()
        u = self.state.add_node()
        v = self.state.add_node()
        self.state.add_edge(u, v, weight=1.0)
        self.state.curvature_field = compute_local_curvature_field(self.state)
        
        self.initial_serialized = serialize_state(self.state)
        self.history = []
        # Сохраняем ВСЕ состояния для идеальной анимации
        self.all_states = [self._copy_state(self.state)]
    
    def _copy_state(self, state):
        """Глубокое копирование состояния для истории"""
        import copy
        ns = RealityState()
        ns.graph = copy.deepcopy(state.graph)
        ns.triples = state.triples.copy()
        ns.curvature_field = state.curvature_field.copy()
        ns.time = state.time
        ns._next_id = state._next_id
        return ns
    
    def step(self) -> Dict:
        C = compute_curvature(self.state)
        I = compute_integrated_info(self.state)
        U = compute_accumulated_complexity(self.state, self.initial_serialized)

        lcs = self.state.largest_component_size()
        E = (self.alpha * C
             - self.beta * I
             + self.gamma * U
             - self.vacuum * self.state.node_count
             - self.connectivity * (lcs / max(1, self.state.node_count)))

        moves = generate_moves(self.state)
        scores: List[float] = []
        candidate_states: Dict[int, RealityState] = {}

        for i, move in enumerate(moves):
            score, ns = score_move_cheap(
                self.state, move, E,
                self.alpha, self.beta, self.gamma, self.vacuum, self.connectivity,
                C, lcs, U,
            )
            scores.append(score)
            if ns is not None:
                candidate_states[i] = ns

        total = sum(scores)
        probs = [s / total for s in scores] if total > 0 else [1 / len(scores)] * len(scores)

        chosen_idx = np.random.choice(len(moves), p=probs)
        chosen = moves[chosen_idx]

        # Reuse the state already computed during scoring when available (expand/integrate),
        # otherwise apply the move now (deflate/seed/connect were not materialized).
        if chosen_idx in candidate_states:
            self.state = candidate_states[chosen_idx]
        else:
            self.state = apply_move(self.state, chosen)

        self.all_states.append(self._copy_state(self.state))
        
        curvs = list(self.state.curvature_field.values()) or [0.0]
        
        record = {
            'time': self.state.time,
            'nodes': self.state.node_count,
            'edges': self.state.edge_count,
            'triples': self.state.triple_count,
            'largest_component': lcs,
            'C': C, 'I': I, 'U': U,
            'move_type': chosen[0],
            'complexity_bytes': len(zlib.compress(serialize_state(self.state), level=9)),
            'curvature_mean': np.mean(curvs),
            'curvature_std': np.std(curvs),
            'curvature_max': max(abs(np.array(curvs))) if curvs else 0.0,
        }
        self.history.append(record)
        return record
    
    def run(self, steps=100, verbose=True):
        print(f"{'='*70}")
        print(f"  RELATE: Квантовое дыхание графовой реальности")
        print(f"  α={self.alpha}, β={self.beta}, γ={self.gamma}")
        print(f"  vacuum={self.vacuum}, connectivity={self.connectivity}")
        print(f"{'='*70}\n")
        
        for i in range(steps):
            r = self.step()
            if verbose and (i % 20 == 0 or i == steps-1):
                print(f"t={r['time']:4d} | N={r['nodes']:4d} E={r['edges']:4d} "
                      f"CC={r['largest_component']:3d} | {r['move_type']:9s} | "
                      f"max|δ|={r['curvature_max']:.2e} | I={r['I']:.4f}")
        print()
        return self.history

# ============================================================
# ИДЕАЛЬНАЯ АНИМАЦИЯ С ИСПОЛЬЗОВАНИЕМ ВСЕХ СОСТОЯНИЙ
# ============================================================

def create_perfect_animation(sim: RealitySimulation, history: List[Dict], 
                             step=1, fps=5, filename='quantum_breath.gif'):
    """
    Создаёт анимацию, используя сохранённую историю ВСЕХ состояний.
    Теперь каждый кадр — реальное состояние графа.
    """
    states = sim.all_states
    
    # Берём каждый step-й кадр для файла разумного размера
    indices = list(range(0, len(states), step))
    if indices[-1] != len(states) - 1:
        indices.append(len(states) - 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), 
                                    gridspec_kw={'width_ratios': [1.5, 1]})
    
    # Предварительно вычисляем границы для графиков
    all_max_curv = [max(abs(np.array(list(s.curvature_field.values()) or [0]))) 
                    for s in states]
    global_max = max(all_max_curv) if all_max_curv else 1e-6
    
    times = [h['time'] for h in history]
    curv_max_hist = [h['curvature_max'] for h in history]
    curv_std_hist = [h['curvature_std'] for h in history]
    
    current_pos = None

    def animate(frame_idx):
        nonlocal current_pos
        ax1.clear()
        ax2.clear()
        
        idx = indices[frame_idx]
        state = states[idx]
        t = state.time
        
        # ЛЕВЫЙ ГРАФИК: поле кривизны
        ax1.set_facecolor('#1a1a2e')
        
        if state.node_count > 0:
            if frame_idx == 0:
                current_pos = nx.spring_layout(state.graph, k=2, iterations=50, seed=42)
            else:
                try:
                    current_pos = nx.spring_layout(state.graph, k=2, iterations=15, 
                                                   seed=42, pos=current_pos)
                except (nx.NetworkXError, ValueError):
                    current_pos = nx.spring_layout(state.graph, k=2, iterations=50, seed=42)
            
            curvatures = state.curvature_field
            if curvatures:
                node_list = list(state.graph.nodes())
                curv_vals = [curvatures.get(n, 0.0) for n in node_list]
                
                nx.draw_networkx_nodes(
                    state.graph, current_pos, ax=ax1,
                    node_color=curv_vals,
                    cmap='RdBu_r',
                    vmin=-global_max, vmax=global_max,
                    node_size=[20 + 80 * abs(c)/global_max for c in curv_vals],
                    alpha=0.9, edgecolors='white', linewidths=0.3
                )
            
            nx.draw_networkx_edges(state.graph, current_pos, ax=ax1, 
                                   alpha=0.15, width=0.3, edge_color='white')
        
        ax1.set_title(f'Поле кривизны (t={t}, N={state.node_count})', 
                     fontsize=12, fontweight='bold', color='white')
        ax1.axis('off')
        
        # ПРАВЫЙ ГРАФИК: амплитуда ряби
        ax2.set_facecolor('#1a1a2e')
        current_times = times[:min(t, len(times))]
        current_max = curv_max_hist[:min(t, len(times))]
        current_std = curv_std_hist[:min(t, len(times))]
        
        if current_times:
            ax2.fill_between(current_times, 0, current_std, alpha=0.2, color='cyan')
            ax2.plot(current_times, current_max, 'r-', linewidth=1.5, label='max |δ|')
            ax2.plot(current_times, current_std, 'b-', linewidth=1.5, label='σ(δ)')
            
            # INTEGRATE события
            for i, h in enumerate(history[:min(t, len(history))]):
                if h['move_type'] == 'integrate' and i < len(current_max):
                    ax2.scatter(h['time'], current_max[i], c='yellow', s=80, 
                              zorder=5, marker='*', edgecolors='orange', linewidths=0.5)
            
            ax2.axvline(x=t, color='lime', linestyle='--', alpha=0.5, linewidth=1.5)
        
        ax2.set_xlabel('Время', color='white')
        ax2.set_ylabel('Амплитуда', color='white')
        ax2.set_title('Квантовое дыхание', fontsize=12, fontweight='bold', color='white')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(alpha=0.2, color='white')
        ax2.set_yscale('log')
        ax2.tick_params(colors='white')
        
        fig.patch.set_facecolor('#0f0f23')
        fig.suptitle(f'Эволюция реальности: шаг {t}', 
                     fontsize=14, fontweight='bold', color='white', y=1.01)
        
        return [ax1, ax2]
    ani = FuncAnimation(fig, animate, frames=len(indices), 
                       interval=1000/fps, blit=False, repeat=True)
    
    # Сохраняем
    writer = PillowWriter(fps=fps)
    ani.save(filename, writer=writer, dpi=80)
    plt.close()
    
    return ani, filename

# ============================================================
# ФИНАЛЬНЫЙ АНАЛИЗ
# ============================================================

def plot_final_universe(sim: RealitySimulation, history: List[Dict]):
    fig = plt.figure(figsize=(22, 13), facecolor='#0f0f23')
    
    times = [h['time'] for h in history]
    
    # 1. Граф реальности с полем кривизны
    ax1 = fig.add_subplot(2, 3, (1, 2), facecolor='#1a1a2e')
    
    state = sim.state
    if state.node_count > 0:
        pos = nx.spring_layout(state.graph, k=1.5/np.sqrt(state.node_count+1), 
                              iterations=100, seed=42)
        
        curvatures = state.curvature_field
        if curvatures:
            node_list = list(state.graph.nodes())
            curv_vals = [curvatures.get(n, 0.0) for n in node_list]
            vmax = max(max(abs(np.array(curv_vals))), 1e-6)
            
            nodes_draw = nx.draw_networkx_nodes(
                state.graph, pos, ax=ax1,
                node_color=curv_vals,
                cmap='RdBu_r',
                vmin=-vmax, vmax=vmax,
                node_size=[30 + 120 * abs(c)/vmax for c in curv_vals],
                alpha=0.9, edgecolors='white', linewidths=0.3
            )
            cbar = plt.colorbar(nodes_draw, ax=ax1, label='Локальная кривизна δ', shrink=0.8)
            cbar.ax.yaxis.label.set_color('white')
            cbar.ax.tick_params(colors='white')
        
        nx.draw_networkx_edges(state.graph, pos, ax=ax1, 
                              alpha=0.15, width=0.3, edge_color='white')
    
    ax1.set_title(f'ФИНАЛЬНАЯ ВСЕЛЕННАЯ\n'
                 f'{state.node_count} узлов, {state.edge_count} рёбер, '
                 f'{state.triple_count} троек\n'
                 f'Главная компонента: {state.largest_component_size()} '
                 f'({100*state.largest_component_size()/max(1,state.node_count):.1f}%)',
                 fontsize=12, fontweight='bold', color='white')
    ax1.axis('off')
    
    # 2. Рост + связность
    ax2 = fig.add_subplot(2, 3, 3, facecolor='#1a1a2e')
    ax2.plot(times, [h['nodes'] for h in history], 'cyan', label='Узлы', linewidth=2)
    ax2.plot(times, [h['edges'] for h in history], 'magenta', label='Рёбра', linewidth=2)
    ax2.plot(times, [h['triples'] for h in history], 'yellow', label='Тройки', linewidth=2)
    ax2.plot(times, [h['largest_component'] for h in history], 'white', 
            linestyle='--', label='Главная компонента', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Время', color='white')
    ax2.set_ylabel('Количество', color='white')
    ax2.set_title('Рост и связность', color='white', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2, color='white')
    ax2.tick_params(colors='white')
    
    # 3. Квантовое дыхание
    ax3 = fig.add_subplot(2, 3, 4, facecolor='#1a1a2e')
    curv_std = [h['curvature_std'] for h in history]
    curv_max = [h['curvature_max'] for h in history]
    ax3.fill_between(times, 0, curv_std, alpha=0.2, color='cyan')
    ax3.plot(times, curv_max, 'r-', linewidth=2, label='max |δ|')
    ax3.plot(times, curv_std, 'b-', linewidth=2, label='σ(δ)')
    
    integrate_times = [h['time'] for h in history if h['move_type'] == 'integrate']
    integrate_vals = [curv_max[min(t-1, len(curv_max)-1)] for t in integrate_times]
    if integrate_vals:
        ax3.scatter(integrate_times, integrate_vals, c='yellow', s=120, 
                   zorder=5, marker='*', edgecolors='orange', linewidths=0.5,
                   label='Рождение частиц')
    
    ax3.set_xlabel('Время', color='white')
    ax3.set_ylabel('Амплитуда', color='white')
    ax3.set_title('Квантовое дыхание', color='white', fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.2, color='white')
    ax3.set_yscale('log')
    ax3.tick_params(colors='white')
    
    # 4. Интегрированная информация
    ax4 = fig.add_subplot(2, 3, 5, facecolor='#1a1a2e')
    I_vals = [h['I'] for h in history]
    ax4.plot(times, I_vals, 'orange', linewidth=2)
    ax4.fill_between(times, 0, I_vals, alpha=0.15, color='orange')
    ax4.set_xlabel('Время', color='white')
    ax4.set_ylabel('I(G)', color='white')
    ax4.set_title('Интегрированная информация', color='white', fontweight='bold')
    ax4.grid(alpha=0.2, color='white')
    ax4.tick_params(colors='white')
    
    # 5. Накопленная сложность + типы ходов
    ax5 = fig.add_subplot(2, 3, 6, facecolor='#1a1a2e')
    ax5_secondary = ax5.twinx()
    
    ax5.plot(times, [h['U'] for h in history], 'brown', linewidth=2, label='U(G)')
    ax5.set_xlabel('Время', color='white')
    ax5.set_ylabel('Накопленная сложность', color='brown')
    ax5.tick_params(axis='y', colors='brown')
    
    # Распределение ходов (скользящее окно)
    window = 20
    move_colors = {'expand': '#3498db', 'integrate': '#e74c3c', 
                   'connect': '#2ecc71', 'seed': '#f39c12', 'deflate': '#95a5a6'}
    bottom = np.zeros(len(times))
    for move_type in ['expand', 'integrate', 'connect', 'seed', 'deflate']:
        counts = []
        for i in range(len(history)):
            start = max(0, i-window+1)
            window_slice = history[start:i+1]
            count = sum(1 for h in window_slice if h['move_type'] == move_type)
            counts.append(count)
        ax5_secondary.fill_between(times, bottom, bottom + np.array(counts), 
                                   alpha=0.3, color=move_colors[move_type], 
                                   label=move_type)
        bottom = bottom + np.array(counts)
    
    ax5_secondary.set_ylabel('Частота ходов (окно 20)', color='white')
    ax5_secondary.tick_params(axis='y', colors='white')
    ax5_secondary.legend(fontsize=7, loc='upper left')
    
    ax5.set_title('Стрела времени и типы ходов', color='white', fontweight='bold')
    ax5.grid(alpha=0.2, color='white')
    
    plt.suptitle('КВАНТОВАЯ РЯБЬ В СТАБИЛЬНОЙ ГРАФОВОЙ ВСЕЛЕННОЙ', 
                 fontsize=16, fontweight='bold', color='white', y=1.01)
    plt.tight_layout()
    plt.show()

# ============================================================
# ГЛАВНЫЙ ЗАПУСК
# ============================================================

np.random.seed(42)
random.seed(42)

print("""
╔══════════════════════════════════════════════════════════════╗
║     RELATE: КВАНТОВОЕ ДЫХАНИЕ ГРАФОВОЙ РЕАЛЬНОСТИ           ║
║     Полная версия с идеальной анимацией и анализом           ║
╚══════════════════════════════════════════════════════════════╝
""")

# Создаём симуляцию
sim = RealitySimulation(
    alpha=0.3,
    beta=3.0,
    gamma=0.02,
    vacuum=0.8,
    connectivity=3.0
)

# Запускаем
history = sim.run(steps=100, verbose=True)

# Итоговая статистика
print(f"\n{'='*70}")
print(f"РЕЗУЛЬТАТЫ СИМУЛЯЦИИ")
print(f"{'='*70}")
print(f"Финальное состояние:")
print(f"  Узлов:         {sim.state.node_count}")
print(f"  Рёбер:         {sim.state.edge_count}")
print(f"  Троек:         {sim.state.triple_count}")
print(f"  Главная комп.: {sim.state.largest_component_size()} "
      f"({100*sim.state.largest_component_size()/max(1,sim.state.node_count):.1f}%)")
print(f"\nСтатистика событий:")
for mt in ['expand', 'integrate', 'connect', 'seed', 'deflate']:
    count = sum(1 for h in history if h['move_type'] == mt)
    print(f"  {mt:12s}: {count}")
print(f"\nКвантовая рябь:")
print(f"  Макс. амплитуда: {max(h['curvature_max'] for h in history):.4e}")
print(f"  Средняя ампл.:   {np.mean([h['curvature_std'] for h in history]):.4e}")
print(f"  Событий INTEGRATE (рождений частиц): {sum(1 for h in history if h['move_type']=='integrate')}")
print(f"\nИнформация:")
print(f"  Финальная I(G): {history[-1]['I']:.4f}")
print(f"  Накопленная сложность U(G): {history[-1]['U']:.1f} байт")
print(f"{'='*70}")

# Финальный график
print("\nПостроение финального анализа...")
plot_final_universe(sim, history)

# СОЗДАНИЕ ИДЕАЛЬНОЙ АНИМАЦИИ
print("\nСоздание анимации квантового дыхания...")
print("(это может занять около минуты)")

ani, filename = create_perfect_animation(
    sim, history, 
    step=max(1, len(sim.all_states)//60),  # ~60 кадров
    fps=6,
    filename='quantum_breath.gif'
)

print(f"\n✅ Анимация сохранена как '{filename}'")
print(f"   Размер: {len(sim.all_states)} состояний, {len(sim.all_states)//max(1, len(sim.all_states)//60)} кадров")

# Показываем анимацию в Jupyter
try:
    from IPython.display import Image as IPImage
    display(IPImage(filename))
except:
    print("Для просмотра анимации откройте файл quantum_breath.gif")
