import json
from pathlib import Path


CONFIG_OLD = "# Link settings.\nLINK_MAX_DISTANCE_UM = 7.0"
CONFIG_NEW = """# Link settings.
LINK_MAX_DISTANCE_UM = 7.0
ENABLE_GLOBAL_FLOW = True
FLOW_CONFIDENT_DISTANCE_UM = 4.0
"""

LINK_OLD = """def link_nodes(nodes: list[dict]) -> list[tuple[int, int]]:
    by_t = defaultdict(list)
    for node in nodes:
        by_t[node['t']].append(node)

    edges = []
    times = sorted(by_t)
    for t0, t1 in zip(times[:-1], times[1:]):
        if t1 != t0 + 1:
            continue
        prev = by_t[t0]
        curr = by_t[t1]
        if not prev or not curr:
            continue

        prev_xyz = np.asarray([[n['z'], n['y'], n['x']] for n in prev], dtype=np.float32) * SCALE_ZYX
        curr_xyz = np.asarray([[n['z'], n['y'], n['x']] for n in curr], dtype=np.float32) * SCALE_ZYX
        dist = np.linalg.norm(prev_xyz[:, None, :] - curr_xyz[None, :, :], axis=2)
        cost = dist.copy()
        cost[cost > LINK_MAX_DISTANCE_UM] = 1e6
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if dist[r, c] <= LINK_MAX_DISTANCE_UM:
                edges.append((prev[r]['node_id'], curr[c]['node_id']))
    return edges
"""

LINK_NEW = """def solve_frame_links(
    prev_xyz: np.ndarray,
    curr_xyz: np.ndarray,
    max_distance_um: float,
    flow_um: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    if flow_um is None:
        flow_um = np.zeros(3, dtype=np.float32)
    predicted_xyz = prev_xyz + flow_um
    residual = np.linalg.norm(predicted_xyz[:, None, :] - curr_xyz[None, :, :], axis=2)
    cost = residual.copy()
    cost[cost > max_distance_um] = 1e6
    row_ind, col_ind = linear_sum_assignment(cost)

    links = []
    for r, c in zip(row_ind, col_ind):
        residual_distance = float(residual[r, c])
        if residual_distance <= max_distance_um:
            links.append((int(r), int(c), residual_distance))
    return links


def estimate_global_flow_um(prev_xyz: np.ndarray, curr_xyz: np.ndarray) -> np.ndarray:
    initial_links = solve_frame_links(prev_xyz, curr_xyz, LINK_MAX_DISTANCE_UM)
    displacements = []
    for r, c, _ in initial_links:
        actual_distance = float(np.linalg.norm(curr_xyz[c] - prev_xyz[r]))
        if actual_distance <= FLOW_CONFIDENT_DISTANCE_UM:
            displacements.append(curr_xyz[c] - prev_xyz[r])
    if not displacements:
        return np.zeros(3, dtype=np.float32)
    return np.median(np.asarray(displacements, dtype=np.float32), axis=0).astype(np.float32)


def link_nodes(nodes: list[dict]) -> list[tuple[int, int]]:
    by_t = defaultdict(list)
    for node in nodes:
        by_t[node['t']].append(node)

    edges = []
    times = sorted(by_t)
    for t0, t1 in zip(times[:-1], times[1:]):
        if t1 != t0 + 1:
            continue
        prev = by_t[t0]
        curr = by_t[t1]
        if not prev or not curr:
            continue

        prev_xyz = np.asarray([[n['z'], n['y'], n['x']] for n in prev], dtype=np.float32) * SCALE_ZYX
        curr_xyz = np.asarray([[n['z'], n['y'], n['x']] for n in curr], dtype=np.float32) * SCALE_ZYX
        flow_um = estimate_global_flow_um(prev_xyz, curr_xyz) if ENABLE_GLOBAL_FLOW else np.zeros(3, dtype=np.float32)
        links = solve_frame_links(prev_xyz, curr_xyz, LINK_MAX_DISTANCE_UM, flow_um=flow_um)
        for r, c, _ in links:
            edges.append((prev[r]['node_id'], curr[c]['node_id']))
    return edges
"""


def patch_notebook(path: Path) -> None:
    nb = json.loads(path.read_text())
    changed = False
    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        if CONFIG_OLD in source and "ENABLE_GLOBAL_FLOW" not in source:
            source = source.replace(CONFIG_OLD, CONFIG_NEW)
            changed = True
        if LINK_OLD in source:
            source = source.replace(LINK_OLD, LINK_NEW)
            changed = True
        elif LINK_OLD.rstrip() in source:
            source = source.replace(LINK_OLD.rstrip(), LINK_NEW.rstrip())
            changed = True
        cell["source"] = source.splitlines(keepends=True)
    if changed:
        path.write_text(json.dumps(nb, indent=1))
        print(f"patched {path}")
    else:
        print(f"no changes needed for {path}")


for notebook in [
    Path("biohub_classical_submission.ipynb"),
    Path("biohub_classical_submission_fast.ipynb"),
]:
    patch_notebook(notebook)
