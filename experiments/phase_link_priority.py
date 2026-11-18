"""Experimental phase-link ordering; frozen detector and eligibility unchanged.

No new rejection threshold. Full signed-phase support, duration, existing delay
and preceding arrival order rank competing links. This is not a wave classifier.
"""
from unittest.mock import patch
import numpy as np
from uterine_wave import radial_events as baseline


def phase_similarity(a, b, filtered):
    """Cosine of peak-aligned full phase templates, including unequal tails.

    Zero outside a phase defines its comparison template, not imputed motion.
    Native samples only; there is no time scaling or optimal-shift search.
    """
    lo = min(a['first']-a['peak_index'], b['first']-b['peak_index'])
    hi = max(a['last']-a['peak_index'], b['last']-b['peak_index'])+1
    templates = []
    for node in (a, b):
        values = filtered[node['first']:node['last']+1, node['column']]
        if not np.isfinite(values).all():
            raise ValueError('A phase template must remain in one finite run')
        vector = np.zeros(hi-lo)
        start = node['first']-node['peak_index']-lo
        vector[start:start+len(values)] = values
        templates.append(vector)
    x, y = templates
    norm = np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.clip(x@y/norm, 0, 1)) if norm > 0 else 0.


def prioritize_ridges(columns, time_s, space, filtered, diagnostics):
    nodes = []
    for column in columns:
        for node in column:
            node['id'] = len(nodes)
            nodes.append(node)
    parent = list(range(len(nodes)))
    predecessor = np.full(len(nodes), -1, int)
    dt = float(np.median(np.diff(time_s)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def order_agreement(a, b):
        slopes = []
        i = a['id']
        while predecessor[i] >= 0:
            previous = nodes[predecessor[i]]
            node = nodes[i]
            slopes.append((node['peak_time_s']-previous['peak_time_s']) /
                          (space[node['column']]-space[previous['column']]))
            i = predecessor[i]
        if not slopes:
            return 1.
        spacing = space[b['column']]-space[a['column']]
        slope = (b['peak_time_s']-a['peak_time_s'])/spacing
        return float(1/(1+abs(slope-np.median(slopes))/(np.median(abs(np.asarray(slopes)))+dt/spacing)))

    selected_links = []
    eligible_links = 0
    changed_boundaries = 0
    baseline_selected = 0
    baseline_competing = 0
    for left, right in zip(columns[:-1], columns[1:]):
        choices = []
        for a in left:
            for b in right:
                delay = abs(a['peak_time_s']-b['peak_time_s'])
                wa, wb = a['end_s']-a['start_s'], b['end_s']-b['start_s']
                scale = .5*(wa+wb)
                overlap = min(a['end_s'], b['end_s'])-max(a['start_s'], b['start_s'])
                if a['sign'] != b['sign'] or delay > scale or overlap < 0:
                    continue
                shape = phase_similarity(a, b, filtered)
                duration = (min(wa, wb)+dt)/(max(wa, wb)+dt)
                order = order_agreement(a, b)
                proximity = 1/(1+delay/(scale+dt))
                # Equal geometric combination, one predeclared variant; no cutoff.
                priority = float((shape*duration*order*proximity)**.25)
                choices.append(dict(left=a['id'], right=b['id'], delay=delay,
                                    shape=shape, duration=duration, order=order,
                                    priority=priority))
        eligible_links += len(choices)
        def assign(ordered):
            used_left, used_right, chosen = set(), set(), []
            for edge in ordered:
                a, b = edge['left'], edge['right']
                if a in used_left or b in used_right:
                    continue
                chosen.append(edge)
                used_left.add(a); used_right.add(b)
            return chosen
        old = assign(sorted(choices, key=lambda e:(e['delay'], e['left'], e['right'])))
        chosen = assign(sorted(choices, key=lambda e:(-e['priority'], e['delay'], e['left'], e['right'])))
        baseline_selected += len(old)
        baseline_competing += sum(any(
            (other['left'] == edge['left'] and other['right'] != edge['right']) or
            (other['right'] == edge['right'] and other['left'] != edge['left'])
            for other in choices) for edge in old)
        changed_boundaries += int({(e['left'], e['right']) for e in old} !=
                                  {(e['left'], e['right']) for e in chosen})
        for edge in chosen:
            a, b = edge['left'], edge['right']
            parent[find(a)] = find(b)
            predecessor[b] = a
        selected_links.extend(chosen)
    groups = {}
    for node in nodes:
        groups.setdefault(find(node['id']), []).append(node)
    ridges, short_paths = [], []
    for group in groups.values():
        group = sorted(group, key=lambda n:n['column'])
        if len(group) < 3:
            short_paths.append([dict(column=n['column'], peak_time_s=n['peak_time_s'], sign=n['sign'])
                                for n in group])
            continue
        ids = np.array([node['column'] for node in group])
        arrival = np.array([node['peak_time_s'] for node in group])
        ridges.append(dict(id=len(ridges), sign=group[0]['sign'], nodes=group,
                           fit=baseline._fit(space[ids], arrival), center_s=float(np.median(arrival))))
    diagnostics.update(eligible_links=eligible_links, selected_links=selected_links,
                       changed_spatial_boundaries=changed_boundaries,
                       baseline_selected_link_count=baseline_selected,
                       baseline_competing_link_count=baseline_competing,
                       short_phase_paths=short_paths, no_new_rejection_threshold=True)
    return ridges


def detect_priority(signal, valid, time_s, space, *, noise_floor, config=None):
    # Context-scoped replacement for a paired experiment, never a persistent hook.
    filtered, _ = baseline._lobes(np.asarray(signal, float),
                                  np.asarray(valid, bool)&np.isfinite(signal),
                                  np.asarray(time_s, float),
                                  np.broadcast_to(np.asarray(noise_floor, float), (len(space),)))
    diagnostics = {}
    def builder(columns, t, s):
        return prioritize_ridges(columns, t, s, filtered, diagnostics)
    with patch.object(baseline, '_ridges', builder):
        report = baseline.detect_radial_events(signal, valid, time_s, space,
                                               noise_floor=noise_floor, config=config)
    np.testing.assert_array_equal(report['filtered_A'], filtered)
    report['candidate_algorithm'] = 'experimental_full_phase_priority_v1'
    report['link_priority_diagnostics'] = diagnostics
    return report
