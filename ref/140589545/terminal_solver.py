"""Bounded, terminal-only search for the public Hydrapple policy.

Only visible, deterministic resources may enter a winning certificate. No
value-head estimates, opponent turns, coin outcomes, or newly drawn cards
are used to override the policy. This is deliberately incomplete.
"""
from collections import Counter, deque
import ctypes
import itertools
import json
import time

from cg.api import all_card_data
from cg.sim import lib


def fingerprint(obs):
    c, s = obs['current'], obs['select']
    # Search and live observations differ in round/win annotations and clocks.
    # Unknown opposing hand identities are never part of this certificate.
    state = {k: c.get(k) for k in ('turn', 'yourIndex', 'firstPlayer',
             'supporterPlayed', 'stadiumPlayed', 'energyAttached', 'retreated',
             'stadium', 'looking')}
    you = c['yourIndex']
    state['players'] = []
    for i, p in enumerate(c['players']):
        d = {k: p.get(k) for k in ('active', 'bench', 'discard', 'deckCount',
             'handCount', 'benchMax', 'poisoned', 'burned', 'asleep',
             'paralyzed', 'confused')}
        d['prizeCount'] = len(p['prize'])
        if i == you:
            d['hand'] = p.get('hand')
        state['players'].append(d)
    return json.dumps((s, state), sort_keys=True, separators=(',', ':'))


class TerminalSolver:
    def __init__(self, policy, stats, max_nodes=700, budget_s=.035, depth=9):
        self.policy, self.stats = policy, stats
        self.max_nodes, self.budget_s, self.depth = max_nodes, budget_s, depth
        self.cards = {c.cardId: c for c in all_card_data()}
        self.ptr = None
        self.pending = []

    def reset(self):
        self.pending = []

    def _raw(self, ptr):
        r = json.loads(ctypes.string_at(ptr).decode())
        if r['error']:
            raise RuntimeError('Search API error: '+str(r['error']))
        return r['state']

    def _root(self, obs):
        if self.ptr is None:
            self.ptr = lib.AgentStart()
        det, _ = self.policy._determinize(obs)
        sbi = obs['search_begin_input']
        arr = lambda xs: (ctypes.c_int * len(xs))(*xs)
        return self._raw(lib.SearchBegin(self.ptr, sbi.encode('ascii'), len(sbi),
                         *(arr(xs) for xs in det), 1))  # expose every coin

    def _step(self, state, action):
        return self._raw(lib.SearchStep(self.ptr, state['searchId'],
                        (ctypes.c_int * len(action))(*action), len(action)))

    @staticmethod
    def known_hand_resources(obs, you):
        p = obs['current']['players'][you]
        visible = list(p.get('hand') or []) + list(p.get('discard') or [])
        for pk in p['active'] + p['bench']:
            if pk:
                visible.extend(pk.get('energyCards') or [])
                visible.extend(pk.get('preEvolution') or [])
                visible.extend(pk.get('tools') or [])
                visible.append(pk)
        return {(c['serial'], c['id']) for c in visible if c}

    def _main_actions(self, obs):
        me = obs['current']['players'][obs['current']['yourIndex']]
        ans, seen = [], set()
        for i, op in enumerate(obs['select']['option']):
            ot = op['type']
            if ot == 13:
                key = (ot, op['attackId'])
            elif ot in (8, 9):
                zone = me['hand'] if op.get('area') == 2 else None
                if zone is None:
                    continue
                source = zone[op['index']]['id']
                key = (ot, source, op.get('inPlayArea'), op.get('inPlayIndex'))
            elif ot == 12:
                key = (ot,)
            elif ot == 7:
                cid = me['hand'][op['index']]['id']
                # Boss, Night Stretcher, Lana's Aid: all resources are visible.
                if cid not in (1182, 1097, 1184):
                    continue
                key = (ot, cid)
            elif ot == 10:
                zone = me['active'] if op.get('area') == 4 else me['bench']
                cid = zone[op['index']]['id']
                if cid != 150:  # Ripening Charge, no card draw
                    continue
                key = (ot, op.get('area'), op.get('index'))
            else:
                continue
            if key not in seen:
                seen.add(key)
                ans.append([i])
        # Try finishing attacks before enabling actions; breadth-first search
        # finds shorter certificates and avoids a large low-value subtree.
        return sorted(ans, key=lambda a: obs['select']['option'][a[0]]['type'] != 13)

    def _actions(self, obs):
        s = obs['select']
        n, lo, hi = len(s['option']), s['minCount'], s['maxCount']
        if s['context'] == 0:
            return self._main_actions(obs)
        if s['context'] == 46:  # COIN_HEAD; reject every stochastic line
            return []
        if lo == hi == n:
            return [list(range(n))]
        # Drawing a prize cannot affect whether the final prize wins. Any
        # non-terminal reveal of an unknown card is pruned below.
        if s['context'] == 7 and all(o.get('area') == 6 for o in s['option']):
            return [list(range(lo))]
        # Do not inspect/search any hidden deck, including a now-revealed one.
        if s.get('deck') or any(o.get('area') == 1 for o in s['option']):
            return []
        result = []
        for k in range(lo, hi + 1):
            for choice in itertools.combinations(range(n), k):
                result.append(list(choice))
                if len(result) >= 48:
                    return result
        return result

    def choose(self, obs):
        if self.pending:
            sig, action = self.pending.pop(0)
            if fingerprint(obs) == sig:
                self.stats['certificate_steps'] += 1
                return action
            self.pending = []
            self.stats['certificate_mismatch'] += 1
        s, cur = obs['select'], obs['current']
        if s['context'] != 0 or not obs.get('search_begin_input'):
            return None
        you = cur['yourIndex']
        them = cur['players'][1-you]
        board = [p for p in them['active'] + them['bench'] if p]
        if not board:
            return None
        prize_max = max(3 if self.cards[p['id']].megaEx else
                        2 if self.cards[p['id']].ex else 1 for p in board)
        if len(cur['players'][you]['prize']) > prize_max and len(board) > 1:
            return None
        if obs.get('remainingOverageTime', 600) < 45:
            return None
        self.stats['searches'] += 1
        deadline = time.perf_counter() + self.budget_s
        known = self.known_hand_resources(obs, you)
        queue, visited, nodes = deque(), set(), 0
        try:
            root = self._root(obs)
            # No override if reconstruction changes the visible decision.
            if fingerprint(root['observation']) != fingerprint(obs):
                self.stats['root_mismatch'] += 1
                return None
            queue.append((root, []))
            while queue and nodes < self.max_nodes and time.perf_counter() < deadline:
                state, path = queue.popleft()
                ob = state['observation']
                c = ob['current']
                if c['result'] == you:
                    self.pending = path[1:]
                    self.stats['certificates'] += 1
                    self.stats['nodes'] += nodes
                    return path[0][1]
                if c['result'] != -1 or c['yourIndex'] != you or c['turn'] != cur['turn']:
                    continue
                if len(path) >= self.depth:
                    continue
                # Sampled future draws cannot become resources for this line.
                if any((p['serial'], p['id']) not in known
                       for p in c['players'][you].get('hand') or []):
                    continue
                sig = fingerprint(ob)
                if sig in visited:
                    continue
                visited.add(sig)
                for action in self._actions(ob):
                    if nodes >= self.max_nodes or time.perf_counter() >= deadline:
                        break
                    try:
                        child = self._step(state, action)
                    except RuntimeError:
                        self.stats['rejected_search_actions'] += 1
                        continue
                    nodes += 1
                    queue.append((child, path + [(sig, action)]))
            self.stats['nodes'] += nodes
        except Exception:
            self.stats['errors'] += 1
        finally:
            if self.ptr:
                lib.SearchEnd(self.ptr)
        return None
