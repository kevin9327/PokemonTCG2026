"""Own BO3 arena: separate agent processes, masked observations, measured clocks.

Official C++ RNG is unseeded: comparisons are independent, never seed-paired.
Persistent workers reset via the actual deck-request ABI before EVERY match.
Fresh-worker preflight additionally checks the submission's cold-start path.
"""
import contextlib
import copy
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from pathlib import Path

CONFIG = dict(bo=3, version=2, actTimeout=0, runTimeout=6000, episodeSteps=10000000)


def worker_main(folder, connection, log_path):
    log = open(log_path, 'a', encoding='utf-8', buffering=1)
    sys.stdout = sys.stderr = log
    os.environ['PTCG_DEBUG'] = '1'
    os.environ.pop('QCOL_RIVDECK', None)
    os.chdir(folder)
    sys.path.insert(0, folder)
    ns = {}
    try:
        started = time.perf_counter()
        source = Path('main.py').read_text(encoding='utf-8-sig')
        # Kaggle's loader deliberately supplies neither __file__ nor __name__.
        exec(compile(source, str(Path(folder)/'main.py'), 'exec'), ns)
        fn = [value for value in ns.values() if callable(value)][-1]
        connection.send(dict(ready=True, entry=getattr(fn, '__name__', '?'),
                             load_s=time.perf_counter()-started))
    except Exception:
        connection.send(dict(ready=False, error=traceback.format_exc()))
        return
    while True:
        message = connection.recv()
        if message is None:
            return
        try:
            if 'seed' in message:
                random.seed(message['seed'])
            obs = message['observation']
            log.flush()
            before_position = log.tell()
            t0 = time.perf_counter()
            argc = getattr(getattr(fn, '__code__', None), 'co_argcount', 1)
            args = [obs, dict(CONFIG, __raw_path__=str(Path(folder)/'main.py'))][:argc]
            result = fn(*args)
            elapsed = time.perf_counter()-t0
            log.flush()
            # Detect exceptions swallowed by a public agent's random fallback.
            with open(log_path, encoding='utf-8', errors='replace') as reader:
                reader.seek(before_position)
                new_log = reader.read()
            connection.send(dict(action=result, elapsed=elapsed,
                                 fallback_errors=new_log.count('Traceback (most recent call last)'),
                                 guard=copy.deepcopy(ns.get('GUARD', {}))))
        except Exception:
            connection.send(dict(error=traceback.format_exc()))


class AgentProcess:
    def __init__(self, name, folder, log_dir):
        self.name = name
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = log_dir/(name+'.log')
        ctx = mp.get_context('spawn')
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(target=worker_main, args=(str(folder), child, str(self.log)))
        self.process.start()
        if not self.connection.poll(180):
            raise TimeoutError('Agent load timeout: '+name)
        self.ready = self.connection.recv()
        if not self.ready.get('ready'):
            raise RuntimeError(name+': '+self.ready.get('error', 'load failed'))

    def call(self, observation, seed=None):
        message = dict(observation=observation)
        if seed is not None:
            message['seed'] = seed
        self.connection.send(message)
        if not self.connection.poll(60):
            raise TimeoutError('Decision exceeded 60 s: '+self.name)
        reply = self.connection.recv()
        if reply.get('error'):
            raise RuntimeError(self.name+': '+reply['error'])
        return reply

    def close(self):
        if self.process.is_alive():
            self.connection.send(None)
            self.process.join(5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()


def valid(selection, action):
    return (isinstance(action, list) and len(set(action)) == len(action)
            and selection['minCount'] <= len(action) <= selection['maxCount']
            and all(type(i) is int and 0 <= i < len(selection['option']) for i in action))


def play_match(left, right, match_id, output, trace=False):
    from cg.game import battle_start, battle_select, battle_finish
    agents = [left, right]
    remaining = [600.-agent.ready['load_s'] for agent in agents]
    fallback_errors = [0, 0]
    guard_end = [{}, {}]
    action_ms = [[], []]
    wins, rounds = [0, 0], []
    error, steps = None, 0
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    import gzip
    stream = gzip.open(output/f'trace-{match_id}.jsonl.gz', 'wt', encoding='utf-8') if trace else None
    try:
        decks = []
        for i, agent in enumerate(agents):
            reply = agent.call(dict(select=None, current=None, logs=[], search_begin_input=None,
                                    player=i, remainingOverageTime=remaining[i]), 20261016+match_id*2+i)
            remaining[i] -= reply['elapsed']
            deck = reply['action']
            if len(deck) != 60 or not all(type(cid) is int for cid in deck):
                raise ValueError('Invalid deck returned by '+agent.name)
            decks.append(deck)
        for round_index in range(3):
            alive = False
            try:
                obs, status = battle_start(*decks, reverse_player=bool(round_index % 2))
                alive = bool(status.battlePtr)
                if not alive or status.errorPlayer >= 0:
                    raise ValueError(f'Engine deck error {status.errorPlayer}/{status.errorType}')
                for turn_step in range(4000):
                    cur = obs['current']
                    if cur['result'] >= 0:
                        break
                    cur['round'] = round_index+1
                    for i in range(2):
                        cur['players'][i]['win'] = wins[i]
                    player = cur['yourIndex']
                    obs['player'], obs['step'] = player, steps
                    obs['remainingOverageTime'] = remaining[player]
                    reply = agents[player].call(obs)
                    remaining[player] -= reply['elapsed']
                    fallback_errors[player] += reply['fallback_errors']
                    guard_end[player] = reply['guard']
                    action_ms[player].append(reply['elapsed']*1000)
                    action = reply['action']
                    if remaining[player] <= 0:
                        raise TimeoutError(f'600-second budget exceeded by {agents[player].name}')
                    if not valid(obs['select'], action):
                        raise ValueError(f'Illegal action by {agents[player].name}: {action}')
                    if stream:
                        stream.write(json.dumps(dict(round=round_index+1, player=player,
                                                     observation=obs, action=action))+'\n')
                    obs = battle_select(action)
                    steps += 1
                else:
                    raise TimeoutError('4000-decision cap; classified as failure, never as draw')
                result = obs['current']['result']
                rounds.append(dict(round=round_index+1, winner=result,
                                   first_player=obs['current']['firstPlayer'], turn=obs['current']['turn']))
                if result in (0, 1):
                    wins[result] += 1
            finally:
                if alive:
                    battle_finish()
            if max(wins) > min(wins) + 3-len(rounds):
                break
    except Exception:
        error = traceback.format_exc()[-1800:]
    finally:
        if stream:
            stream.close()
    return dict(match_id=match_id, left=left.name, right=right.name, wins=wins,
                rounds=rounds, winner=None if error else 0 if wins[0]>wins[1] else 1 if wins[1]>wins[0] else 2,
                error=error, fallback_errors=fallback_errors, guard_end=guard_end,
                think_s=[600-x for x in remaining], cold_load_s=[a.ready['load_s'] for a in agents],
                max_action_ms=[max(x, default=0) for x in action_ms],
                decisions=steps, elapsed_s=time.perf_counter()-started)


def run_panel(processes, candidates, opponents, n, output, split, start_id):
    assert n % 2 == 0
    records = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/(split+'.jsonl')).open('w', encoding='utf-8') as stream:
        for block in range(n):
            for opponent in opponents:
                for candidate in candidates:
                    seat = block % 2
                    names = [candidate, opponent] if seat == 0 else [opponent, candidate]
                    record = play_match(processes[names[0]], processes[names[1]], start_id+len(records),
                                        output, trace=(block == 0))
                    winner = record['winner']
                    record.update(candidate=candidate, opponent=opponent, seat=seat, split=split,
                                  score=0. if winner is None else .5 if winner == 2 else float(winner == seat))
                    records.append(record)
                    stream.write(json.dumps(record)+'\n')
                    stream.flush()
                    if record['error']:
                        print('MATCH ERROR:', candidate, opponent, record['error'][-600:], flush=True)
            if block in (0, n-1) or (block+1) % 8 == 0:
                print(f'{split}: {block+1}/{n} blocks, {len(records)} BO3 matches', flush=True)
    return records
