"""BO3-level summaries; observations/rounds are not independent samples."""
import math
import numpy as np
import pandas as pd


def interval(wins, n):
    z = 1.959963984540054
    p = wins/n
    den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-radius, center+radius


def summarize(records):
    rows = []
    for (candidate, opponent), group in pd.DataFrame(records).groupby(['candidate','opponent']):
        wins = sum(row['winner'] == row['seat'] for row in group.to_dict('records'))
        draws = int((group.winner == 2).sum())
        errors = sum(bool(row['error']) or any(row['fallback_errors']) for row in group.to_dict('records'))
        lo, hi = interval(wins, len(group))
        rows.append(dict(candidate=candidate, opponent=opponent, matches=len(group),
                         wins=wins, draws=draws, score=group.score.mean(), win_rate=wins/len(group),
                         ci_low=lo, ci_high=hi, errors=errors,
                         max_think_s=max(max(x) for x in group.think_s)))
    return pd.DataFrame(rows)


def seat_order(records):
    rows = []
    for match in records:
        for game in match['rounds']:
            rows.append(dict(candidate=match['candidate'], opponent=match['opponent'],
                             seat=match['seat'], went_first=game['first_player']==match['seat'],
                             win=game['winner']==match['seat']))
    return pd.DataFrame(rows).groupby(['candidate','seat','went_first']).agg(
        games=('win','size'), wins=('win','sum'), win_rate=('win','mean')).reset_index()


def delta_ci(records, selected):
    df = pd.DataFrame(records)
    rng = np.random.default_rng(20261016)
    bootstrap = np.zeros(3000)
    strata = list(df.groupby(['opponent','seat']))
    for _, group in strata:
        old = group[group.candidate=='previous_rule'].score.to_numpy()
        new = group[group.candidate==selected].score.to_numpy()
        bootstrap += (rng.choice(new,(3000,len(new))).mean(axis=1)
                      -rng.choice(old,(3000,len(old))).mean(axis=1))/len(strata)
    means = df.groupby('candidate').score.mean()
    return dict(previous_score=float(means['previous_rule']), selected_score=float(means[selected]),
                delta=float(means[selected]-means['previous_rule']),
                ci_low=float(np.quantile(bootstrap,.025)), ci_high=float(np.quantile(bootstrap,.975)),
                unit='BO3 match', paired=False)
