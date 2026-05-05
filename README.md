# kickscore

[![Build package](https://github.com/lucasmaystre/kickscore/actions/workflows/build.yml/badge.svg)](https://github.com/lucasmaystre/kickscore/actions/workflows/build.yml)
[![Code coverage](https://codecov.io/gh/lucasmaystre/kickscore/branch/master/graph/badge.svg)](https://codecov.io/gh/lucasmaystre/kickscore)

`kickscore` is the dynamic skill rating system powering [Kickoff.ai](https://kickoff.ai/).

In short, `kickscore` can be used to understand & visualize the skill of players (or teams) competing in pairwise matches, and to predict outcomes of future matches. It extends the [Elo rating system](https://en.wikipedia.org/wiki/Elo_rating_system) and [TrueSkill](https://en.wikipedia.org/wiki/TrueSkill).

![evolution of NBA teams\' skill over history](https://lum-public.s3-eu-west-1.amazonaws.com/kickscore-nba-history.svg)

## Getting started

To install the latest release directly from PyPI, simply type:

    pip install kickscore

To get started, you might want to explore one of these notebooks:

- [Basic example illustrating the API](examples/kickscore-basics.ipynb) ([interactive version](https://colab.research.google.com/github/lucasmaystre/kickscore/blob/master/examples/kickscore-basics.ipynb))
- [Visualizing the history of the NBA](examples/nba-history.ipynb) ([interactive version](https://colab.research.google.com/github/lucasmaystre/kickscore/blob/master/examples/nba-history.ipynb))

## Choice-set observations

For top-1 Plackett-Luce / conditional-logit data, use `ChoiceModel`. Each observation
contains a choice set and the selected winner; inference uses the KL/CVI updates.

```python
import kickscore as ks

model = ks.ChoiceModel(num_samples=128, random_state=0)
kernel = ks.kernel.Matern52(var=1.0, lscale=10.0)

for competitor in ["A", "B", "C"]:
    model.add_item(competitor, kernel=kernel)

model.observe(choices=["A", "B", "C"], winner="A", t=1.0)
model.fit()  # ChoiceModel defaults to method="kl".
probs = model.probabilities(["A", "B", "C"], t=2.0)
```

## References

- Lucas Maystre, Victor Kristof, Matthias Grossglauser, [Pairwise Comparisons with Flexible Time-Dynamics](https://arxiv.org/abs/1903.07746), KDD 2019
