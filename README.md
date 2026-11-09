# xStocks Movers Forecast

This repository contains a Node 22 forecast for make.fun's `xstocks-movers-forecast-01` competition.

The model estimates each name's daily volatility from close-to-close and Parkinson range components, shrinks those estimates toward the competition's quoted 60-day volatilities, and produces per-day log-return moments. The first window day also uses the hours already observed between its previous close and the history cutoff. Size probabilities use a 92% normal / 8% wide-normal mixture over 42 return bins. Order probabilities come from 200,000 seeded Monte Carlo draws, and the order guess uses an exact bitmask dynamic program when the number of names equals the number of places.

The program reads `task.json` and the bars files from `$MAKEFUN_INPUT_DIR` (default `/input`) and writes the requested output file to `$MAKEFUN_OUTPUT_DIR` (default `/output`).

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=/tmp/xstocks/xstocks-movers-forecast-01-kit/input \
MAKEFUN_OUTPUT_DIR=out \
node main.ts
```

It uses only the Node 22 standard library, with no network access and no installed dependencies.
