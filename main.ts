import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const HOUR_MS = 3_600_000;
const DAY_VARIANCE_FACTOR = 76;
const DRAWS = 200_000;
const SMOOTHING = 0.5;
const SEED = 20_261_109;
const MICRO = 1_000_000;
const MIX_CORE = 0.92;
const MIX_WIDE = 2.6;
const PARTIAL_DAY_WEIGHT = 0.9;
const WEEKEND_VOL_FACTOR = 1.1;

const PRIORS: Record<string, number> = {
  AAPLX: 0.0139,
  AMZNX: 0.0195,
  GOOGLX: 0.0225,
  METAX: 0.0276,
  MSFTX: 0.0168,
  MSTRX: 0.0559,
  NVDAX: 0.0254,
  SPCXX: 0.0361,
  TSLAX: 0.0295,
};

interface Day {
  date: string;
  previous: string;
  twap_until: string;
}

interface Task {
  places: number;
  output: string;
  bin_edges: number[];
  window: { days: Day[] };
  universe: Array<{ symbol: string; file: string }>;
}

interface Bar {
  t: number;
  h: string;
  l: string;
  c: string;
}

interface DayForecast {
  date: string;
  mean: number[];
  sd: number[];
}

interface SizeRow {
  symbol: string;
  bin: number;
  p: number;
}

interface ProbabilityRow {
  symbol: string;
  place: number;
  p: number;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
const symbols = task.universe.map((u) => u.symbol);
const names = symbols.length;
const places = task.places;
const edges = task.bin_edges;
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));

function readBars(file: string): Bar[] {
  return readFileSync(join(INPUT_DIR, file), 'utf8')
    .split('\n')
    .filter((line) => line !== '')
    .map((line) => JSON.parse(line) as Bar)
    .sort((a, b) => a.t - b.t);
}

function dailyVol(file: string): number {
  const bars = readBars(file);
  const returns: number[] = [];
  for (let i = 1; i < bars.length; i++) {
    if (bars[i]!.t - bars[i - 1]!.t === HOUR_MS) {
      returns.push(Math.log(Number(bars[i]!.c) / Number(bars[i - 1]!.c)));
    }
  }

  const closeVariance = returns.length >= 2
    ? returns.reduce((sum, r) => sum + r * r, 0) / returns.length
    : null;

  let parkinsonSum = 0;
  let parkinsonCount = 0;
  for (const bar of bars) {
    const high = Number(bar.h);
    const low = Number(bar.l);
    if (high > low && low > 0) {
      const range = Math.log(high / low);
      parkinsonSum += (range * range) / (4 * Math.LN2);
      parkinsonCount += 1;
    }
  }
  const parkinson = parkinsonCount > 0 ? parkinsonSum / parkinsonCount : null;

  let hourly: number;
  if (closeVariance !== null && parkinson !== null) hourly = 0.5 * closeVariance + 0.5 * parkinson;
  else if (closeVariance !== null) hourly = closeVariance;
  else if (parkinson !== null) hourly = parkinson;
  else return 0.02;

  return clamp(Math.sqrt(24 * 0.92 * hourly), 0.005, 0.5);
}

function forecastVolumes(sampleVols: number[]): number[] {
  const priors = symbols.map((symbol) => PRIORS[symbol] ?? 0.02);
  const d = sampleVols.map((vol, i) => 2 * Math.log(vol / priors[i]!));
  const common = d.reduce((sum, value) => sum + value, 0) / names;
  return priors.map((prior, i) => clamp(
    prior * Math.exp(0.5 * clamp(0.65 * common + 0.30 * (d[i]! - common), -1.5, 1.5)),
    0.005,
    0.5,
  ));
}

/** erfc(x), Numerical Recipes' erfcc (fractional error under 1.2e-7 everywhere). */
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}

const phi = (x: number): number => 0.5 * erfc(-x / Math.SQRT2);

/** mulberry32: a 32-bit generator, uniform on [0, 1). */
function uniforms(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normals by the polar Box-Muller method over `next`. */
function normals(next: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u: number;
    let v: number;
    let s: number;
    do {
      u = 2 * next() - 1;
      v = 2 * next() - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const factor = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * factor;
    return u * factor;
  };
}

const micro = (p: number): number => Math.floor(p * MICRO);

function dayForecasts(forecastVols: number[]): DayForecast[] {
  const bars = task.universe.map((u) => readBars(u.file));
  return task.window.days.map((day, dayIndex) => {
    const mean = new Array<number>(names).fill(0);
    const sd = forecastVols.slice();

    if (dayIndex === 0) {
      const closeTime = Date.parse(day.twap_until) - 24 * HOUR_MS;
      const closeBar = bars.map((history) => history.find((bar) => bar.t === closeTime));
      for (let i = 0; i < names; i++) {
        const last = bars[i]![bars[i]!.length - 1];
        if (closeBar[i] !== undefined && last !== undefined && last.t > closeTime) {
          mean[i] = PARTIAL_DAY_WEIGHT * Math.log(Number(last.c) / Number(closeBar[i]!.c));
          const knownHours = clamp(Math.round((last.t - closeTime) / HOUR_MS), 0, 24);
          sd[i] = forecastVols[i]! * Math.sqrt(Math.max(0, 1 - knownHours / DAY_VARIANCE_FACTOR));
        }
      }
    }

    if (Date.parse(`${day.date}T00:00:00Z`) - Date.parse(`${day.previous}T00:00:00Z`) > 24 * HOUR_MS + 1000) {
      for (let i = 0; i < names; i++) sd[i] = sd[i]! * WEEKEND_VOL_FACTOR;
    }

    return { date: day.date, mean, sd };
  });
}

function sizeRows(forecast: DayForecast): SizeRow[] {
  const rows: SizeRow[] = [];
  symbols.forEach((symbol, i) => {
    const mean = forecast.mean[i]!;
    const sd = forecast.sd[i]!;
    for (let bin = 0; bin <= edges.length; bin++) {
      const lo = bin === 0 ? -Infinity : edges[bin - 1]!;
      const hi = bin === edges.length ? Infinity : edges[bin]!;
      const l = lo === -Infinity ? -Infinity : Math.log1p(lo);
      const h = hi === Infinity ? Infinity : Math.log1p(hi);
      const core = phi((h - mean) / sd) - phi((l - mean) / sd);
      const wide = phi((h - mean) / (MIX_WIDE * sd)) - phi((l - mean) / (MIX_WIDE * sd));
      const units = micro(MIX_CORE * core + (1 - MIX_CORE) * wide);
      if (units > 0) rows.push({ symbol, bin, p: units / MICRO });
    }
  });
  return rows;
}

function orderForecast(forecasts: DayForecast[]): Array<{ probabilities: ProbabilityRow[]; order: string[] }> {
  const component = uniforms(SEED ^ 0x9e3779b9);
  const gauss = normals(uniforms(SEED));
  return forecasts.map((forecast) => {
    const counts = Array.from({ length: places }, () => new Array<number>(names).fill(0));
    const draw = new Array<number>(names).fill(0);
    const index = symbols.map((_, i) => i);

    // Draw order: for each draw, for each name one uniform for the component
    // choice, then one normal from the Box-Muller generator. Values are
    // mean + sd * scale * z, sorted descending with ties by ascending index.
    for (let d = 0; d < DRAWS; d++) {
      for (let i = 0; i < names; i++) {
        const scale = component() < 1 - MIX_CORE ? MIX_WIDE : 1;
        draw[i] = forecast.mean[i]! + forecast.sd[i]! * scale * gauss();
      }
      index.sort((a, b) => draw[b]! - draw[a]! || a - b);
      for (let k = 0; k < places; k++) counts[k]![index[k]!]! += 1;
    }

    const units = counts.map((row) => row.map((count) => micro((count + SMOOTHING) / (DRAWS + SMOOTHING * names))));
    const probabilities: ProbabilityRow[] = [];
    units.forEach((row, k) => row.forEach((unit, i) => {
      probabilities.push({ symbol: symbols[i]!, place: k + 1, p: unit / MICRO });
    }));

    let order: string[];
    if (names === places) {
      const size = 1 << names;
      const dp = new Array<number>(size).fill(Infinity);
      const choice = new Array<number>(size).fill(-1);
      dp[0] = 0;
      for (let mask = 0; mask < size; mask++) {
        if (dp[mask] === Infinity) continue;
        let placed = 0;
        for (let bits = mask; bits !== 0; bits >>>= 1) placed += bits & 1;
        for (let i = 0; i < names; i++) {
          if ((mask & (1 << i)) !== 0) continue;
          const expected = counts.reduce(
            (sum, row, j) => sum + Math.abs(placed - j) * row[i]! / DRAWS,
            0,
          );
          const nextMask = mask | (1 << i);
          const candidate = dp[mask]! + expected;
          if (candidate < dp[nextMask]! - 1e-12) {
            dp[nextMask] = candidate;
            choice[nextMask] = i;
          }
        }
      }
      order = new Array<string>(names);
      for (let mask = size - 1, placed = names - 1; mask !== 0; placed--) {
        const i = choice[mask]!;
        order[placed] = symbols[i]!;
        mask ^= 1 << i;
      }
    } else {
      const expectedPlace = symbols.map((_, i) => counts.reduce((sum, row, j) => sum + (j + 1) * row[i]!, 0) / DRAWS);
      order = symbols.map((_, i) => i)
        .sort((a, b) => expectedPlace[a]! - expectedPlace[b]! || a - b)
        .slice(0, places)
        .map((i) => symbols[i]!);
    }
    return { probabilities, order };
  });
}

const sampleVols = task.universe.map((u) => dailyVol(u.file));
const forecastVols = forecastVolumes(sampleVols);
const forecasts = dayForecasts(forecastVols);
const orders = orderForecast(forecasts);
const days = forecasts.map((forecast, i) => ({
  date: forecast.date,
  probabilities: orders[i]!.probabilities,
  order: orders[i]!.order,
  sizes: sizeRows(forecast),
}));

writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ days })}\n`);
