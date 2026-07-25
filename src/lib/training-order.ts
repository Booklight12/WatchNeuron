export function createTrainingSeed() {
  try {
    const seed = new Uint32Array(1);
    crypto.getRandomValues(seed);
    return seed[0];
  } catch {
    return (Date.now() ^ Math.floor(performance.now() * 1000)) >>> 0;
  }
}

export function shuffleForNextEpoch<T>(
  values: T[],
  random: () => number,
) {
  if (values.length < 2) return values;
  const previous = values.slice();
  for (let cursor = values.length - 1; cursor > 0; cursor--) {
    const swap = Math.floor(random() * (cursor + 1));
    [values[cursor], values[swap]] = [values[swap], values[cursor]];
  }
  if (values.every((value, index) => value === previous[index])) {
    const offset = 1 + Math.floor(random() * (values.length - 1));
    values.push(...values.splice(0, offset));
  }
  return values;
}
