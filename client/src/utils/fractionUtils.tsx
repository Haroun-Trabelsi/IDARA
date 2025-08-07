function toFraction(decimal: number, maxDenominator = 1000): string {
    if (decimal < 0.05) return "0";
  const tolerance = 1.0E-6;
  let h1 = 1, h2 = 0;
  let k1 = 0, k2 = 1;
  let b = decimal;

  do {
    const a = Math.floor(b);
    const aux = h1;
    h1 = a * h1 + h2;
    h2 = aux;
    const auxK = k1;
    k1 = a * k1 + k2;
    k2 = auxK;
    b = 1 / (b - a);
  } while (Math.abs(decimal - h1 / k1) > decimal * tolerance && k1 < maxDenominator);

  const numerator = h1;
  const denominator = k1;

  // Convert to mixed number if improper
  if (numerator > denominator) {
    const whole = Math.floor(numerator / denominator);
    const rem = numerator % denominator;
    return rem === 0 ? `${whole}` : `${whole} ${rem}/${denominator}`;
  } else {
    return `${numerator}/${denominator}`;
  }
}
export { toFraction };