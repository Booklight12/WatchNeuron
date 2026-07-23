import { createRequire } from "node:module";
import { writeFile } from "node:fs/promises";

const require = createRequire(import.meta.url);
const mnist = require("mnist");

const TRAINING_PER_DIGIT = 400;
const VALIDATION_PER_DIGIT = 100;
const samples = [];

for (let digit = 0; digit < 10; digit++) {
  const digitSamples = mnist[digit].range(
    0,
    TRAINING_PER_DIGIT + VALIDATION_PER_DIGIT - 1,
  );
  digitSamples.forEach((pixels, index) => {
    const activePixels = [];
    pixels.forEach((value, pixelIndex) => {
      if (value > 0.008) {
        activePixels.push([pixelIndex, Math.round(value * 255)]);
      }
    });
    samples.push({
      split: index < TRAINING_PER_DIGIT ? 0 : 1,
      digit,
      activePixels,
    });
  });
}

const byteLength =
  12 +
  samples.reduce(
    (total, sample) => total + 4 + sample.activePixels.length * 3,
    0,
  );
const buffer = Buffer.allocUnsafe(byteLength);
buffer.write("WNDS", 0, 4, "ascii");
buffer.writeUInt16LE(1, 4);
buffer.writeUInt32LE(samples.length, 6);
buffer.writeUInt16LE(0, 10);

let offset = 12;
for (const sample of samples) {
  buffer.writeUInt8(sample.split, offset);
  buffer.writeUInt8(sample.digit, offset + 1);
  buffer.writeUInt16LE(sample.activePixels.length, offset + 2);
  offset += 4;
  for (const [pixelIndex, value] of sample.activePixels) {
    buffer.writeUInt16LE(pixelIndex, offset);
    buffer.writeUInt8(value, offset + 2);
    offset += 3;
  }
}

await writeFile("public/mnist-training.bin", buffer);
console.log(
  `wrote public/mnist-training.bin (${TRAINING_PER_DIGIT * 10} training, ${VALIDATION_PER_DIGIT * 10} validation, ${(buffer.length / 1024 / 1024).toFixed(2)} MiB)`,
);
