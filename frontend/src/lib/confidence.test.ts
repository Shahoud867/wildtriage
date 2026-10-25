import { describe, expect, it } from "vitest";

import { confidenceLevel, formatPercent, reasonText } from "./confidence";

describe("confidence helpers", () => {
  it("maps calibrated probabilities to levels", () => {
    expect(confidenceLevel(0.95)).toBe("high");
    expect(confidenceLevel(0.9)).toBe("high");
    expect(confidenceLevel(0.75)).toBe("medium");
    expect(confidenceLevel(0.2)).toBe("low");
  });

  it("formats percentages and missing values", () => {
    expect(formatPercent(0.876)).toBe("88%");
    expect(formatPercent(null)).toBe("–");
  });

  it("explains review reasons", () => {
    expect(reasonText("tiny_object")).toBe("Tiny animal");
    expect(reasonText("something_new")).toBe("something new");
  });
});
