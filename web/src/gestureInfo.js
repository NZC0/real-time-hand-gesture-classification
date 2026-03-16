/**
 * Metadata for each gesture class.
 * Displayed in the UI overlay when a gesture is detected.
 */

/** @typedef {{ name: string, emoji: string, cultures: string[], description: string, color: string }} GestureInfo */

/** @type {Record<string, GestureInfo>} */
export const GESTURE_INFO = {
  middle_finger: {
    name: "Middle Finger",
    emoji: "🖕",
    cultures: ["USA", "Europe", "Most of the world"],
    description:
      "The universal sign of disrespect. Traced back to ancient Greece, this gesture has been offending people for over 2500 years — impressively efficient.",
    color: "#ff4757",
  },
  reversed_v: {
    name: "Reversed V",
    emoji: "✌️",
    cultures: ["UK", "Australia", "Ireland", "New Zealand"],
    description:
      'The "V for Victory" sign — but only if the palm faces outward. Flip it around (back of hand toward the person) and it becomes deeply offensive in British culture. Churchill knew this.',
    color: "#ff6b81",
  },
  thumbs_up: {
    name: "Thumbs Up",
    emoji: "👍",
    cultures: ["Iran", "Iraq", "Afghanistan", "Parts of West Africa"],
    description:
      'In the Middle East and parts of Africa, the thumbs-up is equivalent to the middle finger in Western culture. Not "like", more like "sit on this".',
    color: "#ffa502",
  },
  corna: {
    name: "Corna (Il Cornuto)",
    emoji: "🤘",
    cultures: ["Italy", "Spain", "Greece", "Latin America"],
    description:
      'The "horns" gesture in Mediterranean culture implies that someone\'s partner is unfaithful — you\'re calling them a cuckold. In metal concerts, same gesture means something completely different.',
    color: "#eccc68",
  },
  crossed_fingers: {
    name: "Crossed Fingers",
    emoji: "🤞",
    cultures: ["Vietnam"],
    description:
      'In Vietnam, crossed fingers (majeur sur index) resemble female genitalia and are considered very vulgar. In the West it just means "good luck" — context is everything.',
    color: "#7bed9f",
  },
  ok_sign: {
    name: "OK Sign",
    emoji: "👌",
    cultures: ["Brazil", "Turkey", "Parts of the Mediterranean"],
    description:
      'In Brazil and Turkey, the OK circle formed by thumb and index finger is a crude insult (it represents a body orifice). In the USA it means everything is fine — or 3 in underwater diving.',
    color: "#70a1ff",
  },
  neutral: {
    name: "Neutral",
    emoji: "✋",
    cultures: [],
    description:
      "No offensive gesture detected. Your hands appear to be on their best behaviour.",
    color: "#747d8c",
  },
};

/**
 * Returns gesture info for a class name, falling back to neutral.
 * @param {string} className
 * @returns {GestureInfo}
 */
export function getGestureInfo(className) {
  return GESTURE_INFO[className] ?? GESTURE_INFO["neutral"];
}

/** Ordered list of class names — must match model output indices. */
export const CLASS_NAMES = [
  "middle_finger",
  "reversed_v",
  "thumbs_up",
  "corna",
  "crossed_fingers",
  "ok_sign",
  "neutral",
];
