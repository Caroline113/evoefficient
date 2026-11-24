---
name: browser-grounding
description: Browser actions whose element IDs or visible state may change after navigation.
---
Use current screenshot and observation identifiers for the next browser action.
After navigation, use the new observation instead of reusing stale element IDs.
If a goal references an image, compare the relevant visual attributes before selecting it.
Before stopping, inspect whether all requested changes are visible in the current state.
Keep the official action format and do not infer success from an action merely being accepted.
