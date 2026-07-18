/**
 * Detect intentional /sorge slash commands in PR comments.
 * Ignores markdown quotes and code so "Quote reply" does not re-dispatch.
 */

export function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Strip blockquotes + code so only author-authored prose/commands remain. */
export function stripNonCommandMarkdown(body) {
  let text = String(body);
  // Fenced code blocks (``` or ~~~)
  text = text.replace(/```[\s\S]*?```/g, "\n");
  text = text.replace(/~~~[\s\S]*?~~~/g, "\n");
  // Inline code (including `/sorge` in bot guidance)
  text = text.replace(/`[^`\n]*`/g, " ");
  // Markdown blockquote lines (Quote reply / nested >>)
  text = text
    .split("\n")
    .filter((line) => !/^\s{0,3}>/.test(line))
    .join("\n");
  return text;
}

/**
 * True when a PR comment body contains an intentional /sorge slash command.
 * Requires the command at the start of a non-quoted, non-code line.
 */
export function hasSorgeSlashCommand(body, extraLogin) {
  if (!body || typeof body !== "string") return false;

  const handles = ["sorge"];
  if (extraLogin) {
    handles.push(String(extraLogin).replace(/^\//, "").replace(/^@/, ""));
  }

  const actionable = stripNonCommandMarkdown(body);

  return handles.some((handle) => {
    const escaped = escapeRegex(handle);
    // Intentional slash command: beginning of a line (optional leading whitespace).
    const pattern = new RegExp(`(?:^|\\n)\\s*/${escaped}(?:-[\\w-]+)?\\b`, "i");
    return pattern.test(actionable);
  });
}
