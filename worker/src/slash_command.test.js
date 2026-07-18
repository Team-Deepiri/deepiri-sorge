import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { hasSorgeSlashCommand, stripNonCommandMarkdown } from "./slash_command.js";

describe("hasSorgeSlashCommand", () => {
  it("accepts a bare /sorge comment", () => {
    assert.equal(hasSorgeSlashCommand("/sorge"), true);
    assert.equal(hasSorgeSlashCommand("  /sorge --force\n"), true);
  });

  it("accepts /sorge on its own line after other prose", () => {
    assert.equal(hasSorgeSlashCommand("please review\n/sorge\n"), true);
  });

  it("ignores Quote reply blockquotes that mention /sorge", () => {
    const body = [
      "> deepiri-sorge[bot] commented:",
      "> /sorge",
      ">",
      "> **Model:** openai/gpt-oss-120b (groq)",
      "",
      "thanks, that looks good",
    ].join("\n");
    assert.equal(hasSorgeSlashCommand(body), false);
  });

  it("ignores nested quote lines", () => {
    assert.equal(hasSorgeSlashCommand(">> /sorge\n\nok"), false);
  });

  it("still fires when a real /sorge follows a quote", () => {
    const body = ["> previous /sorge run", "", "/sorge --force"].join("\n");
    assert.equal(hasSorgeSlashCommand(body), true);
  });

  it("ignores inline code mentions of /sorge", () => {
    assert.equal(hasSorgeSlashCommand("re-run `/sorge` when ready"), false);
  });

  it("ignores fenced code blocks", () => {
    const body = ["```", "/sorge", "```", "just documenting"].join("\n");
    assert.equal(hasSorgeSlashCommand(body), false);
  });

  it("does not match mid-sentence /sorge without a line start", () => {
    assert.equal(hasSorgeSlashCommand("can you /sorge this later?"), false);
  });

  it("supports extra bot login handles", () => {
    assert.equal(hasSorgeSlashCommand("/deepiri-sorge", "deepiri-sorge"), true);
  });
});

describe("stripNonCommandMarkdown", () => {
  it("drops quote lines and keeps author text", () => {
    const out = stripNonCommandMarkdown("> /sorge\n\nhello");
    assert.match(out, /hello/);
    assert.doesNotMatch(out, /\/sorge/);
  });
});
