"""Parse and analyze PR diffs"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FileChange:
    path: str
    status: str
    additions: int = 0
    deletions: int = 0
    hunks: List[str] = field(default_factory=list)


@dataclass
class ParsedDiff:
    raw: str
    files: List[str] = field(default_factory=list)
    file_changes: Dict[str, FileChange] = field(default_factory=dict)
    lines_added: int = 0
    lines_deleted: int = 0
    files_changed: int = 0
    language_counts: Dict[str, int] = field(default_factory=dict)
    
    def get_language(self, filepath: str) -> Optional[str]:
        ext = filepath.split(".")[-1].lower() if "." in filepath else ""
        lang_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "jsx": "javascript",
            "tsx": "typescript",
            "java": "java",
            "go": "go",
            "rs": "rust",
            "rb": "ruby",
            "php": "php",
            "cs": "csharp",
            "cpp": "cpp",
            "c": "c",
            "h": "c",
            "hpp": "cpp",
            "swift": "swift",
            "kt": "kotlin",
            "scala": "scala",
            "r": "r",
            "lua": "lua",
            "sh": "shell",
            "bash": "shell",
            "zsh": "shell",
            "yml": "yaml",
            "yaml": "yaml",
            "json": "json",
            "toml": "toml",
            "xml": "xml",
            "html": "html",
            "css": "css",
            "scss": "scss",
            "sql": "sql",
            "md": "markdown",
            "rst": "rst",
        }
        return lang_map.get(ext)
    
    def get_summary(self) -> str:
        return (
            f"{self.files_changed} file(s) changed, "
            f"{self.lines_added} additions, "
            f"{self.lines_deleted} deletions"
        )


class DiffParser:
    
    DIFF_HEADER_PATTERN = re.compile(
        r"diff --git a/(.+) b/(.+)\n"
        r"([^@]+@@\s+[+-]?\d+(?:,\d+)?\s+[+-]?\d+(?:,\d+)?\s+@@.*\n)?"
    )
    
    LINE_ADD_PATTERN = re.compile(r"^\+(?!\+\+)(.+)$")
    LINE_DEL_PATTERN = re.compile(r"^-(?!--)(.+)$")
    HUNK_HEADER_PATTERN = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    
    def parse(self, diff_content: str) -> ParsedDiff:
        if not diff_content.strip():
            return ParsedDiff(
                raw=diff_content,
                files=[],
                file_changes={},
                lines_added=0,
                lines_deleted=0,
                files_changed=0
            )
        
        files: List[str] = []
        file_changes: Dict[str, FileChange] = {}
        lines_added = 0
        lines_deleted = 0
        language_counts: Dict[str, int] = {}
        
        current_file: Optional[str] = None
        current_change: Optional[FileChange] = None
        in_diff = False
        
        for line in diff_content.split("\n"):
            if line.startswith("diff --git"):
                parts = line.split(" ")
                if len(parts) >= 4:
                    a_path = parts[2][2:]
                    b_path = parts[3]
                    
                    if b_path.startswith("b/"):
                        clean_path = b_path[2:]
                    else:
                        clean_path = b_path
                    
                    current_file = clean_path
                    files.append(clean_path)
                    
                    status = self._get_file_status(b_path)
                    current_change = FileChange(
                        path=clean_path,
                        status=status,
                    )
                    file_changes[clean_path] = current_change
                    in_diff = True
                    
                    lang = self._detect_language(clean_path)
                    if lang:
                        language_counts[lang] = language_counts.get(lang, 0) + 1
            
            elif line.startswith("@@"):
                in_diff = True
                if current_change is not None:
                    current_change.hunks.append(line)
            
            elif in_diff and current_change:
                if line.startswith("+") and not line.startswith("++"):
                    current_change.additions += 1
                    lines_added += 1
                elif line.startswith("-") and not line.startswith("--"):
                    current_change.deletions += 1
                    lines_deleted += 1
        
        return ParsedDiff(
            raw=diff_content,
            files=files,
            file_changes=file_changes,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            files_changed=len(files),
            language_counts=language_counts
        )
    
    def _get_file_status(self, path: str) -> str:
        if path.startswith("a/") or path.startswith("b/"):
            path = path[2:]
        
        if "/dev/null" in path:
            return "added"
        
        return "modified"
    
    def _detect_language(self, filepath: str) -> Optional[str]:
        ext = filepath.split(".")[-1].lower() if "." in filepath else ""
        
        lang_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "jsx": "javascript",
            "tsx": "typescript",
            "java": "java",
            "go": "go",
            "rs": "rust",
            "rb": "ruby",
            "php": "php",
            "cs": "csharp",
            "cpp": "cpp",
            "c": "c",
            "h": "c",
            "hpp": "cpp",
            "swift": "swift",
            "kt": "kotlin",
            "scala": "scala",
            "lua": "lua",
            "sh": "shell",
            "bash": "shell",
            "yml": "yaml",
            "yaml": "yaml",
            "json": "json",
            "toml": "toml",
            "xml": "xml",
            "html": "html",
            "css": "css",
            "sql": "sql",
            "md": "markdown",
        }
        
        return lang_map.get(ext)
    
    def parse_from_github(self, diff_data: List[Dict]) -> ParsedDiff:
        raw_lines = []
        files = []
        file_changes = {}
        lines_added = 0
        lines_deleted = 0
        language_counts: Dict[str, int] = {}
        
        for file_data in diff_data:
            filename = file_data.get("filename", "")
            status = file_data.get("status", "modified")
            additions = file_data.get("additions", 0)
            deletions = file_data.get("deletions", 0)
            patch = file_data.get("patch", "")
            
            if patch:
                raw_lines.append(f"diff --git a/{filename} b/{filename}")
                raw_lines.append(patch)
            
            files.append(filename)
            file_changes[filename] = FileChange(
                path=filename,
                status=status,
                additions=additions,
                deletions=deletions
            )
            
            lines_added += additions
            lines_deleted += deletions
            
            lang = self._detect_language(filename)
            if lang:
                language_counts[lang] = language_counts.get(lang, 0) + 1
        
        return ParsedDiff(
            raw="\n".join(raw_lines),
            files=files,
            file_changes=file_changes,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            files_changed=len(files),
            language_counts=language_counts
        )
