"""Interactive Video QA grounded in Orion perception outputs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console

try:
    import ollama
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "VideoQASystem requires the 'ollama' Python package. Install with 'pip install ollama'."
    ) from exc


console = Console()
DEFAULT_SYSTEM_PROMPT = (
    "You are Orion's video QA assistant. Use the provided structured context to answer "
    "questions about the analyzed video. If the context does not contain the answer, say you "
    "do not know and suggest running the perception pipeline again."
)


@dataclass
class VideoQASystem:
    """Interactive Q&A session that pulls context from Orion results directories."""

    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    llm_model: str = "gemma3:4b"
    system_prompt: Optional[str] = None
    results_dir: Optional[str] = None
    extra_entities_path: Optional[str] = None
    context_frames: int = 200
    max_objects: int = 20
    max_relations: int = 10
    history_turns: int = 4

    _context_block: str = field(init=False, default="")
    _base_messages: List[Dict[str, str]] = field(init=False, default_factory=list)
    _history: List[Dict[str, str]] = field(init=False, default_factory=list)
    _object_lookup: Dict[str, str] = field(init=False, default_factory=dict)
    _extra_entities: List[Dict[str, object]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.results_path = Path(self.results_dir).expanduser() if self.results_dir else None
        self.extra_entities_path = self._resolve_extra_entities_path(self.extra_entities_path)
        self._extra_entities = self._load_extra_entities()
        self._context_block = self._build_context()
        prompt = self.system_prompt or DEFAULT_SYSTEM_PROMPT
        if self._context_block:
            label = self.results_path.name if self.results_path else "unknown video"
            prompt = f"{prompt}\n\nVideo context ({label}):\n{self._context_block}"
        else:
            prompt = (
                f"{prompt}\n\nNo structured video context was available. "
                "Answer generically and encourage the user to provide a results directory."
            )
        self._base_messages = [{"role": "system", "content": prompt}]
        self._history = []

    def start_interactive_session(self) -> None:
        if self.results_path and self._context_block:
            console.print(
                f"[dim]Context loaded from {self.results_path} (showing up to {self.max_objects} objects "
                f"and {self.max_relations} relation summaries).[/dim]"
            )
        else:
            console.print(
                "[yellow]Warning: no results directory supplied; answers will be generic.[/yellow]"
            )

        console.print("[green]Video Q&A mode ready.[/green]")
        console.print("[dim]Type 'exit' or 'quit' to leave.[/dim]\n")

        while True:
            try:
                question = console.input("[bold cyan]Q> [/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Session terminated by user.[/dim]")
                break

            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                console.print("[dim]Goodbye.[/dim]")
                break

            answer = self._query_llm(question)
            if answer:
                console.print(f"[bold yellow]A>[/bold yellow] {answer}\n")

    # ------------------------------------------------------------------
    # Context loading helpers
    # ------------------------------------------------------------------
    def _build_context(self) -> str:
        if not self.results_path:
            return ""
        if not self.results_path.exists():
            console.print(
                f"[yellow]Results directory {self.results_path} does not exist. Skipping context.[/yellow]"
            )
            return ""

        sections: List[str] = []
        try:
            objects = self._load_objects()
            if objects:
                sections.append("Objects observed:")
                for entry in objects[: self.max_objects]:
                    sections.append(self._format_object_entry(entry))
            else:
                sections.append("Objects observed: none found (run perception pipeline).")

            relation_lines = self._load_relations()
            if relation_lines:
                sections.append("\nRepresentative relations:")
                sections.extend(relation_lines)
            else:
                sections.append("\nNo scene graph relations were available in this results directory.")
        except Exception as exc:  # pragma: no cover - defensive logging
            console.print(
                f"[yellow]Warning: failed to load video context from {self.results_path}: {exc}[/yellow]"
            )
            return ""

        return "\n".join(sections)

    def _load_objects(self) -> List[Dict[str, object]]:
        if not self.results_path:
            return []
        memory_path = self.results_path / "memory.json"
        entities_path = self.results_path / "entities.json"
        entries: List[Dict[str, object]] = []

        if memory_path.exists():
            data = json.loads(memory_path.read_text(encoding="utf-8"))
            for obj in data.get("objects", []):
                entry = {
                    "label": obj.get("memory_id"),
                    "class": obj.get("class", "object"),
                    "first": obj.get("first_seen_frame"),
                    "last": obj.get("last_seen_frame"),
                    "observations": obj.get("total_observations", 0),
                    "description": (obj.get("metadata") or {}).get("description")
                    if isinstance(obj.get("metadata"), dict)
                    else None,
                }
                entries.append(entry)
        elif entities_path.exists():
            data = json.loads(entities_path.read_text(encoding="utf-8"))
            for ent in data.get("entities", []):
                entry = {
                    "label": f"entity_{ent.get('id')}",
                    "class": ent.get("class", "object"),
                    "first": ent.get("first_frame"),
                    "last": ent.get("last_frame"),
                    "observations": ent.get("observation_count", 0),
                    "description": ent.get("description"),
                }
                entries.append(entry)

        entries.sort(key=lambda item: item.get("observations", 0) or 0, reverse=True)
        self._object_lookup = {
            item["label"]: item.get("class", "object")
            for item in entries
            if item.get("label")
        }
        if entries and self._extra_entities:
            self._enrich_with_extra(entries)
        return entries

    def _enrich_with_extra(self, entries: List[Dict[str, object]]) -> None:
        for entry in entries:
            if entry.get("description"):
                continue
            match = self._match_extra_entry(entry)
            if match and match.get("description"):
                entry["description"] = match["description"]

    def _match_extra_entry(self, entry: Dict[str, object]) -> Optional[Dict[str, object]]:
        target_class = entry.get("class")
        if not target_class:
            return None
        first = entry.get("first")
        last = entry.get("last")
        best: Optional[Dict[str, object]] = None
        best_score = float("-inf")
        for extra in self._extra_entities:
            if extra.get("class") != target_class:
                continue
            score = self._range_overlap_score(first, last, extra.get("first_frame"), extra.get("last_frame"))
            if score > best_score:
                best = extra
                best_score = score
        return best

    @staticmethod
    def _range_overlap_score(
        base_start: Optional[int],
        base_end: Optional[int],
        other_start: Optional[int],
        other_end: Optional[int],
    ) -> float:
        if base_start is None or base_end is None or other_start is None or other_end is None:
            # fall back to loose matching
            base_mid = ((base_start or 0) + (base_end or 0)) / 2.0
            other_mid = ((other_start or 0) + (other_end or 0)) / 2.0
            return -abs(base_mid - other_mid)
        overlap = min(base_end, other_end) - max(base_start, other_start)
        if overlap >= 0:
            return overlap
        # prefer closer time ranges if no overlap
        base_mid = 0.5 * (base_start + base_end)
        other_mid = 0.5 * (other_start + other_end)
        return -abs(base_mid - other_mid)

    def _load_relations(self) -> List[str]:
        if not self.results_path:
            return []
        graph_path = self.results_path / "scene_graph.jsonl"
        if not graph_path.exists():
            return []

        combos: Counter = Counter()
        sample_frame: Dict[str, int] = {}
        with graph_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if self.context_frames and idx >= self.context_frames:
                    break
                line = line.strip()
                if not line:
                    continue
                graph = json.loads(line)
                nodes = {
                    node.get("memory_id"): node.get("class", "object")
                    for node in graph.get("nodes", [])
                }
                frame_id = graph.get("frame")
                for edge in graph.get("edges", []):
                    subj_id = edge.get("subject")
                    obj_id = edge.get("object")
                    predicate = edge.get("relation", "related_to")
                    subj_cls = self._object_lookup.get(subj_id) or nodes.get(subj_id) or subj_id
                    obj_cls = self._object_lookup.get(obj_id) or nodes.get(obj_id) or obj_id
                    key = f"{subj_cls} {predicate} {obj_cls}"
                    combos[key] += 1
                    sample_frame.setdefault(key, frame_id)

        if not combos:
            return []

        lines: List[str] = []
        for relation, count in combos.most_common(self.max_relations):
            frame = sample_frame.get(relation)
            relation_text = relation.replace("_", " ")
            detail = f"~{count} frame(s)"
            if frame is not None:
                detail += f", e.g., frame {frame}"
            lines.append(f"- {relation_text} ({detail})")
        return lines

    @staticmethod
    def _format_object_entry(entry: Dict[str, object]) -> str:
        label = entry.get("label", "object")
        cls_name = entry.get("class", "object")
        first = entry.get("first")
        last = entry.get("last")
        observations = entry.get("observations") or 0
        desc = entry.get("description")
        summary = f"- {label} ({cls_name}), seen frames {first}-{last}, {observations} detections"
        if desc:
            summary += f". Description: {desc}"
        return summary

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------
    def _query_llm(self, question: str) -> Optional[str]:
        messages = list(self._base_messages)
        if self.history_turns > 0 and self._history:
            messages.extend(self._history[-(self.history_turns * 2):])
        messages.append({"role": "user", "content": question})

        try:
            response = ollama.chat(model=self.llm_model, messages=messages)
        except Exception as exc:
            console.print(f"[red]Ollama request failed: {exc}[/red]")
            return None

        answer = response.get("message", {}).get("content", "").strip()
        if not answer:
            console.print("[yellow]No response received from Ollama.[/yellow]")
            return None

        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})
        return answer

    # ------------------------------------------------------------------
    # Extra entities helpers
    # ------------------------------------------------------------------
    def _resolve_extra_entities_path(self, explicit: Optional[str]) -> Optional[Path]:
        if explicit:
            path = Path(explicit).expanduser()
            return path if path.exists() else None
        default_path = Path("data/testing/entities.json")
        return default_path if default_path.exists() else None

    def _load_extra_entities(self) -> List[Dict[str, object]]:
        if not self.extra_entities_path:
            return []
        try:
            data = json.loads(self.extra_entities_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - safety
            console.print(f"[yellow]Warning: failed to read {self.extra_entities_path}: {exc}[/yellow]")
            return []
        entities = data.get("entities")
        if isinstance(entities, list):
            return entities
        return []
