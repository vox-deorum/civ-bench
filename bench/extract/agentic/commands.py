"""Read a workspace shell command without running it.

A strategist with a workspace runs shell commands such as
``record=/workspace/records/turn-12; cat "$record/options.md" | head -40``. This
module finds which paths such a command reads and writes, and how it reads them.
It is a heuristic parser: it expands simple variables, ``for`` lists, and brace
lists, then walks each simple command and treats only the file arguments of known
commands as paths. Anything it cannot follow (a computed path, an unbalanced
quote) is skipped rather than guessed.
"""

from __future__ import annotations

import itertools
import posixpath
import re
import shlex
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Set

# Commands whose positional arguments are files they read in full.
FULL_READERS = {"cat", "less", "more", "nl", "tac", "bat"}
# Commands that read part of a file, or filter a stream down.
FILTERS = {"grep", "egrep", "fgrep", "rg", "sed", "awk", "jq", "head", "tail", "cut", "sort", "uniq", "diff", "wc"}
# Commands that list or probe paths without reading their contents.
LISTERS = {"ls", "find", "tree", "du", "stat", "file", "test", "[", "[[", "realpath", "readlink"}
# Filters whose first positional argument is a pattern or program, not a file.
PATTERN_FIRST = {"grep", "egrep", "fgrep", "rg", "sed", "awk", "jq"}
# Options that take the next token as their value.
OPTION_VALUES = {
    "grep": {"-A", "-B", "-C", "-m", "-e", "-f", "--max-count", "--regexp", "--file"},
    "egrep": {"-A", "-B", "-C", "-m", "-e", "-f"},
    "fgrep": {"-A", "-B", "-C", "-m", "-e", "-f"},
    "rg": {"-A", "-B", "-C", "-m", "-e", "-f", "-g", "-t", "--glob", "--type"},
    "sed": {"-e", "-f", "--expression", "--file"},
    "awk": {"-F", "-v", "-f"},
    "jq": {"--arg", "--argjson"},
    "head": {"-n", "-c"},
    "tail": {"-n", "-c"},
    "cut": {"-d", "-f", "-c", "-b"},
    "sort": {"-k", "-t", "-o"},
    "xargs": {"-n", "-I", "-d", "-L", "-P"},
    "tee": set(),
}
# Options that supply the pattern/program, so the first positional is a file.
PATTERN_OPTIONS = {"-e", "-f", "--regexp", "--file", "--expression"}
# Commands that write their positional arguments.
WRITE_ALL = {"mkdir", "touch", "rm", "rmdir", "tee"}
# Commands whose last positional argument is the destination.
WRITE_LAST = {"cp", "mv", "ln", "install"}
# Shell words that open or close a block; the command word follows them.
BLOCK_WORDS = {"then", "do", "else", "elif", "if", "while", "until", "!", "{", "}", "fi", "done", "esac", "time"}
SEPARATORS = {";", "&&", "||", "|", "&", "(", ")", ";;", "|&"}
PIPE = {"|", "|&"}

MAX_VARIANTS = 64
MAX_RANGE = 64

_HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n(.*?)(?:\n\s*\2\s*(?:\n|$)|$)", re.S)
_ASSIGN = re.compile(r"(?:^|(?<=[\s;&|(]))(?:export\s+|local\s+)?([A-Za-z_]\w*)=(\"[^\"]*\"|'[^']*'|[^\s;&|)]*)")
_FOR = re.compile(r"\bfor\s+([A-Za-z_]\w*)\s+in\s+([^;\n]*?)\s*(?:;|\n)\s*do\b")
_NULL_REDIRECT = re.compile(r"\d*>>?\s*/dev/null|&>\s*/dev/null|\d*>&\d+-?")
_FD_REDIRECT = re.compile(r"(?<![\w$])(?:\d|&)(>>?)")
_BRACE = re.compile(r"\{([^{}]*)\}")
_RANGE = re.compile(r"^(-?\d{1,18})\.\.(-?\d{1,18})$")
_OPERATOR = re.compile(r"&&|\|\||;;|>>|<<<|<<-|<<|>\||\|&|[;&|<>()]")


@dataclass
class CommandAccess:
    """What one command touches. Paths are absolute where they could be resolved."""

    reads: Set[str] = field(default_factory=set)
    writes: Set[str] = field(default_factory=set)
    # "full" (a whole file reached the output), "targeted" (only part of a file),
    # "listing" (names or existence only), or "none" (no file read).
    read_style: str = "none"


def strip_heredocs(command: str) -> str:
    """Drop heredoc bodies, which are text rather than commands."""
    return _HEREDOC.sub(lambda m: m.group(0)[: m.start(3) - m.start(0)].rstrip("\n") + "\n", command)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _split_words(text: str) -> List[str]:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _brace_items(body: str) -> Optional[Iterator[str]]:
    """The values of one brace group, lazily, or ``None`` when bash leaves it as is."""
    range_match = _RANGE.match(body)
    if range_match:
        start, stop = int(range_match.group(1)), int(range_match.group(2))
        step = 1 if stop >= start else -1
        return (str(n) for n in range(start, stop + step, step))
    if "," in body:
        return iter(body.split(","))
    return None


def expand_braces(word: str, limit: int = MAX_RANGE) -> List[str]:
    """Expand ``a{b,c}d`` and ``{1..3}`` lists the way bash does, keeping the first ``limit``.

    Values are generated only as far as the limit needs, so ``{1..1000000000}``
    or many groups in one word cost no more than a short list.
    """
    out: List[str] = []
    pending = [word]
    while pending and len(out) < limit:
        current = pending.pop(0)
        parts: List[str] = []
        groups: List[List[str]] = []
        last = 0
        for match in _BRACE.finditer(current):
            items = _brace_items(match.group(1))
            if items is None:
                continue
            parts.append(current[last:match.start()])
            groups.append(list(itertools.islice(items, limit)))
            last = match.end()
        if not groups:
            out.append(current)
            continue
        tail = current[last:]
        room = limit - len(out)
        variants = [
            "".join(part + item for part, item in zip(parts, combo)) + tail
            for combo in itertools.islice(itertools.product(*groups), room)
        ]
        # A variant may still hold an outer group ({a,{b,c}}); expand it before the rest.
        pending = (variants + pending)[:room]
    return out


def _variables(command: str) -> Dict[str, List[str]]:
    """Each variable's possible values from assignments and ``for`` lists."""
    values: Dict[str, List[str]] = {}
    for name, raw in _ASSIGN.findall(command):
        values.setdefault(name, [])
        value = _unquote(raw)
        if value not in values[name]:
            values[name].append(value)
    for name, raw in _FOR.findall(command):
        if "$(" in raw or "`" in raw:
            # A computed list (for f in $(ls ...)) has no values to expand.
            continue
        items = []
        for word in _split_words(raw):
            items.extend(expand_braces(word))
        if items:
            values[name] = items
    # Values may refer to other variables (record="/workspace/records/turn-$turn").
    for _ in range(3):
        for name, items in values.items():
            resolved: List[str] = []
            for item in items:
                for variant in _substitute(item, {k: v for k, v in values.items() if k != name}):
                    if variant not in resolved:
                        resolved.append(variant)
            values[name] = resolved[:MAX_RANGE]
    return values


def _substitute(text: str, values: Dict[str, List[str]]) -> List[str]:
    used = [name for name in values if re.search(r"\$\{%s\}|\$%s(?!\w)" % (name, name), text)]
    if not used:
        return [text]
    out = []
    for combo in itertools.islice(itertools.product(*(values[n] for n in used)), MAX_VARIANTS):
        variant = text
        for name, value in zip(used, combo):
            variant = re.sub(r"\$\{%s\}|\$%s(?!\w)" % (name, name), lambda _m, v=value: v, variant)
        out.append(variant)
    return out


def expand_command(command: str) -> List[str]:
    """The command once per combination of its variables' values (capped)."""
    command = strip_heredocs(command)
    values = {name: items for name, items in _variables(command).items() if items}
    return _substitute(command, values)


def _tokens(command: str) -> List[str]:
    """Shell words and operators, after dropping redirects to ``/dev/null``."""
    text = _NULL_REDIRECT.sub(" ", command)
    text = re.sub(r">&\s*(?=[^\s\d&])", "> ", text)
    text = _FD_REDIRECT.sub(r" \1", text)
    text = text.replace("$(", " ( ").replace("<(", " ( ").replace("`", " ; ").replace("\n", " ; ")
    lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|<>()")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        words = list(lexer)
    except ValueError:
        # Unbalanced quotes: fall back to a plain split of the operator-spaced text.
        words = re.sub(r"([;|&<>()])", r" \1 ", text).split()
    tokens: List[str] = []
    for word in words:
        # The lexer returns a run such as ")|" as one word; split it into operators.
        if word and all(ch in ";&|<>()" for ch in word):
            tokens.extend(_OPERATOR.findall(word))
        else:
            tokens.append(word)
    return tokens


def _simple_commands(tokens: List[str]) -> List[List[List[str]]]:
    """Group tokens into pipelines of simple commands."""
    pipelines: List[List[List[str]]] = [[[]]]
    for token in tokens:
        if token in PIPE:
            pipelines[-1].append([])
        elif token in SEPARATORS:
            pipelines.append([[]])
        else:
            pipelines[-1][-1].append(token)
    return [[cmd for cmd in pipe if cmd] for pipe in pipelines if any(pipe)]


def _strip_prefix(words: List[str]) -> List[str]:
    """Drop block keywords, assignments, and ``for`` headers before the command word."""
    while words:
        head = words[0]
        if head in BLOCK_WORDS:
            words = words[1:]
        elif head == "for":
            return []
        elif re.match(r"^[A-Za-z_]\w*=", head):
            words = words[1:]
        elif head in {"sudo", "command", "nohup", "env"}:
            words = words[1:]
        else:
            break
    return words


def _split_redirects(words: List[str]):
    """Separate ``> file`` / ``< file`` redirects from a command's arguments."""
    args, outputs, inputs = [], [], []
    skip_next = None
    for word in words:
        if skip_next == "out":
            outputs.append(word)
            skip_next = None
            continue
        if skip_next == "in":
            inputs.append(word)
            skip_next = None
            continue
        if skip_next == "skip":
            skip_next = None
            continue
        if word in {">", ">>", ">|"}:
            skip_next = "out"
        elif word == "<":
            skip_next = "in"
        elif word in {"<<", "<<-", "<<<"}:
            skip_next = "skip"
        else:
            args.append(word)
    return args, outputs, inputs


def _positionals(verb: str, args: List[str]) -> List[str]:
    """File operands of ``verb``, skipping options, option values, and patterns."""
    value_options = OPTION_VALUES.get(verb, set())
    positionals = []
    pattern_given = False
    skip = 0
    end_of_options = False
    for arg in args:
        if skip:
            skip -= 1
            continue
        if not end_of_options and arg == "--":
            end_of_options = True
            continue
        if not end_of_options and arg.startswith("-") and len(arg) > 1:
            if arg in value_options:
                skip = 2 if arg in {"--arg", "--argjson"} else 1
                if arg in PATTERN_OPTIONS:
                    pattern_given = True
            elif verb in {"head", "tail"} and arg[1:].isdigit():
                pass
            elif arg.split("=", 1)[0] in PATTERN_OPTIONS or arg[:2] in {"-e", "-f"} and verb in PATTERN_FIRST:
                pattern_given = True
            continue
        positionals.append(arg)
    if verb in PATTERN_FIRST and not pattern_given and positionals:
        positionals = positionals[1:]
    if verb == "find":
        # find PATH... [expression]: paths come before the first expression word.
        paths = []
        for arg in args:
            if arg.startswith("-") or arg in {"(", "!", ")"}:
                break
            paths.append(arg)
        return paths
    if verb in {"test", "[", "[["}:
        return [arg for arg in args if arg.startswith("/") or arg.startswith(".")]
    return positionals


def _clean_path(word: str, cwd: Optional[str]) -> Optional[str]:
    word = word.strip().rstrip(")").rstrip(",")
    if not word or word in {"-", "]", "]]", "{}"} or word.startswith("$") or "=" in word.split("/", 1)[0]:
        return None
    if word.startswith("~"):
        word = "/home/user" + word[1:]
    if not word.startswith("/"):
        if cwd is None:
            return word
        word = posixpath.join(cwd, word)
    normalized = posixpath.normpath(word)
    return "/" + normalized.lstrip("/")


def _paths(words: List[str], cwd: Optional[str]) -> List[str]:
    out = []
    for word in words:
        for expanded in expand_braces(word):
            path = _clean_path(expanded, cwd)
            if path:
                out.append(path)
    return out


def _is_filter_tail(pipeline: List[List[str]]) -> bool:
    for words in pipeline[1:]:
        words = _strip_prefix(words)
        if words and posixpath.basename(words[0]) in FILTERS | {"wc"}:
            return True
    return False


_STYLE_RANK = {"none": 0, "listing": 1, "targeted": 2, "full": 3}


def parse_command(command: str) -> CommandAccess:
    """Collect the paths a command reads and writes, and how it reads them."""
    access = CommandAccess()
    for variant in expand_command(command):
        _parse_variant(variant, access)
    return access


def _parse_variant(command: str, access: CommandAccess) -> None:
    cwd: Optional[str] = None
    for pipeline in _simple_commands(_tokens(command)):
        filtered = _is_filter_tail(pipeline)
        for position, raw_words in enumerate(pipeline):
            words = _strip_prefix(raw_words)
            if not words:
                continue
            if posixpath.basename(words[0]) == "xargs":
                # xargs runs the command that follows its options on piped names.
                rest = _positionals("xargs", words[1:])
                index = words.index(rest[0]) if rest else len(words)
                words = words[index:]
                if not words:
                    continue
            args, outputs, inputs = _split_redirects(words)
            access.writes.update(_paths(outputs, cwd))
            if not args:
                verb = ""
            else:
                verb, args = posixpath.basename(args[0]), args[1:]
            if verb == "cd":
                target = args[0] if args else "/home/user"
                cwd = _clean_path(target, cwd) if target.startswith("/") or cwd else None
                continue
            operands = _positionals(verb, args)
            reads: List[str] = []
            if verb in WRITE_ALL:
                access.writes.update(_paths(operands, cwd))
            elif verb in WRITE_LAST and len(operands) >= 2:
                access.writes.update(_paths(operands[-1:], cwd))
                reads = operands[:-1]
            elif verb == "sed" and any(a.startswith("-i") or a.startswith("--in-place") for a in args):
                access.writes.update(_paths(operands, cwd))
            elif verb == "sort" and "-o" in args[:-1]:
                access.writes.update(_paths([args[args.index("-o") + 1]], cwd))
                reads = operands
            elif verb in FULL_READERS or verb in FILTERS or verb in LISTERS:
                reads = operands
            paths = _paths(reads + inputs, cwd)
            if not paths:
                continue
            access.reads.update(paths)
            if verb in LISTERS:
                style = "listing"
            elif verb in FULL_READERS and not (filtered and position == 0):
                style = "full"
            else:
                style = "targeted"
            if _STYLE_RANK[style] > _STYLE_RANK[access.read_style]:
                access.read_style = style
