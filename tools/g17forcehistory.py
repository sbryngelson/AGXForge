#!/usr/bin/env python3
"""Make a SHARED-HISTORY copy of a decode graph: the given tokens written into its token log after the prompt.

    python3 tools/g17forcehistory.py GRAPH.json TOKENS.json --name NAME

The matched study's shared-history arms (MM 25.211: graph_forced, graph_twin_all_forced) decode one fixed token
sequence in every arm. The generation step does that already: it takes log[q0 + 1] when the prompt fixed it and the
argmax only where the log holds 0xFFFFFFFF (tools/g17gen.py). So forcing a history is a DATA change, not a program
change: copy R_init.bin, write the tokens into the log right after the prompt, and point a copy of the graph at the
copy (kept under the name R_init.bin, in history_NAME/, because the executor resets state from files of that
name). Nothing else in the graph changes - arenas, bindings, dispatch order, bundles (twin or ours) are the original's.

TOKENS.json is a JSON list of token ids, or an object with "shared_tokens" (evidence/g17-matched-study-v1/
decode-shared-history.json is one). Writes history_NAME/R_init.bin and graph_NAME.json beside GRAPH.json; for the
study:

    python3 tools/g17forcehistory.py G/graph.json SHARED.json --name forced                  -> G/graph_forced.json
    python3 tools/g17forcehistory.py G/graph_twin_all.json SHARED.json --name twin_all_forced -> G/graph_twin_all_forced.json

Refuses, rather than guesses, when: the graph has no single-sequence generation region; R_init.bin is not loaded into
that region's arena; the log does not hold exactly the graph's prompt_ids followed by unset entries; or prompt plus
tokens exceed the log. CPU only; never loads Metal.
"""
import argparse
import json
import sys
from pathlib import Path

UNSET = 0xFFFFFFFF


def force(graph_path, tokens, name):
    graph_path = Path(graph_path)
    g = json.loads(graph_path.read_text())
    region = g.get("gen_region")
    if not region or region.get("batch", 1) != 1:
        raise ValueError("a single-sequence graph with a gen_region is required")
    inits = [i for i in g["arena_init"] if i["arena"] == region["arena"]
             and Path(i["file"]).name.startswith("R_init") and Path(i["file"]).suffix == ".bin"]
    if len(inits) != 1:
        raise ValueError("expected exactly one R_init.bin loaded into arena %s, found %d" % (region["arena"], len(inits)))
    init = inits[0]
    src = Path(init["file"])
    if not src.is_absolute():
        src = graph_path.parent / src
    data = bytearray(src.read_bytes())
    at = region["offset"] - init["offset"] + region["log_offset"]
    n = region["log_entries"]
    if at < 0 or at + 4 * n > len(data):
        raise ValueError("the token log (%d entries at %d) lies outside R_init.bin (%d bytes)" % (n, at, len(data)))
    log = [int.from_bytes(data[at + 4 * i:at + 4 * i + 4], "little") for i in range(n)]
    prompt = [int(t) for t in g["prompt_ids"]]
    L = len(prompt)
    if log[:L] != prompt:
        raise ValueError("the log does not begin with the graph's prompt_ids")
    if any(v != UNSET for v in log[L:]):
        raise ValueError("the log past the prompt is not all unset (0xFFFFFFFF): already forced?")
    tokens = [int(t) for t in tokens]
    if L + len(tokens) > n:
        raise ValueError("prompt %d + %d tokens exceed the log's %d entries" % (L, len(tokens), n))
    if any(t < 0 or t >= UNSET for t in tokens):
        raise ValueError("token ids must be uint32 and not 0xFFFFFFFF")
    for i, t in enumerate(tokens):
        data[at + 4 * (L + i):at + 4 * (L + i) + 4] = t.to_bytes(4, "little")
    # THE COPY KEEPS THE NAME R_init.bin, in its own directory: tools/g17decodegen's pipelined mode re-applies
    # arena_init entries whose file is named exactly R_init.bin between its warm and timed passes, so a renamed copy
    # would be loaded once and never reset - the timed pass would start from the warm pass's consumed log.
    out_bin = graph_path.parent / ("history_%s" % name) / "R_init.bin"
    out_graph = graph_path.parent / ("graph_%s.json" % name)
    for p in (out_bin.parent, out_graph):
        if p.exists():
            raise ValueError("%s exists; choose another --name" % p)
    out_bin.parent.mkdir()
    out_bin.write_bytes(bytes(data))
    forced = dict(g)
    forced["arena_init"] = [dict(i, file=str(out_bin)) if i is init else i for i in g["arena_init"]]
    forced["forced_history"] = dict(tokens=len(tokens), first_position=L, source_graph=graph_path.name,
                                    note="tokens written into the token log after the prompt (tools/g17forcehistory.py)")
    out_graph.write_text(json.dumps(forced, indent=1) + "\n")
    return out_graph, out_bin


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("graph")
    ap.add_argument("tokens")
    ap.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    t = json.loads(Path(a.tokens).read_text())
    tokens = t["shared_tokens"] if isinstance(t, dict) else t
    out_graph, out_bin = force(a.graph, tokens, a.name)
    print("wrote %s and %s: %d tokens after the prompt" % (out_graph, out_bin, len(tokens)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
