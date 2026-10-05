#!/usr/bin/env python3
"""A pure validator for the resident attention graph's shared-buffer binding contract.

The image says what one kernel binds; the graph says which allocation each index receives, how big
it is, and in what order the stages run. Neither checks the other, and nothing at runtime will
notice a disagreement: a binding longer than its allocation reads or writes past the payload, two
stages writing one region race, and a stage reading an intermediate whose producer has not run
reads whatever was there.

These are RUNTIME ALLOCATION facts, deliberately kept apart from compiler ABI facts. Nothing here
proposes a wire-format change; the image has no field for any of it - see
ledger/g17-shared-intermediates-carry-no-image-side-ordering.toml - which is exactly why the
contract has to be validated somewhere, and why passing it is not evidence of hardware ordering.

Pure: no I/O, no process, no image built. `validate(graph, contracts)` returns a list of findings,
empty when the graph is admissible.
"""


class Finding(str):
    """One rejection reason, carrying its stage so a caller can report which one failed."""

    def __new__(cls, stage, text):
        self = super().__new__(cls, "%s: %s" % (stage, text) if stage else text)
        self.stage = stage
        return self


def _element_bytes(element_type):
    return {"float": 4, "half": 2, "uint": 4, "atomic_uint": 4, "atomic_float": 4, "int": 4, "ushort": 2, "long": 8, "ulong": 8, "float2": 8, "float4": 16, "uint2": 8, "uint4": 16}.get(element_type)


def runtime_element_bytes(kind):
    """Element widths the common worker transports; signed int is not yet a runtime type."""
    return _element_bytes(kind) if isinstance(kind,str) and kind != "int" else None


def validate(graph, contracts, shapes=None, *, initialized_allocations=()):
    """Findings for one graph against {program: [(index, offset, written, element_type)]}.

    `shapes` optionally maps a program to its (rows, columns) so the exact grid can be checked;
    a program absent from it has its grid recorded and not judged, which is honest rather than
    permissive - an unchecked grid is reported as unchecked by the caller, not passed silently.
    `initialized_allocations` is supplied only after g17buffergraph validates the
    complete named source-byte initialization profile; raw graph rows grant nothing here.
    """
    findings = []
    allocations = graph.get("allocations") or {}
    windowed = graph.get("binding_windows") == "disjoint-v1"
    produced = {name for name, a in allocations.items()
                if a.get("role") in ("input", "parameter")}
    produced.update(initialized_allocations)

    for stage in graph.get("stages") or []:
        name = stage.get("name")
        program = stage.get("program")
        declared = contracts.get(program)
        if declared is None:
            findings.append(Finding(name, "no contract for program %r" % program))
            continue
        by_index = {i: (off, written, element) for i, off, written, element in declared}

        indices = [b["index"] for b in stage.get("bindings") or []]
        if sorted(indices) != sorted(by_index):
            findings.append(Finding(name, "graph binds %s; the contract binds %s"
                                    % (sorted(indices), sorted(by_index))))
            continue
        if len(set(indices)) != len(indices):
            findings.append(Finding(name, "binds an index twice: %s" % indices))
            continue

        # written ranges within this stage, to catch a stage writing over its own output
        written_ranges = []
        seen_allocations = {}

        for binding in stage["bindings"]:
            index = binding["index"]
            allocation = binding["allocation"]
            offset, written, element = by_index[index]
            where = "index %d (%s)" % (index, allocation)

            if bool(binding.get("written")) != bool(written):
                findings.append(Finding(name, "%s: graph written=%s, contract written=%s"
                                        % (where, binding.get("written"), written)))

            declared_allocation = allocations.get(allocation)
            if declared_allocation is None:
                findings.append(Finding(name, "%s: no such allocation" % where))
                continue

            # OUT OF BOUNDS: the bound window must fit inside the allocation
            start = binding.get("offset")
            length = binding.get("length")
            total = declared_allocation.get("allocation_bytes")
            if start is None or length is None:
                findings.append(Finding(name, "%s: binding states no offset/length" % where))
                continue
            if total is not None and start + length > total:
                findings.append(Finding(
                    name, "%s: binds [%d,%d) beyond the %d-byte allocation"
                    % (where, start, start + length, total)))
            # THESE TWO ARE THE WHOLE-BUFFER ASSUMPTION, AND ONLY THESE TWO. A graph that asks
            # for binding_windows=disjoint-v1 is asking to bind a sub-range, so requiring the
            # window to be the entire payload at the payload's own offset is the wrong question
            # for it - g17buffergraph checks the window lies inside the payload, is element
            # aligned, does not overlap an earlier write and is covered before it is read.
            # Everything else in this function, including the beyond-allocation check above and
            # the alias and element-type checks below, applies to both profiles unchanged.
            if not windowed:
                if length != declared_allocation.get("payload_bytes"):
                    findings.append(Finding(
                        name, "%s: binds %d bytes, the allocation holds %d"
                        % (where, length, declared_allocation.get("payload_bytes"))))
                if start != declared_allocation.get("offset"):
                    findings.append(Finding(name, "%s: binds at offset %d, allocation declares %d"
                                            % (where, start, declared_allocation.get("offset"))))
            if element != declared_allocation.get("element_type"):
                findings.append(Finding(name, "%s: contract element %s, allocation %s"
                                        % (where, element, declared_allocation.get("element_type"))))

            # INCOMPATIBLE ALIAS: one allocation reached twice in a stage must be reached the
            # same way, and must not be both read and written by the same launch.
            if allocation in seen_allocations:
                other_start, other_length, other_written, other_element = seen_allocations[allocation]
                if (other_start, other_length, other_element) != (start, length, element):
                    findings.append(Finding(
                        name, "%s: aliases the same allocation on incompatible terms" % where))
                if bool(written) != bool(other_written):
                    findings.append(Finding(
                        name, "%s: the same allocation is both read and written by one launch; "
                              "the image cannot express that and nothing orders it" % where))
            seen_allocations[allocation] = (start, length, bool(written), element)

            # READ BEFORE PRODUCED
            if not written and allocation not in produced:
                findings.append(Finding(name, "%s: read before any stage writes it" % where))

            if written:
                # OVERLAPPING WRITES within this launch
                for other_allocation, other_start, other_end in written_ranges:
                    if other_allocation == allocation and start < other_end and other_start < start + length:
                        findings.append(Finding(
                            name, "%s: writes [%d,%d) overlapping an earlier written range [%d,%d)"
                            % (where, start, start + length, other_start, other_end)))
                written_ranges.append((allocation, start, start + length))

        # EXACT GRID, where the caller supplied the shape to check it against
        grid = stage.get("grid")
        if shapes and program in shapes and grid is not None:
            if tuple(grid) != tuple(shapes[program]):
                findings.append(Finding(name, "grid %s is not the program's exact grid %s"
                                        % (list(grid), list(shapes[program]))))

        for binding in stage["bindings"]:
            if binding.get("written"):
                produced.add(binding["allocation"])

    return findings
