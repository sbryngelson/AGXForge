"""Bounded certificate for an explicit compact device-buffer compilation.

This verifies a translation of compiler-produced code, not correctness of the
source arithmetic. The compiler must independently derive uses from the original
verified IR; final descriptor bytes cannot prove that a declaration was unused.
No authoring or runtime admission is performed here.
"""
from collections import Counter
import hashlib
import json
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass

CONFIG = ConfigDict(strict=True, extra="forbid", ser_json_bytes="hex", val_json_bytes="hex")
Index = Annotated[int, Field(ge=0, le=30)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

# Explicitly inspected forms in the four compact-emission controls. New forms
# need classification before this certificate can claim a descriptor-only change.
MEMORY_FORMS = {(12646, 14): "load", (12682, 14): "load",
                (17193, 14): "store_at", (17229, 8): "store_at"}
OTHER_FORMS = frozenset({(424, 4), (590, 4), (684, 4), (9700, 14),
    (10279, 12), (10283, 12), (10825, 14), (11372, 10), (11842, 8),
    (13575, 4), (14047, 10), (14059, 4), (14391, 14), (17013, 14),
    (17771, 4)})


@dataclass(frozen=True, config=CONFIG)
class PublicDeclaration:
    index: Index
    element_type: str
    element_bytes: Annotated[int, Field(ge=1, le=16)]

    def __post_init__(self):
        from .abi import ELEMENT_PHYSICAL
        physical = ELEMENT_PHYSICAL.get(self.lowering_type)
        if physical is None or self.element_bytes != physical[0]:
            raise ValueError("public declaration has unknown type or incompatible width")

    @property
    def lowering_type(self):
        # Preserve the original spelling; these are the existing frontend aliases.
        return {"int": "uint", "short": "ushort"}.get(self.element_type, self.element_type)


@dataclass(frozen=True, config=CONFIG)
class OriginalBufferUse:
    ordinal: Annotated[int, Field(ge=0)]
    index: Index
    kind: Literal["load", "store_at"]


def declaration_digest(declarations):
    values = [dict(index=d.index, element_type=d.element_type,
                   element_bytes=d.element_bytes) for d in declarations]
    return hashlib.sha256(json.dumps(values, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, config=CONFIG)
class ProjectionCertificate:
    mode: Literal["unused-device-buffer-v1"]
    declarations: Annotated[tuple[PublicDeclaration, ...], Field(min_length=2, max_length=31)]
    declarations_sha256: Digest
    # Display/shape provenance only: repr(fn) omits buffer types, can collide
    # for equal names, and includes construction-order names for anonymous SSA.
    # Declaration, use/owner and code facts are checked separately below.
    original_ir_shape_sha256: Digest
    original_uses: Annotated[tuple[OriginalBufferUse, ...], Field(min_length=1)]
    active_index: Index
    eliminated: Annotated[tuple[Index, ...], Field(min_length=1)]
    elimination_reason: Literal["no_buffer_reference_in_original_ir"]
    original_code: bytes
    original_code_sha256: Digest

    def __post_init__(self):
        indices = [d.index for d in self.declarations]
        if indices != sorted(set(indices)):
            raise ValueError("projection declarations must have unique ascending indices")
        if declaration_digest(self.declarations) != self.declarations_sha256:
            raise ValueError("projection public declaration identity differs")
        if self.active_index not in indices:
            raise ValueError("projection active index is not publicly declared")
        if list(self.eliminated) != [i for i in indices if i != self.active_index]:
            raise ValueError("projection declarations are not exactly partitioned")
        ordinals = [u.ordinal for u in self.original_uses]
        if ordinals != sorted(set(ordinals)):
            raise ValueError("original resource uses must have unique ascending ordinals")
        if any(u.index != self.active_index for u in self.original_uses):
            raise ValueError("projection would omit an original resource use")
        if not any(u.kind == "store_at" for u in self.original_uses):
            raise ValueError("projection requires the single active buffer to be written")
        if hashlib.sha256(self.original_code).hexdigest() != self.original_code_sha256:
            raise ValueError("projection original code identity differs")
        instructions = self._instructions(self.original_code)
        expected_base = 4 * indices.index(self.active_index)
        actual = Counter()
        for ins in instructions:
            kind = MEMORY_FORMS.get((ins.opcode.id, len(ins.raw)))
            if kind is not None:
                if ins.raw[1] != expected_base:
                    raise ValueError("original descriptor does not name the active public declaration")
                actual[kind] += 1
        if actual != Counter(u.kind for u in self.original_uses):
            raise ValueError("original IR use counts differ from emitted memory instructions")

    @staticmethod
    def _instructions(code):
        from .model import decode
        instructions = list(decode(code, 0))
        cursor = 0
        for ins in instructions:
            if ins.offset != cursor or not ins.raw:
                raise ValueError("projection instruction boundaries do not cover code")
            cursor += len(ins.raw)
            if ins.opcode is None or (ins.opcode.id, len(ins.raw)) not in MEMORY_FORMS.keys() | OTHER_FORMS:
                raise ValueError("projection contains an unclassified instruction form")
        if cursor != len(code) or not instructions:
            raise ValueError("projection instruction boundaries do not cover code")
        return instructions

    def check_bindings(self, bindings):
        """Check the active mapping independently of its code identity."""
        if len(bindings) != 1:
            raise ValueError("projection requires exactly one emitted binding")
        binding = bindings[0]
        declaration = next(d for d in self.declarations if d.index == self.active_index)
        if (binding.index, binding.offset, binding.written, binding.element_type, binding.element_bytes) != (
                declaration.index, 0, True, declaration.lowering_type, declaration.element_bytes):
            raise ValueError("projection emission mapping disagrees with its public declaration")

    def check_emission(self, code, bindings):
        """Require the exact active contract and descriptor-only code translation."""
        self.check_bindings(bindings)
        if type(code) is not bytes or len(code) != len(self.original_code):
            raise ValueError("projection changed code length or representation")
        old, new = self._instructions(self.original_code), self._instructions(code)
        if len(old) != len(new):
            raise ValueError("projection changed instruction count")
        for a, b in zip(old, new):
            if (a.offset, a.opcode.id, len(a.raw)) != (b.offset, b.opcode.id, len(b.raw)):
                raise ValueError("projection changed instruction boundaries or opcode")
            if (a.opcode.id, len(a.raw)) in MEMORY_FORMS:
                if b.raw[1] != 0:
                    raise ValueError("compact descriptor does not name rank zero")
                if a.raw[:1] + a.raw[2:] != b.raw[:1] + b.raw[2:]:
                    raise ValueError("projection changed a non-descriptor memory field")
            elif a.raw != b.raw:
                raise ValueError("projection changed a non-resource instruction")

    def to_dict(self):
        return ADAPTER.dump_python(self, mode="json")

    @classmethod
    def from_dict(cls, value):
        return ADAPTER.validate_json(json.dumps(value))


ADAPTER = TypeAdapter(ProjectionCertificate)


def compile_compact(fn, *, regs, name, resolved, fma_always_load_wait):
    """Compile both mappings from independent copies; never patch emitted bytes.

    All original Buffer references count, including unreachable blocks and dead
    operations. Only direct load/store_at uses are admitted in this first domain.
    A caller supplies no eliminated list or replacement declaration facts.
    """
    import copy
    from . import cc, ir

    ir.verify(fn)
    if resolved is not None or fn.threadgroup is not None:
        raise cc.Unsupported("compact resources require ordinary device buffers without a resolved layout or threadgroup state")
    buffers = sorted(fn.buffers, key=lambda b: b.slot)
    if len(buffers) < 2 or len({b.slot for b in buffers}) != len(buffers):
        raise cc.Unsupported("compact resources require distinct public declarations with an unused member")
    declarations = tuple(PublicDeclaration(b.slot, ir.check_declared_element(b.elem, b.declared_element),
                                           ir.ELEM_BYTES[b.elem]) for b in buffers)
    declared_ids = {id(b) for b in buffers}

    def references(value, *, attribute=False, ancestors=frozenset()):
        if type(value) is ir.Buffer:
            return [value]
        if type(value) in (list, tuple, set, frozenset, dict):
            if id(value) in ancestors:
                raise cc.Unsupported("compact resources cannot inspect a cyclic operand or attribute container")
            nested = ancestors | {id(value)}
            items = (item for pair in value.items() for item in pair) if type(value) is dict else value
            return [b for item in items for b in references(item, attribute=attribute, ancestors=nested)]
        if type(value) in (str, int, bool, float, bytes, type(None)):
            return []
        # SSA arguments refer to separately visited original operations. Do not
        # treat an arbitrary object (or a scalar/container subclass) as empty.
        if not attribute and type(value) in (ir.Value, ir.Imm):
            return []
        raise cc.Unsupported("compact resources cannot inspect operand or attribute type " + type(value).__name__)

    uses = []
    for ordinal, op in enumerate(op for block in fn.blocks for op in block.ops):
        refs = references(op.args) + references(op.attrs, attribute=True)
        if not refs:
            continue
        if any(id(b) not in declared_ids for b in refs):
            raise cc.Unsupported("compact resource use names a Buffer outside the original declarations")
        if (op.kind not in ("load", "store_at") or len(refs) != 1 or
                not op.args or refs[0] is not op.args[0] or references(op.attrs, attribute=True)):
            raise cc.Unsupported("compact resources require direct load/store_at uses; address or other resource uses are not projected")
        uses.append(OriginalBufferUse(ordinal, refs[0].slot, op.kind))
    active = {u.index for u in uses}
    if len(active) != 1 or not any(u.kind == "store_at" for u in uses):
        raise cc.Unsupported("compact resources require exactly one accessed, written buffer")
    active_index = next(iter(active))
    original_ir_sha = hashlib.sha256(repr(fn).encode()).hexdigest()
    original_fn, compact_fn = copy.deepcopy(fn), copy.deepcopy(fn)
    compact_fn.buffers = [b for b in compact_fn.buffers if b.slot == active_index]
    options = dict(regs=tuple(regs), name=name, fma_always_load_wait=fma_always_load_wait)
    original = cc.compile_function(original_fn, **options)
    compact = cc.compile_function(compact_fn, **options)
    for program in (original, compact):
        contract = program.contract()
        if (contract.resources is not None or contract.spill_state is not None or
                contract.promoted_ranges is not None or contract.execution is not None or
                contract.uses_threadgroup or program._abi_constant_pool):
            raise cc.Unsupported("compact resources do not admit internal resources, spill, promotion, execution layout or constant pool")
    try:
        certificate = ProjectionCertificate(
            mode="unused-device-buffer-v1", declarations=declarations,
            declarations_sha256=declaration_digest(declarations),
            original_ir_shape_sha256=original_ir_sha, original_uses=tuple(uses),
            active_index=active_index,
            eliminated=tuple(b.slot for b in buffers if b.slot != active_index),
            elimination_reason="no_buffer_reference_in_original_ir",
            original_code=original.code,
            original_code_sha256=hashlib.sha256(original.code).hexdigest())
        certificate.check_emission(compact.code, compact.contract().bindings)
    except ValueError as exc:
        raise cc.Unsupported("compact resource translation refused: " + str(exc)) from exc
    compact._abi_projection = certificate
    compact.contract().check_code(compact.code)
    return compact
