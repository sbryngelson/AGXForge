"""Join an existing g17link.Kernel to the semantic object author.

Compiler-owned launch fields are required; no donor, metadata-class fallback or
per-kernel constants are installed here. The returned archive is checked through
the independent verifier, then its delivered binding records are compared with
the compiler's ordered binding contract. This module never calls Metal.
"""
from dataclasses import dataclass
from collections.abc import Mapping as _Mapping
import hashlib
import struct


@dataclass(frozen=True)
class Image:
    archive: bytes
    library: bytes
    object: bytes
    field_ledger: dict
    sha256: str


def object_from_archive(archive):
    from . import verify as V
    if len(archive) < 8:
        raise ValueError("archive is shorter than its fat header")
    magic, count = struct.unpack_from(">II", archive)
    if magic != V.FAT_MAGIC or 8 + 20 * count > len(archive):
        raise ValueError("invalid fat archive header")
    for i in range(count):
        _cpu, _sub, offset, size, _align = struct.unpack_from(">IIIII", archive, 8 + 20 * i)
        if offset + size > len(archive):
            raise ValueError("archive slice exceeds the delivered image")
        container = archive[offset:offset + size]
        if len(container) >= 32 and struct.unpack_from("<I", container)[0] == V.MH_MAGIC_64:
            sections, _, _ = V._sections(container)
            if "__compute" in sections:
                off, length = sections["__compute"]
                if off + length > len(container):
                    raise ValueError("native object exceeds its archive slice")
                return container[off:off + length]
    raise ValueError("archive contains no native object")


def binding_records(metadata):
    """Read records with the verifier's parser, which is independent of authoring."""
    from . import verify as V
    errors = V.verify_metadata(metadata)
    if errors:
        raise ValueError("invalid metadata: " + "; ".join(errors))
    root = struct.unpack_from("<I", metadata)[0]
    _, slots = V._table(metadata, root)
    pk = V._ref(metadata, root, slots, 0)
    _, slots = V._table(metadata, pk)
    records = V._vector(metadata, V._ref(metadata, pk, slots, 4))
    out = []
    for rec in records:
        _, fields = V._table(metadata, rec)
        def get(slot, fmt):
            return struct.unpack_from(fmt, metadata, rec + fields[slot])[0] if slot in fields else 0
        out.append((get(1, "<I"), get(2, "<I"), bool(get(3, "<B"))))
    return out


def verify_contract(archive, library, expected_bindings):
    from . import verify as V
    errors = list(V.verify(archive, library))
    if errors:
        raise ValueError("finished image verification failed: " + "; ".join(errors))
    obj = object_from_archive(archive)
    md = V._objsect(obj, "__GPU_METADATA")
    if md is None:
        raise ValueError("delivered object has no metadata")
    actual = binding_records(md)
    expected = [(int(i), int(o), bool(w)) for i, o, w in expected_bindings]
    if actual != expected:
        raise ValueError(f"delivered binding contract {actual} differs from compiler contract {expected}")
    return obj


def expected_records_for_contract(abi, bindings):
    """Return the complete metadata record vector for a captured resource contract.

    Buffer contracts contain one record per user binding.  The measured narrow texture class has
    two internal records as well, and its vector order is ``44, users ascending, 48`` rather than
    the resource declaration order.  Comparing only the user records lets a texture image carry
    the wrong internal ranks while still reporting a matching user contract.  This helper derives
    the full vector from the texture planner and checks that its user entries agree with the
    compiler's ordered bindings before packaging.
    """
    resources = abi.get("resources") if isinstance(abi, dict) else None
    if not isinstance(resources, dict) or not resources.get("textures"):
        return [(int(row[0]), int(row[1]), bool(row[2])) for row in bindings]
    from . import teximage as g17teximage
    placed = g17teximage.plan(abi)["placed"]
    # A REFUSAL THAT NAMES THE WRONG QUESTION IS WORSE THAN A CRASH, because a reader acts on it -
    # the warning g17teximage's own routing comment carries. Give an unmeasured resource shape its
    # own message: when the planner recognises none of it, it places NO records at all, and
    # reporting that as "binding record for 0 differs from the compiler contract" sends a reader to
    # the one binding that is not in question. Measured by handing this the class's own contract
    # with internals 46/48 or 44/46 in place of the measured 44/48: the planner places nothing and
    # the old message named user record 0.
    ordered = placed.get("binding record order") or ()
    placed_records = placed.get("binding records") or ()
    if bindings and not placed_records:
        raise ValueError(
            "the texture planner placed NO binding records for this contract, so its resource "
            "shape is outside the measured class - check resources.internal and resources.access "
            "before the user bindings %s, which are not what this refusal is about"
            % ([tuple(row[:3]) for row in bindings],))
    records = {int(row["index"]): row for row in placed_records}
    for row in bindings:
        index, offset, written = row[:3]
        record = records.get(int(index))
        if record is None:
            raise ValueError("the texture planner placed no binding record for user index %d, "
                             "though it placed %s" % (int(index), sorted(records)))
        if int(record.get("field2 (offset)") or 0) != int(offset):
            raise ValueError("texture planner binding record %d states offset %s and the compiler "
                             "contract states %s" % (int(index),
                                                     record.get("field2 (offset)"), offset))
        if bool(record.get("field3 (written)") or 0) != bool(written):
            raise ValueError("texture planner binding record %d states written=%s and the compiler "
                             "contract states %s" % (int(index),
                                                     bool(record.get("field3 (written)") or 0),
                                                     bool(written)))
    expected = []
    for index in ordered:
        record = records.get(int(index))
        if record is None:
            raise ValueError("texture planner order names missing binding record %d" % int(index))
        expected.append((int(index), int(record.get("field2 (offset)") or 0),
                         bool(record.get("field3 (written)") or 0)))
    return expected


# THE FOUR-WORD VECTOR STORE, and the rule that ties one to a binding. `g17cc` emits these with
# `desc_const = 4 * binding_rank` - its own docstring states the rule and `_buf_rank` applies it -
# so a store's descriptor constant names a POSITION in the contract's binding list, not a public
# buffer index. The two are different numbers whenever the declared indices are not 0, 1, 2 ...
VEC4_STORE_OPCODE = 17256
VEC4_DESC_PER_RANK = 4


def spill_store_targets(code, contract):
    """Public binding index -> count of four-word vector stores addressing it.

    Read from the DELIVERED bytes through the repository's own field decoder, located by the
    contract's own instruction list rather than by re-walking the stream, so the offsets are the
    compiler's and the fields are `g17asm`'s. No subprocess and no second disassembler.
    """
    from . import asm as g17asm
    order = [b.index for b in contract.bindings]
    counts = {}
    for instruction in contract.instructions:
        if instruction.opcode != VEC4_STORE_OPCODE:
            continue
        unit = code[instruction.offset:instruction.offset + instruction.length]
        descriptor = g17asm.decode_vec4(unit)["desc_const"]
        if descriptor % VEC4_DESC_PER_RANK:
            raise ValueError('a four-word vector store at offset %d carries descriptor constant '
                             '%d, which is not %d times any binding rank; this side cannot say '
                             'which binding it addresses'
                             % (instruction.offset, descriptor, VEC4_DESC_PER_RANK))
        rank = descriptor // VEC4_DESC_PER_RANK
        if rank >= len(order):
            raise ValueError('a four-word vector store at offset %d addresses binding rank %d and '
                             'the contract declares %d binding(s)'
                             % (instruction.offset, rank, len(order)))
        counts[order[rank]] = counts.get(order[rank], 0) + 1
    return counts


def check_spill_target(code, contract):
    """The declared scratch binding must be the one the emitted spill stores actually address.

    WHY THIS EXISTS. `spill_state.binding_index` was checked for MEMBERSHIP in the declared set and
    nothing else, so a contract could name the OUTPUT buffer as its scratch region and author a
    byte-identical image - the bytes carry no scratch declaration, so nothing downstream would
    disagree either. `groups` was already validated against the emitted vector stores; the binding
    beside it was not, which is the asymmetry this closes.

    THIS IS A CONSISTENCY CHECK, NOT A PROOF OF SPILL SEMANTICS. It says the contract's statement
    about which binding holds scratch agrees with the stores the compiler emitted. It says nothing
    about lifetimes, reload correctness, or whether a launch allocates the region.

    A contract with NO `spill_state` is untouched: an ordinary program's user vector stores are not
    spills and are not counted as any.
    """
    state = getattr(contract, 'spill_state', None)
    if state is None:
        return
    declared = state.binding_index
    binding = next((b for b in contract.bindings if b.index == declared), None)
    if binding is None:
        raise ValueError('spill state names binding %d, which the contract does not declare'
                         % declared)
    # THE SCRATCH BINDING MUST BE WRITABLE AND A WORD. The form moves four consecutive 32-bit
    # registers, so a declaration this side cannot store words through is not a scratch region
    # whatever the contract calls it.
    if not binding.written:
        raise ValueError('spill state names binding %d as scratch, and the contract declares it '
                         'read-only; the four-word store writes to it' % declared)
    if binding.element_bytes != VEC4_DESC_PER_RANK:
        raise ValueError('spill state names binding %d as scratch and the contract declares a '
                         '%d-byte element; the four-word vector store moves 4-byte words'
                         % (declared, binding.element_bytes))
    counts = spill_store_targets(code, contract)
    reached = counts.get(declared, 0)
    if reached != state.groups:
        raise ValueError('spill state declares %d group(s) on binding %d, and the delivered code '
                         'has %d four-word vector store(s) addressing that binding%s. The state\'s '
                         'own basis is the emitted stores, so the two have to agree.'
                         % (state.groups, declared, reached,
                            '' if not counts else
                            ' (stores by binding: %s)'
                            % ', '.join('%d->%d' % kv for kv in sorted(counts.items()))))


def author_view(kernel,abi,bindings,program_contract):
    """Bind the typed compiler layout to these bytes before deriving metadata.

    The wire ABI remains unchanged. These two values are an author-internal
    view of the captured instruction list, not a second caller-maintained copy.
    op458 is the measured back-edge form; forward branch op462 does not count.
    """
    from . import abi as g17abi
    if not isinstance(program_contract,g17abi.ProgramABI):
        raise ValueError('program_contract must be the typed compiler contract')
    contract=g17abi.ProgramABI.from_dict(program_contract.to_dict())
    contract.check_code(kernel.code)
    if contract.name!=kernel.name or contract.entry!=kernel.entry or contract.prologue!=kernel.prologue:
        raise ValueError('typed contract differs from kernel identity or entry')
    actual=[(b.index,b.offset,b.written,b.element_type) for b in contract.bindings]
    if actual!=bindings:raise ValueError('typed contract differs from delivered bindings')
    check_spill_target(kernel.code,contract)
    if contract.forms!=tuple(sorted(tuple(f) for f in abi['forms'])):
        raise ValueError('typed instruction forms differ from ABI')
    for key in ('arch_flag','uses_threadgroup','writes_buffer','writes_texture'):
        if type(abi.get(key)) is not bool or getattr(contract,key)!=abi[key]:
            raise ValueError(f'typed contract differs from ABI {key}')
    launch=abi.get('launch')
    if not isinstance(launch,_Mapping) or 'exact_grid_required' not in launch:
        # A NAMED REFUSAL, not a KeyError. The emission ABI carries the launch facts and the typed
        # contract carries the grid requirement; a missing block used to surface as `KeyError:
        # 'launch'`, which refuses without saying what is missing or who has to supply it.
        raise ValueError('the emission ABI must state its launch block with exact_grid_required; '
                         'the contract states exact_grid_required=%r and the ABI states %r'
                         %(contract.exact_grid_required,launch))
    if contract.exact_grid_required!=abi['launch']['exact_grid_required']:
        raise ValueError('typed launch requirement differs from ABI')
    # Execution is a compiler requirement, not a caller-selected image option.
    # Mapping accepts the compiler's frozen ABI as well as its JSON wire view.
    from collections.abc import Mapping
    if contract.execution is not None:
        block=abi.get('execution')
        if (type(abi.get('abi_version')) is not int or abi['abi_version']!=5 or
            not isinstance(block,Mapping) or set(block)!={'simd_width','tensor'} or
            type(block.get('simd_width')) is not int or type(block.get('tensor')) is not bool or
            block['simd_width']!=contract.execution.simd_width or
            block['tensor']!=contract.execution.tensor):
            raise ValueError('typed execution requirement differs from ABI')
    elif 'execution' in abi or abi.get('abi_version')==5:
        raise ValueError('execution requirement without a captured tensor contract')
    # Requantization is a compiler-owned primitive, not an authoring option. Preserve its exact
    # marker across the typed and emission views so removing it cannot fall through to the generic
    # serializer, which has no measured writable-uchar metadata class.
    if contract.requantization is not None:
        marker = contract.requantization
        expected = {
            "kind": marker.kind,
            "source_binding": marker.source_binding,
            "destination_binding": marker.destination_binding,
            "elements": marker.elements,
            "groups": marker.groups,
            "scale": marker.scale,
            "rounding": marker.rounding,
            "saturation": marker.saturation,
            "dispatch_boundary": marker.dispatch_boundary,
        }
        if marker.scale_binding is not None:
            expected["scale_binding"] = marker.scale_binding
        if marker.metadata_class is not None:
            expected["metadata_class"] = marker.metadata_class
        if marker.scale_storage is not None:
            expected["scale_storage"] = marker.scale_storage
        if marker.scale_addressing is not None:
            expected["scale_addressing"] = marker.scale_addressing
        if marker.output_storage is not None:
            expected["output_storage"] = marker.output_storage
        if abi.get("requantization") != expected:
            raise ValueError("typed requantization marker differs from ABI")
    elif "requantization" in abi:
        raise ValueError("requantization marker without a captured compiler contract")
    if contract.resources is not None:
        # Compare the entire captured resource declaration, including explicit
        # unknowns. JSON comparison distinguishes bool/int and int/float while
        # treating frozen tuples and wire lists as the same serialized sequence.
        import json
        def plain(value):
            if isinstance(value, Mapping):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(item) for item in value]
            return value
        block=abi.get('resources')
        if (type(abi.get('abi_version')) is not int or abi['abi_version']!=6 or
            not isinstance(block,Mapping) or
            json.dumps(plain(block),sort_keys=True)!=json.dumps(contract.to_dict()['resources'],sort_keys=True)):
            raise ValueError('typed resource requirement differs from ABI')
    elif 'resources' in abi or abi.get('abi_version')==6:
        raise ValueError('resource requirement without a captured texture contract')
    if contract.uses_threadgroup or contract.execution is not None or contract.resources is not None:
        pool=abi.get('constant_pool')
        if (not isinstance(pool,(list,tuple)) or any(type(v) is not int or not 0<=v<=255 for v in pool)
            or tuple(pool)!=contract.constant_pool):
            raise ValueError('typed constant-pool declaration differs from ABI')
    if contract.uses_threadgroup:
        block=abi.get('threadgroup',{})
        if (tuple(block.get('required_size',()))!=contract.threadgroup.required_size or
            block.get('static_memory_bytes')!=contract.threadgroup.static_memory_bytes or
            block.get('static_memory_alignment')!=contract.threadgroup.static_memory_alignment or
            tuple(block.get('dynamic_memory',()))!=contract.threadgroup.dynamic_memory):
            raise ValueError('typed threadgroup declaration differs from ABI')
    view=dict(abi)
    # THE EMITTED ARGUMENT STATE TRAVELS WITH THE VIEW, for the same reason the instruction facts
    # below do: the author needs a compiler-owned fact that the wire ABI does not carry, and the
    # typed contract is where it lives. It is copied, not re-derived, so the author cannot disagree
    # with the compiler about what the compiler staged.
    if contract.argument_state is not None:
        view['argument_state']=contract.to_dict()['argument_state']
    if contract.resources is not None:
        # Serialize only at the author boundary after validating the captured
        # declaration; the author consumes the same view for frozen and wire ABI.
        view['resources']=contract.to_dict()['resources']
    facts=dict(instruction_count=len(contract.instructions),
               has_back_edge=any(i.opcode==458 for i in contract.instructions))
    for key,value in facts.items():
        if key in view and (type(view[key]) is not type(value) or view[key]!=value):
            raise ValueError(f'{key} conflicts with captured compiler instructions')
        view[key]=value
    return view


def _resolve_narrow_texture_contract(abi, program_contract):
    """Resolve the three section facts for the measured one-texture runtime class.

    ABI v6 deliberately carries these as ``not_stated`` until a linker class resolves them.  The
    narrow class has now supplied that resolution: an empty eight-byte zero vector, no kind-9
    record, and the measured 44/user/48 record order.  Keep this behind the exact resource shape
    and leave every other texture contract refused.  ``program_contract`` remains the original
    compiler snapshot; this returned mapping is the linker's resolved author view.
    """
    resources = abi.get("resources") if isinstance(abi, dict) else None
    if not isinstance(resources, dict) or not resources.get("textures"):
        return abi
    textures = resources.get("textures") or ()
    public = abi.get("texture_public_indices")
    if len(textures) == 2 and not public:
        # The pair needs its [[texture(n)]] slots stated as an authoring option. Nothing in the
        # delivered program carries them, so leaving the contract unresolved here is what makes the
        # emitter refuse by name instead of authoring a mask for slots nobody chose.
        return abi
    if not (len(textures) in (1, 2) and
            len(resources.get("internal") or ()) == 2 and
            [(r.get("apple_index"), r.get("rank")) for r in resources.get("internal") or ()]
            == [(44, 0), (48, 1)] and resources.get("samplers") == [] and
            resources.get("argument_bytes") == 8 and
            list(resources.get("not_stated") or ()) ==
            ["slot27_contents", "slot2_resource_record", "slot2_kind9_record"]):
        return abi
    resolved = dict(resources)
    resolved["not_stated"] = []
    if len(textures) == 2:
        # RECORD WHAT IS APPLIED. The mapping is an authoring input, so the resolved author view
        # carries the exact slots the section was authored for rather than leaving a consumer to
        # infer them from the dense order.
        resolved["texture_public_indices"] = list(public)
    # ONE STATEMENT OF THE POOL FORM, NOT TWO. g17texpoolform owns it and carries its evidence:
    # the form is NOT a constant of the texture class - over the Apple sections whose slot-13
    # vector is entirely zero, 846 are texture sections and only 211 carry the eight-byte form -
    # so it is stated as the one-witness class's own value and nothing else. A literal here would
    # be a second copy of a value whose justification lives somewhere else.
    from . import texpoolform as g17texpoolform
    resolved["constant_pool_emptiness"] = g17texpoolform.THE_CLASSS_FORM
    # MAIN'S COUNT, NOT THE WHOLE __text. This field feeds the slot-32 boundary rule, and counting
    # the whole text where main was meant is what produced four false counterexamples to that rule
    # once already. ProgramABI carries the prologue as its own field, so `instructions` is main
    # alone - asserted in the linker's texture suite rather than assumed here.
    out = dict(abi, resources=resolved,
               main_instruction_count=len(program_contract.instructions))
    return out


def link(kernel, abi, *, binding_offsets, program_contract=None):
    """Author and verify one image using the existing Kernel interface.

    binding_offsets is ordered exactly as kernel.bindings. It must be supplied
    from the SAME resource layout used by instruction selection; this adapter
    does not renumber bindings after instructions have been emitted.

    An explicit prologue is required because merely emitting `end` cannot prove
    that the compiler's descriptor/base-register setup has been satisfied.
    """
    from . import link as L
    from . import authorobj as A
    from . import abi as g17abi
    tensor_forms=any(f[0] in g17abi.TENSOR_OPCODES for f in abi.get('forms',()))
    texture_forms=any(f[0] in g17abi.TEXTURE_OPCODES for f in abi.get('forms',()))
    # ``ProgramABI.model_dump`` includes optional fields with a None value; at this final
    # admission boundary an absent optional block and that schema-generated None are equivalent.
    # ``author_view`` above still uses key presence so callers that explicitly add an empty block
    # cannot bypass the captured-contract checks.
    if program_contract is None and (texture_forms or abi.get('resources') is not None or abi.get('abi_version')==6):
        raise A.Missing('texture image admission requires a captured compiler resource contract')
    if program_contract is None and (tensor_forms or abi.get('execution') is not None or abi.get('abi_version')==5):
        raise A.Missing('tensor image admission requires a captured compiler execution contract')
    if abi.get('uses_threadgroup') and program_contract is None:
        raise A.Missing('threadgroup image admission requires an agreed storage/launch ABI and measured resource class; the ordinary buffer layout cannot supply them')
    if kernel.prologue is None:
        raise A.Missing("compiler must supply the constant-program/prologue bytes")
    if kernel.fixups:
        raise A.Missing("this adapter cannot yet finalize symbolic fixups")
    if len(binding_offsets) != len(kernel.bindings):
        raise ValueError("one descriptor offset is required per compiler binding")
    indices = [b.index for b in kernel.bindings]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate compiler binding indices")
    if any(not isinstance(o, int) or o < 0 or o % 2 for o in binding_offsets):
        raise ValueError("binding offsets must be nonnegative, even integer pointer-block offsets")
    if len(set(binding_offsets)) != len(binding_offsets):
        raise ValueError("two bindings cannot share a pointer-block offset")
    # The author currently builds device-buffer kind-5 records. A type-rich
    # Kernel must not imply support for resources the author cannot serialize.
    if any(b.kind != "device_buffer" for b in kernel.bindings):
        raise A.Missing("this adapter currently supports device_buffer records only")
    bindings = [(b.index, off, not b.readonly, b.element_type)
                for b, off in zip(kernel.bindings, binding_offsets)]
    if program_contract is not None:
        abi=author_view(kernel,abi,bindings,program_contract)
        abi = _resolve_narrow_texture_contract(abi, program_contract)
    text = L.text_of(kernel)
    if program_contract is not None and program_contract.resource_projection is not None:
        from . import compactimage
        sections, ledger = compactimage.author_sections(kernel.code, program_contract, abi)
    else:
        sections, ledger = A.author(text, kernel.entry, bindings, abi)
    require_tensor_slot44(tensor_forms, sections["__GPU_METADATA,__compute"])
    require_driver_slots(sections["__GPU_METADATA,__compute"])
    ib = abi.get('imageblock')
    if ib is not None:
        # AN EXPLICIT-LAYOUT IMAGEBLOCK IS DECLARED IN THREE SECTIONS, measured from Apple's compiler
        # and proven on hardware (agxforge.g17.imageblock, isa/g17-imageblock-receipt.json). Without it
        # the pipeline allocates no tile and every imageblock read returns 0, silently.
        from . import imageblock as g17imageblock
        if ib.get('layout') != 'explicit':
            raise A.Missing("imageblock layout %r: only an EXPLICIT layout is a compute construct - "
                            "Apple's own compute backend rejects the implicit one" % ib.get('layout'))
        sections = dict(sections)
        (sections["__GPU_METADATA,__compute"], sections["__GPU_LD_MD,__compute"],
         sections["__GPU_ARCH_LD_MD,__compute"]) = g17imageblock.declare(
            sections["__GPU_METADATA,__compute"], sections["__GPU_LD_MD,__compute"],
            sections["__GPU_ARCH_LD_MD,__compute"], ib.get('element_bytes'),
            per_kernel=not (tensor_forms and g17imageblock.pk_declared(sections["__GPU_METADATA,__compute"])))
        ledger = dict(ledger)
        ledger["imageblock"] = ("per-kernel slot 19, LD_MD t136 slot 23 and an ARCH element size of "
                                "%s bytes (agxforge.g17.imageblock)" % ib.get('element_bytes'))
    expected = expected_records_for_contract(abi, bindings)
    return package_sections(kernel, sections, ledger, expected)



class MissingTensorSlot(ValueError):
    pass


def require_tensor_slot44(is_tensor, metadata):
    """Refuse a tensor image whose per-kernel table lacks slot 44 - nothing downstream would.

    MEASURED 2026-09-23 (tools/g17slot44.py): Apple's own tensor_ops matmul with slot 44's vtable entry
    zeroed still gets a pipeline from the loader, runs at command-buffer status 0, and returns 1,015 of
    1,024 words WRONG, identically over three runs. Neither the loader nor the status catches it, so the
    refusal has to be here. Over every committed object the rule is exceptionless in both directions:
    116 with a tensor opcode all carry slot 44, 628 without one carry none."""
    from . import gpumd as GM
    # In-process on purpose: a build may not start an external process (agxforge.g17.audit), so the
    # caller says whether the program is tensor - link() knows it from the ABI's forms.
    if is_tensor and 44 not in GM.fields(bytes(metadata)):
        raise MissingTensorSlot("a tensor program's per-kernel metadata has no slot 44: the loader accepts it "
                                "and the matmul returns wrong values at status 0 (tools/g17slot44.py)")

class MissingDriverSlots(ValueError):
    pass


# THE PER-KERNEL SLOT APPLE'S DRIVER READS WITHOUT CHECKING. Slot 13 - the kernel's constant data - is present
# in every Apple section in a 1,990-object sample of the cache, in every delivered image (421 distinct archives,
# MM 25.148), and in every measured class this author writes, including the executed range-store class, which
# carries neither 27 nor 29 and runs. The seven real sources whose images lacked it are exactly the seven that
# killed the process inside newComputePipelineState, in AGX::ProgramVariantESLState reading address 0x4 - a count
# read through the missing reference (MM 25.149). Pipeline creation checks nothing else in the metadata
# (MM 25.147), so the refusal has to be here.
DRIVER_SLOTS = (13,)


def require_driver_slots(metadata):
    """Refuse a kernel table missing a slot the driver dereferences - refused here, or it crashes there."""
    from . import gpumd as GM
    missing = [s for s in DRIVER_SLOTS if s not in GM.fields(bytes(metadata))]
    if missing:
        raise MissingDriverSlots(
            "the authored per-kernel metadata has no slot %s. Every Apple section carries slot 13, and "
            "the images without it crashed Apple's driver at pipeline creation in 7 of 7 real sources "
            "(MM 25.149). This binding pattern needs a measured metadata class." % missing)


def package_sections(kernel, sections, ledger, expected_bindings):
    """Package an explicit metadata ABI and check the delivered object and bindings."""
    from . import link as L
    from . import obj as g17obj
    from . import arc as g17arc
    from . import mtlb as g17mtlb
    if kernel.fixups:
        raise ValueError("unresolved fixups cannot be packaged")
    text = L.text_of(kernel)
    obj = g17obj.build(text, sections["__GPU_METADATA,__compute"],
                       sections["__GPU_LD_MD,__compute"], sections["__GPU_ARCH_LD_MD,__compute"],
                       sections["__GPU_STATS_MD,__compute"], entry=kernel.entry,
                       remarks=sections["__GPU_REMARKS_MD,__compute"])
    # Include metadata in the identity so changing only resources cannot reuse
    # the library identity of the preceding image.
    digest = hashlib.sha256(obj + kernel.name.encode()).digest()
    library = g17mtlb.build(name=kernel.name, hash=digest)
    archive = g17arc.emit(obj, library, library, module_hash=digest)
    delivered = verify_contract(archive, library, expected_bindings)
    if delivered != obj:
        raise ValueError("archive did not preserve the object that was authored")
    return Image(archive, library, obj, dict(ledger), hashlib.sha256(archive).hexdigest())


def author(program, contract=None, *, emission=None):
    """THE PUBLIC IMAGE-AUTHORING ENTRY: executable bytes and the captured contract, one call.

        author(compiled_program)                  everything from one object
        author(code, contract, emission=abi)      bytes plus the typed ProgramABI

    `link` below asks a caller for a Kernel, a separate ABI dict and a parallel list of binding
    offsets - three descriptions of one program that must agree, and nothing checks that they do.
    This derives the kernel and the offsets from the contract itself, so the only thing a caller
    states twice is nothing.

    WHAT THE TYPED CONTRACT CANNOT SUPPLY, measured rather than assumed. `ProgramABI` has 21 fields
    and the author reads 36 ABI keys; 26 of those are absent from a retained contract and eight -
    register_count, system_registers, forms, pk_extra, pk_values, launch, abi_version, profile -
    are not ProgramABI FIELDS at all. They are EMISSION facts: what the compiler chose while
    emitting, not what the program is. So bytes plus a contract are not sufficient to author, and
    this refuses by name rather than defaulting them, which is the whole reason slot 0 is a
    per-program register count and not a constant.

    A compiled-program object carries all three views, which is why it is the single-argument form.
    """
    import hashlib
    from . import link as L

    if contract is None:
        code = getattr(program, "code", None)
        if code is None:
            raise ValueError(
                "author(program) needs a compiled-program object carrying code, contract and ABI; "
                "pass author(code, contract, emission=...) for raw bytes")
        contract = program.contract()
        emission = emission if emission is not None else program.abi_plain(program.abi())
    else:
        code = program

    if emission is None:
        raise ValueError(
            "emission ABI not stated: a ProgramABI carries what the program IS, and authoring also "
            "needs what the compiler CHOSE while emitting it - register_count, system_registers, "
            "forms, pk_extra, pk_values, launch, abi_version and profile are not ProgramABI fields. "
            "Pass emission=..., or call author(compiled_program), which carries both views")

    # IDENTITY BEFORE AUTHORING. The contract names the bytes it describes; authoring different
    # bytes under it would produce an image whose metadata describes a program that does not exist.
    digest = hashlib.sha256(code).hexdigest()
    if contract.code_sha256 is not None and digest != contract.code_sha256:
        raise ValueError("code does not match the contract: sha256 %s, contract states %s"
                         % (digest[:16], contract.code_sha256[:16]))
    if contract.code_size is not None and len(code) != contract.code_size:
        raise ValueError("code does not match the contract: %d bytes, contract states %d"
                         % (len(code), contract.code_size))
    entry = emission.get("entry")
    if entry is not None and contract.entry is not None and entry != contract.entry:
        raise ValueError("entry contradicts the contract: emission %r, contract %r"
                         % (entry, contract.entry))
    prologue = contract.prologue
    prologue = bytes(prologue) if not isinstance(prologue, (bytes, bytearray)) else prologue
    stated = emission.get("prologue")
    if stated is not None:
        stated_bytes = bytes.fromhex(stated) if isinstance(stated, str) else bytes(stated)
        if stated_bytes != prologue:
            raise ValueError("prologue contradicts the contract: %d emission bytes against %d"
                             % (len(stated_bytes), len(prologue)))

    kernel = L.Kernel(code=code, name=contract.name, entry=contract.entry, prologue=prologue,
                      bindings=[L.Binding(index=b.index, readonly=not b.written,
                                          element_type=b.element_type) for b in contract.bindings])
    return link(kernel, emission, binding_offsets=[b.offset for b in contract.bindings],
                program_contract=contract)
