"""Independent delivered-image inspection for resident attention; never calls Metal."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def system_register_reads(code, decoded, abi):
    """Compare ABI's encoded SR selectors to delivered main-program reads.

    Decoder MCRegister IDs are another numbering domain: the tensor lane read
    prints reg:45 but its byte-1 selector is 130, also SR_SIMD_ELEM in the
    compiler's established table. ABI v3/v4 carry the encoded selectors.
    """
    import g17asm
    found = set()
    for offset, length, opcode, _fields in decoded:
        if opcode in (14059,14060):
            if length != 4:
                raise ValueError('unmodeled system-register read length')
            found.add(g17asm.decode_sr(code[offset:offset+length])['sr'])
    declared = abi.get('system_registers')
    if (not isinstance(declared,(list,tuple)) or
        any(type(x) is not int for x in declared) or
        list(declared) != sorted(found)):
        raise ValueError('delivered system-register selectors differ from ABI: '
                         'encoded %s, declared %s' % (sorted(found),declared))
    return sorted(found)


def threadgroup_metadata(metadata,abi,decoded):
    """Compare delivered fields with ABI storage and independently decoded program shape."""
    import g17gpumd
    memory=g17gpumd.threadgroup_declaration(metadata)
    if memory!={'memory_use':1,'static_memory_bytes':abi['threadgroup']['static_memory_bytes']}:
        raise ValueError(f'delivered threadgroup declaration differs from ABI: {memory}')
    # Authoring uses captured compiler layout. Admission counts the vendor-decoded
    # delivered bytes instead. This is the measured Apple emission rule, not a
    # claim about scheduling or the driver's interpretation of this field.
    count=len(decoded)
    expected=None if count<=30 else (3 if any(i[2]==458 for i in decoded) else (1 if count<=300 else 2))
    pk=g17gpumd.kernel_table(metadata)
    field=g17gpumd._table_slot(metadata,pk,32)
    actual=None if field is None else struct.unpack_from('<B',metadata,field)[0]
    if actual!=expected:
        raise ValueError(f'delivered slot 32 differs from instruction stream: {actual}, expected {expected}')
    return dict(**memory,program_shape_slot32=actual)


def inspect(bundle, *, expected_graph=None, requirements=None):
    import g17attentiongraph
    import g17attentiongraphcontract
    import g17archcheck
    import g17gpumd
    import g17packedcheck
    import g17storerank
    import g17scanlink
    bundle=Path(bundle)
    manifest=json.loads((bundle/"manifest.json").read_text())
    if manifest.get("format")!="g17-attention-images-v1":
        raise ValueError("unsupported attention image manifest")
    graph=manifest["graph"]
    # This milestone admits the actual application, not an arbitrary graph with
    # internally consistent bindings but different tensor layouts or operators.
    if expected_graph is None:
        expected_graph=g17attentiongraph.graph()
    elif requirements is None:
        raise ValueError("a new graph requires independent program requirements")
    if graph!=expected_graph:
        raise ValueError("graph differs from the fixed MiniLM attention application")
    initialized=frozenset()
    if requirements is not None:
        import g17buffergraph
        if graph.get('format')=='g17-resident-batches-v1':
            import g17bufferbatches
            findings=g17bufferbatches.validate(graph,requirements)
            if findings:raise ValueError('batch schedule refused: '+'; '.join(findings))
            # The original batch document was compared to the independent
            # expected plan above. Flatten only for per-program image checks.
            graph=g17bufferbatches.flatten(graph)
        else:initialized=g17buffergraph.validated_initializations(graph,requirements)
    programs=manifest["programs"]
    expected_programs={stage["program"] for stage in graph["stages"]}
    if set(programs)!=expected_programs:raise ValueError("program set differs from graph")
    contracts={}
    reports={}
    blockers=[]
    for name in sorted(programs):
        record=programs[name];abi=record["abi"]
        if requirements is not None:
            expected=requirements[name]
            if abi!=expected['abi'] or record['sha256']['program.bin']!=expected['code_sha256']:
                raise ValueError(f"{name}: image differs from independent program requirements")
        if abi["abi_version"] not in (3,4,5):raise ValueError(f"{name}: unsupported ABI")
        if abi['abi_version']==5 and requirements is None:
            raise ValueError('tensor image requires independent program requirements')
        contracts[name]=[(b["index"],b["offset"],b["written"],b["element_type"])
                         for b in abi["bindings"]]
        directory=bundle/"programs"/name
        files={file:(directory/file).read_bytes() for file in
               ("program.bin","program.o","program.lib.metallib","program.arc.metallib")}
        if {file:hashlib.sha256(data).hexdigest() for file,data in files.items()}!=record["sha256"]:
            raise ValueError(f"{name}: file hashes differ from manifest")
        obj=g17scanlink.verify_contract(files["program.arc.metallib"],files["program.lib.metallib"],
                                       [b[:3] for b in contracts[name]])
        if obj!=files["program.o"]:raise ValueError(f"{name}: archive object differs")
        sections,symbols=g17archcheck.object_contents(obj)
        entry=abi["entry"];code=files["program.bin"];end=entry+len(code);aligned=(end+15)&~15
        text=sections["__TEXT,__text"]
        if text!=bytes.fromhex(abi["prologue"])+code+bytes.fromhex("0600")*((aligned-end)//2):
            raise ValueError(f"{name}: delivered text, prologue or alignment differs")
        for symbol,address in (("_agc.main",entry),("_agc.main.constant_program",0)):
            if [(s[0],s[4]) for s in symbols if s[0]==symbol]!=[(symbol,address)]:
                raise ValueError(f"{name}: delivered entry symbol differs")
        if abi['abi_version']==5:
            import g17tensorimagecheck
            arch=g17tensorimagecheck.arch(sections["__GPU_ARCH_LD_MD,__compute"])
        else:
            arch=g17archcheck.decode_arch(sections["__GPU_ARCH_LD_MD,__compute"])
        if arch["serialized_flag"]!=abi["arch_flag"]:
            raise ValueError(f"{name}: delivered ARCH differs from semantic ABI")
        decoded=g17packedcheck.decode(code)
        actual_srs=system_register_reads(code,decoded,abi)
        actual=[(off,length,opcode) for off,length,opcode,_ in decoded]
        declared=[(i["offset"],i["length"],i["opcode"]) for i in record["instructions"]]
        if actual!=declared:raise ValueError(f"{name}: instruction boundaries or opcodes differ")
        if sorted({(opcode,length) for _,length,opcode in actual})!=sorted(map(tuple,abi["forms"])):
            raise ValueError(f"{name}: instruction forms differ from ABI")
        # EVERY STORE MUST NAME A BUFFER THE IMAGE DECLARES. Twice now a store form has inherited
        # its binding rank from a template instead of stating it, and twice a rank-1 delivery hid
        # it. Nothing compared the encoded rank against the declared bindings, which is the one
        # comparison that catches it on delivered bytes without a device.
        ranks=g17storerank.check(code,contracts[name])
        if ranks["status"]!="passed":
            raise ValueError("%s: delivered store rank names no declared writable binding: %s"
                             % (name,"; ".join(f["reason"] for f in ranks["findings"])))
        count=g17gpumd.register_count(sections["__GPU_METADATA,__compute"])
        if count is not None and count!=abi["register_count"]:
            raise ValueError(f"{name}: delivered register count differs from ABI")
        if count is None:
            # No class-wide allocation limit can be inferred from a few working
            # programs. Require an explicit count for this new attention release.
            blockers.append(f"{name}: image omits register allocation for {abi['register_count']} registers")
        reports[name]=dict(instructions=len(decoded),register_count=abi["register_count"],
            serialized_register_count=count,serialized_arch=arch["serialized_flag"],
            delivered_system_registers=actual_srs,
            delivered_bindings=[list(b[:3]) for b in contracts[name]])
        if abi['abi_version']==4:
            memory=threadgroup_metadata(sections['__GPU_METADATA,__compute'],abi,decoded)
            reports[name]['serialized_threadgroup']=memory
        if abi['abi_version']==5:
            reports[name]['tensor_resources']=g17tensorimagecheck.resources(
                sections['__GPU_METADATA,__compute'],sections['__GPU_LD_MD,__compute'],
                abi,decoded,contracts[name])
    grids=({name:r['exact_grid'] for name,r in requirements.items()} if requirements is not None
           else {s["program"]:s["grid"] for s in graph["stages"]})
    findings=g17attentiongraphcontract.validate(graph,contracts,grids,
        initialized_allocations=initialized)
    if findings:raise ValueError("graph/image disagreement: "+"; ".join(findings))
    image_blockers=list(blockers)
    blockers.append("complete delivered-byte arithmetic admission and staged semantic measurements pending")
    return dict(status="structurally_checked",programs=reports,stages=len(graph["stages"]),
                blockers=blockers,image_blockers=image_blockers,loader_eligible=False,gpu_dispatched=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("bundle",type=Path)
    args=parser.parse_args()
    try:print(json.dumps(inspect(args.bundle),indent=2))
    except (ValueError,KeyError,TypeError,OSError) as error:parser.exit(2,f"refused: {error}\n")
