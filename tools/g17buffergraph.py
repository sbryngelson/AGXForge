"""CPU contract for typed buffer graphs, before image or Metal admission.

Program requirements are a separate input from the candidate graph. They must
come from the application's verified program delivery, never be inferred from
the graph under test. This checks allocation and launch compatibility only.
"""
import math
import re
import json

DISJOINT_V1 = 'disjoint-v1'
# 74 stages are delivered; the bound is the profile's, stated so a graph that grows past it is
# refused rather than quietly admitted. It is a runtime scheduling limit, not a hardware one.
DISJOINT_V1_STAGES = 128
RESIDENT_64 = 'resident-64-v1'
RESIDENT_64_ALLOCATIONS = 64


def validate(graph, requirements):
    return _validate(graph,requirements)


def _validate(graph, requirements, *, batch_stage_count=None):
    import g17attentiongraphcontract
    findings=[]
    def need(condition, message):
        if not condition: findings.append(message)
    def integer(value, minimum=0):
        return type(value) is int and value>=minimum
    def element_bytes(kind):
        return g17attentiongraphcontract.runtime_element_bytes(kind) if isinstance(kind,str) else None
    if not isinstance(graph,dict) or not isinstance(requirements,dict):
        return ["graph and program requirements must be mappings"]
    if 'textures' in graph or 'texture_bindings' in graph:
        return ['texture resources require common image and executor admission']
    allocations=graph.get("allocations")
    stages=graph.get("stages")
    if isinstance(stages,list) and any(isinstance(s,dict) and ('textures' in s or 'texture_bindings' in s) for s in stages):
        return ['texture stage bindings require common image and executor admission']
    if not isinstance(allocations,dict) or not allocations or not isinstance(stages,list) or not stages:
        return ["graph needs allocations and stages"]
    # Match the bounded storage/schedule profile in g17attentionstorage.h and
    # g17attentionschedule.h. These are runtime limits, not hardware limits.
    # THE PROFILE IS OPT-IN AND THE BOUND IS NAMED, NOT REMOVED. The whole-buffer graph keeps the
    # 32/32 limit it was measured under; a graph asking for binding_windows=disjoint-v1 is asking
    # for the windowed contract below and gets a stage bound of its own. Raising a limit because a
    # graph exceeded it is how a bound stops meaning anything, so the new one is stated here and
    # the old one is untouched for every graph that does not opt in.
    windows=graph.get('binding_windows')
    need(windows is None or windows==DISJOINT_V1,'unknown binding_windows profile')
    disjoint=windows==DISJOINT_V1
    writes=graph.get('write_policy')
    inplace=writes=='inplace-v1'
    multiple=writes in ('multiple-v1','inplace-v1')
    if inplace:
        need(graph.get('reply_selection')=='final-output-only',
             'inplace-v1 requires final-output-only reply selection')
    need('write_policy' not in graph or (multiple and windows is None),
         'unknown or incompatible write policy')
    extended=graph.get('allocation_profile')==RESIDENT_64 and disjoint and batch_stage_count is not None
    need('allocation_profile' not in graph or extended,'unknown or incompatible allocation profile')
    need(len(allocations)<=(RESIDENT_64_ALLOCATIONS if extended else 32),'runtime allocation limit exceeded')
    # Only the batch-plan validator supplies this count, after checking every
    # batch's own bound. All allocation and cross-stage coverage checks below
    # still run over the complete ordered sequence.
    stage_limit=batch_stage_count if batch_stage_count is not None and disjoint else (DISJOINT_V1_STAGES if disjoint else 32)
    need(len(stages)<=stage_limit,'runtime allocation/stage limit exceeded')
    program_limit=16 if batch_stage_count is not None and disjoint else 10
    program_names={s.get('program') for s in stages if isinstance(s,dict) and isinstance(s.get('program'),str)}
    need(len(program_names)<=program_limit,'runtime program limit exceeded')
    roles=[a.get('role') for a in allocations.values() if isinstance(a,dict)]
    need(roles.count('input')==1 and roles.count('output')==1,'runtime requires one input and one output')
    initialization=graph.get('output_initialization')
    sliced=initialization=='copy-input-slice-v1'
    byte_copy=initialization=='copy-input-bytes-v1'
    need('output_initialization_offset' not in graph or sliced or byte_copy,
         'output initialization offset requires slice or bytes mode')
    if 'output_initialization' in graph:
        need(initialization in ('copy-input-v1','copy-input-slice-v1','copy-input-bytes-v1') and windows is None,
             'unknown or incompatible output initialization')
        source=[a for a in allocations.values() if isinstance(a,dict) and a.get('role')=='input']
        output=[a for a in allocations.values() if isinstance(a,dict) and a.get('role')=='output']
        if len(source)==len(output)==1:
            src,out=source[0],output[0]
            if sliced or byte_copy:
                offset=graph.get('output_initialization_offset')
                width=element_bytes((out if byte_copy else src).get('element_type'))
                extent=src.get('payload_bytes');length=out.get('payload_bytes')
                need(integer(offset) and bool(width) and offset%width==0 and
                     (byte_copy or element_bytes(out.get('element_type'))==width) and integer(extent,1) and
                     integer(length,1) and offset<=extent and length<=extent-offset,
                     'output initialization slice exceeds or misaligns input/output extent')
            else:
                need(all(src.get(k)==out.get(k) for k in ('payload_bytes','element_type')),
                     'output initialization requires matching input/output extent and type')
    initialized=set()
    if 'intermediate_initialization' in graph:
        rows=graph['intermediate_initialization']
        valid_rows=isinstance(rows,list) and bool(rows)
        need(valid_rows and multiple and windows is None,'invalid intermediate initialization profile')
        seen_targets=set()
        source=[a for a in allocations.values() if isinstance(a,dict) and a.get('role')=='input']
        for row in rows if valid_rows else []:
            if not isinstance(row,dict) or set(row)!={'allocation','mode','source_offset'}:
                need(False,'invalid intermediate initialization record');continue
            target=row.get('allocation');offset=row.get('source_offset')
            valid_target=isinstance(target,str) and target in allocations and target not in seen_targets
            need(valid_target,'unknown or duplicate intermediate initialization target')
            if not valid_target:continue
            seen_targets.add(target);out=allocations[target]
            if not isinstance(out,dict) or len(source)!=1:
                need(False,'invalid intermediate initialization allocation');continue
            width=element_bytes(out.get('element_type'));extent=source[0].get('payload_bytes');length=out.get('payload_bytes')
            valid_initialization=(row['mode']=='copy-input-bytes-v1' and out.get('role')=='intermediate' and
                 integer(offset) and bool(width) and offset%width==0 and integer(extent,1) and
                 integer(length,1) and offset<=extent and length<=extent-offset)
            need(valid_initialization,'intermediate initialization mode, role, or byte extent differs')
            if valid_initialization:initialized.add(target)
    for name,a in allocations.items():
        if not isinstance(a,dict):
            findings.append(f"{name}: invalid allocation");continue
        shape=a.get("shape")
        valid_shape=isinstance(shape,list) and bool(shape) and all(integer(n,1) for n in shape)
        need(valid_shape,f"{name}: invalid shape")
        need(a.get("value_policy","finite-v1") in ("finite-v1","raw-bits-v1"),
             f"{name}: unsupported allocation value_policy")
        width=element_bytes(a.get("element_type"))
        need(width is not None,f"{name}: unsupported runtime buffer element type")
        need(a.get("role") in ("input","parameter","intermediate","output"),f"{name}: invalid role")
        for key in ("offset","payload_bytes","allocation_bytes"):
            need(integer(a.get(key)),f"{name}: invalid {key}")
        if valid_shape and width:
            need(a.get("payload_bytes")==width*math.prod(shape),f"{name}: shape and payload differ")
        if all(integer(a.get(k)) for k in ("offset","payload_bytes","allocation_bytes")):
            need(a['offset']==128 and a['allocation_bytes']==a['payload_bytes']+256,
                 f"{name}: expected 128-byte guards on each side")
            need(0<a['payload_bytes']<=64*1024*1024,f'{name}: runtime payload limit exceeded')
    if all(isinstance(a,dict) and integer(a.get('allocation_bytes')) for a in allocations.values()):
        need(sum(a['allocation_bytes'] for a in allocations.values())<=128*1024*1024,'runtime total allocation limit exceeded')
    contracts={};grids={};extents={};widths={}
    for name,record in requirements.items():
        if not isinstance(record,dict):
            findings.append(f"{name}: invalid program requirement");continue
        need(bool(re.fullmatch(r"[0-9a-f]{64}",str(record.get("code_sha256","")))),f"{name}: missing code identity")
        grid=record.get("exact_grid")
        need(isinstance(grid,list) and len(grid)==3 and all(integer(n,1) for n in grid),
             f"{name}: missing independent exact grid")
        abi=record.get("abi",{})
        if not isinstance(abi,dict):
            findings.append(f"{name}: invalid ABI");continue
        need('textures' not in abi and 'texture_bindings' not in abi,f'{name}: texture resource requirement')
        version=abi.get('abi_version')
        need(type(version) is int and version in (3,4,5),f"{name}: ABI v3/v4/v5 required")
        need(abi.get('writes_texture') is False,f'{name}: texture resource requirement')
        execution=abi.get('execution')
        if version==5:
            pool=abi.get('constant_pool')
            need(isinstance(pool,(list,tuple)) and
                 all(type(v) is int and 0<=v<=255 for v in pool),
                 f'{name}: ABI v5 requires explicit constant_pool bytes')
            from pydantic import TypeAdapter
            from g17abi import ExecutionABI, TENSOR_OPCODES
            try:TypeAdapter(ExecutionABI).validate_json(json.dumps(execution))
            except (ValueError,TypeError):need(False,f'{name}: invalid ABI v5 execution requirement')
            need(isinstance(execution,dict) and type(execution.get('simd_width')) is int and
                 execution.get('tensor') is True,f'{name}: strict execution requirement types')
            forms=abi.get('forms',[])
            need(isinstance(forms,(list,tuple)) and any(isinstance(f,(list,tuple)) and len(f)==2
                 and type(f[0]) is int and f[0] in TENSOR_OPCODES for f in forms),f'{name}: execution requirement without tensor forms')
        else:
            need(execution is None,f'{name}: execution requirement requires ABI v5')
        if version==4:
            from pydantic import TypeAdapter
            from g17abi import ThreadgroupABI
            need(abi.get('uses_threadgroup') is True,f'{name}: ABI v4 requires threadgroup use')
            try:TypeAdapter(ThreadgroupABI).validate_json(json.dumps(abi.get('threadgroup')))
            except (ValueError,TypeError):need(False,f'{name}: invalid ABI v4 threadgroup declaration')
        else:
            need(abi.get('uses_threadgroup') is False and abi.get('threadgroup') is None,
                 f'{name}: threadgroup resource requires ABI v4')
        bindings=abi.get("bindings")
        elements=record.get("binding_elements")
        if not isinstance(bindings,list) or not bindings or not isinstance(elements,list) or len(elements)!=len(bindings):
            findings.append(f"{name}: missing per-binding extents");continue
        indices=[];offsets=[];contract=[]
        for b,n in zip(bindings,elements):
            if not isinstance(b,dict):
                findings.append(f"{name}: invalid binding");continue
            need(integer(b.get('index')),f"{name}: invalid binding index")
            need(integer(b.get('offset')) and b['offset']%2==0,f"{name}: invalid descriptor offset")
            need(type(b.get('written')) is bool,f"{name}: invalid access flag")
            need(element_bytes(b.get('element_type')) is not None and
                 type(b.get('element_bytes')) is int and
                 b['element_bytes']==element_bytes(b.get('element_type')),f"{name}: unsupported element type")
            need(integer(n,1),f"{name}: invalid binding extent")
            if integer(b.get('index')):
                widths[(name,b['index'])]=b.get("element_bytes") if integer(b.get("element_bytes"),1) else 0
            indices.append(b.get('index'));offsets.append(b.get('offset'))
            contract.append((b.get('index'),b.get('offset'),b.get('written'),b.get('element_type')))
        if all(integer(i) for i in indices+offsets):
            need(len(set(indices))==len(indices) and len(set(offsets))==len(offsets),f"{name}: duplicate binding identity")
        contracts[name]=contract;grids[name]=grid;extents[name]=dict(zip(indices,elements)) if all(integer(i) for i in indices) else {}
        if 'program_contract' in record:
            from g17abi import ProgramABI
            try:
                captured=ProgramABI.from_dict(record['program_contract'])
                need(captured.code_sha256==record.get('code_sha256'),f'{name}: captured code identity differs')
                need([(b.index,b.offset,b.written,b.element_type) for b in captured.bindings]==contract,
                     f'{name}: captured binding contract differs')
                spill=captured.spill_state
                if spill is not None:
                    need(isinstance(grid,list) and len(grid)==3 and grid[1:]==[1,1],
                         f'{name}: spill launch requires an x-only grid')
                    if isinstance(grid,list) and len(grid)==3 and integer(grid[0],1):
                        need(extents[name].get(spill.binding_index)==grid[0]*spill.words_per_thread,
                             f'{name}: scratch extent differs from captured per-thread requirement')
                    need(any(i==spill.binding_index and w is True and typ=='uint' for i,_,w,typ in contract),
                         f'{name}: scratch must be a writable uint32 binding')
            except (TypeError,ValueError) as error:
                need(False,f'{name}: invalid captured program contract: {error}')
    names=[];used=set();produced={n for n,a in allocations.items() if isinstance(a,dict) and a.get('role') in ('input','parameter')}
    # UNDER disjoint-v1 THE UNIT IS THE BYTE RANGE, NOT THE ALLOCATION. `covered` holds, per
    # allocation, the half-open byte intervals published by stages already scheduled; a whole
    # input or parameter is published before the first stage runs. Reads are checked against it
    # and writes against overlap with it, both in stage order, so "produced by an earlier stage"
    # is decided by position in the list and not by any claim the graph makes about itself.
    covered={}
    if disjoint:
        for n,a in allocations.items():
            if isinstance(a,dict) and a.get('role') in ('input','parameter') and integer(a.get('offset')) and integer(a.get('payload_bytes'),1):
                covered[n]=[(a['offset'],a['offset']+a['payload_bytes'])]
            else:
                covered[n]=[]
    def merged(spans):
        out=[]
        for lo,hi in sorted(spans):
            if out and lo<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],hi))
            else:out.append((lo,hi))
        return out
    def contains(spans,lo,hi):
        return any(a<=lo and hi<=b for a,b in merged(spans))
    def overlaps(spans,lo,hi):
        return any(lo<b and a<hi for a,b in spans)
    for stage in stages:
        if not isinstance(stage,dict):
            findings.append('invalid stage');continue
        name=stage.get('name');program=stage.get('program');bindings=stage.get('bindings');grid=stage.get('grid')
        need(isinstance(name,str) and bool(name) and name not in names,'missing or duplicate stage name');names.append(name)
        if not isinstance(program,str) or program not in contracts:
            findings.append(f'{name}: no program requirement');continue
        used.add(program)
        execution=requirements[program]['abi'].get('execution')
        if execution is not None or 'execution' in stage:
            need(isinstance(execution,dict) and stage.get('execution')==execution,
                 f'{name}: execution requirement differs from independent ABI')
            actual_execution=stage.get('execution')
            need(isinstance(actual_execution,dict) and
                 type(actual_execution.get('simd_width')) is int and actual_execution.get('tensor') is True,
                 f'{name}: invalid stage execution requirement types')
            if isinstance(execution,dict) and integer(execution.get('simd_width'),1):
                width=execution['simd_width']
                need(isinstance(grid,list) and len(grid)==3 and integer(grid[0],1) and
                     grid[0]>=width and grid[0]%width==0,
                     f'{name}: execution requires complete SIMD groups along x')
        if requirements[program]['abi'].get('abi_version')==4:
            need('threadgroup' in stage,f'{name}: missing required threadgroup launch')
        if 'threadgroup' in stage:
            declared=requirements[program]['abi'].get('threadgroup')
            need(requirements[program]['abi'].get('uses_threadgroup') is True and
                 isinstance(declared,dict) and stage['threadgroup']==declared,
                 f'{name}: threadgroup launch differs from independent ABI')
            if isinstance(declared,dict):
                group=declared.get('required_size')
                if (isinstance(group,(tuple,list)) and len(group)==3 and all(integer(n,1) for n in group)
                        and isinstance(grid,list) and len(grid)==3 and all(integer(n,1) for n in grid)):
                    need(all(n%g==0 for n,g in zip(grid,group)),f'{name}: partial threadgroup')
        need(isinstance(grid,list) and len(grid)==3 and all(integer(n,1) for n in grid),f'{name}: missing or invalid launch grid')
        if isinstance(grid,list) and len(grid)==3 and all(integer(n,1) for n in grid):
            need(math.prod(grid)<=16*1024*1024,f'{name}: runtime grid limit exceeded')
        if not isinstance(bindings,list) or not bindings:
            findings.append(f'{name}: missing bindings');continue
        # inplace-v1 is an ordered whole-payload rewrite contract, not an alias
        # or metadata admission. Only an earlier shader write establishes a
        # rewrite target; CPU initialization alone does not do so.
        claims=stage.get('inplace_allocations')
        valid_claims=isinstance(claims,list) and all(isinstance(a,str) for a in claims)
        if inplace:
            need(valid_claims and len(set(claims))==len(claims),
                 f'{name}: inplace allocations must be an explicit duplicate-free list')
        written=[];seen=[];rewrites=set()
        for b in bindings:
            if not isinstance(b,dict):
                findings.append(f'{name}: invalid binding');continue
            allocation=b.get('allocation');index=b.get('index')
            need(isinstance(allocation,str) and allocation in allocations,f'{name}: unknown allocation')
            need(allocation not in seen,f'{name}: allocation aliasing is not supported');seen.append(allocation)
            need(integer(index),f'{name}: invalid binding index')
            need(type(b.get('written')) is bool,f'{name}: invalid written flag')
            need(integer(b.get('offset')) and integer(b.get('length'),1),f'{name}: invalid binding window')
            if integer(index) and index in extents[program] and integer(extents[program][index],1):
                need(b.get('length')==widths[(program,index)]*extents[program][index],f'{name}: binding {index} extent differs from program')
            # THE WINDOW MUST LIE IN THE PAYLOAD, NOT THE GUARD. Each allocation carries 128 bytes
            # of guard on each side; a binding that reaches into one is addressing memory the
            # graph declared as a fence, and the alignment check is against the payload's own
            # start so an element-aligned window stays element-aligned wherever it sits.
            if disjoint and isinstance(allocation,str) and allocation in allocations:
                a=allocations[allocation]
                width=element_bytes(a.get('element_type')) if isinstance(a,dict) else None
                if (isinstance(a,dict) and integer(a.get('offset')) and integer(a.get('payload_bytes'),1)
                        and integer(b.get('offset')) and integer(b.get('length'),1) and width):
                    lo,hi=b['offset'],b['offset']+b['length']
                    base,end=a['offset'],a['offset']+a['payload_bytes']
                    need(base<=lo and hi<=end,f'{name}: binding window leaves the payload')
                    need((lo-base)%width==0 and b['length']%width==0,
                         f'{name}: binding window is not {width}-byte element aligned')
                    if b.get('written') is True:
                        need(not overlaps(covered.get(allocation,[]),lo,hi),
                             f'{name}: write window overlaps a range an earlier stage published')
                    else:
                        need(contains(covered.get(allocation,[]),lo,hi),
                             f'{name}: reads {allocation} bytes no earlier stage produced')
            if b.get('written') is True:
                if not disjoint:
                    a=allocations.get(allocation) if isinstance(allocation,str) else None
                    if inplace:
                        need(isinstance(a,dict) and a.get('role') in ('intermediate','output'),
                             f'{name}: inplace policy cannot write input or parameter allocations')
                        if isinstance(allocation,str) and allocation in produced:
                            rewrites.add(allocation)
                    else:
                        need(isinstance(allocation,str) and allocation not in produced,f'{name}: overwrites an existing allocation')
                written.append(allocation)
        if inplace and valid_claims:
            need(set(claims)==rewrites,
                 f'{name}: inplace allocations differ from previously-produced written allocations')
        if multiple:
            need(bool(written),f'{name}: no written allocation')
            need(isinstance(stage.get('result_allocation'),str) and stage['result_allocation'] in written,
                 f'{name}: result allocation must name a written binding')
        else:
            need('result_allocation' not in stage,f'{name}: result allocation requires multiple-v1')
            need(len(written)==1,f'{name}: exactly one new output is supported')
        if all(isinstance(a,str) for a in written):produced.update(written)
        # Publish this stage's window only AFTER its own reads were checked, so a stage cannot
        # satisfy its own read with its own write - the same-launch aliasing the profile refuses,
        # which the per-stage `seen` check already refuses at allocation granularity.
        if disjoint:
            for b in bindings:
                if (isinstance(b,dict) and b.get('written') is True and isinstance(b.get('allocation'),str)
                        and b['allocation'] in covered and integer(b.get('offset')) and integer(b.get('length'),1)):
                    covered[b['allocation']].append((b['offset'],b['offset']+b['length']))
    need(used==set(requirements),'program registry and graph program set differ')
    # Native Prepare/Begin copy these complete payloads before stage execution.
    # Keep them separate from shader publications: initialized scratch may still
    # receive its first shader write, but never a second one.
    need(produced|initialized==set(allocations),'graph leaves allocations unproduced')
    if multiple:
        final=stages[-1].get('result_allocation') if isinstance(stages[-1],dict) else None
        need(isinstance(final,str) and isinstance(allocations.get(final),dict) and
             allocations[final].get('role')=='output','final result allocation must be the output')
    # COMPLETE, WITH NO HOLES. Being written somewhere is not the same as being written
    # everywhere: under disjoint-v1 an intermediate or output allocation must end the graph with
    # its whole payload published, or a later reader takes bytes nobody wrote. Checked as one
    # merged run covering exactly the payload, so a gap in the middle fails as loudly as a short tail.
    if disjoint:
        for n,a in allocations.items():
            if not isinstance(a,dict) or a.get('role') in ('input','parameter'):continue
            if not (integer(a.get('offset')) and integer(a.get('payload_bytes'),1)):continue
            runs=merged(covered.get(n,[]))
            need(runs==[(a['offset'],a['offset']+a['payload_bytes'])],
                 f'{n}: payload is not completely produced ({len(runs)} run(s) published)')
    # Structural checks above protect the older typed binding checker from
    # malformed data; its independent range/read-order checks remain in use.
    if findings:return findings
    return list(g17attentiongraphcontract.validate(graph,contracts,grids,
        initialized_allocations=initialized))


def require(graph, requirements):
    findings=validate(graph,requirements)
    if findings:raise ValueError("buffer graph refused: "+"; ".join(findings))
    return dict(status="graph_contract_checked",stages=len(graph['stages']),
                programs=sorted(requirements),gpu_dispatched=False,loader_eligible=False,
                limitations="Requires independent image identity, delivered arithmetic, source verification and hardware admission.")


def validated_initializations(graph, requirements):
    """Return source-byte producer facts only after the entire graph is valid."""
    require(graph,requirements)
    return frozenset(row['allocation'] for row in graph.get('intermediate_initialization',[]))
