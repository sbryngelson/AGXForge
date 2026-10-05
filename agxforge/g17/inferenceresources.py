"""Physical byte layout of lowered model stages, without GPU addresses/admission."""
import math
from .inferencegraph import ELEMENT_BYTES, Unsupported


def plan(graph, lowering):
    if lowering['format']!='g17-native-model-lowering-v1' or lowering['gpu_admitted'] is not False:
        raise Unsupported('refused: expected native lowering contract')
    descriptors=dict(graph['tensors'],**lowering['prepared_parameters'],**lowering['temporary_tensors'])
    used={b['name'] for s in lowering['stages'] for b in s['bindings']}
    controls=lowering.get('host_control_inputs',[])
    if any(name not in graph['tensors'] or graph['tensors'][name]['role']!='input' for name in controls):
        raise Unsupported('refused: native host control must be a declared input')
    used.update(controls)
    outputs=set(graph['outputs'])
    if not outputs <= used:
        raise Unsupported('refused: missing native model output')
    roles={};birth={};death={};sizes={}
    for name in used:
        if name not in descriptors:
            raise Unsupported('refused: unresolved native buffer')
        descriptor=descriptors[name]
        sizes[name]=math.prod(descriptor['shape'])*ELEMENT_BYTES[descriptor['dtype']]
        role=descriptor.get('role','parameter' if name in lowering['prepared_parameters'] else 'intermediate')
        roles[name]='parameters' if role=='parameter' else 'inputs' if role=='input' else 'state' if role=='state' else 'activations'
        if roles[name]!='activations':birth[name]=-1;death[name]=len(lowering['stages'])
    state_names={n for n in used if roles[n]=='state'}
    if state_names:
        effects=[n for n in graph['nodes'] if n['op']=='kv_cache_update']
        if lowering.get('state_effects')!=effects or {n for effect in effects for n in effect['attrs']['writes']}!=state_names:
            raise Unsupported('refused: stateful native layout requires exact decoder write effects')
        if set(controls)!={effect['inputs'][3] for effect in effects}:
            raise Unsupported('refused: decoder cache bounds control must survive lowering')
    for step,stage in enumerate(lowering['stages']):
        for b in stage['bindings']:
            name=b['name']
            if b['bytes']!=sizes[name]:raise Unsupported('refused: binding byte extent mismatch')
            if b['written']:
                if roles[name] not in ('activations','state'):raise Unsupported('refused: write to immutable native buffer')
                birth.setdefault(name,step)
        for b in stage['bindings']:
            name=b['name']
            if name not in birth:raise Unsupported('refused: native read before producer')
            death[name]=max(death.get(name,step),step)
    for name in outputs:death[name]=len(lowering['stages'])
    arenas=dict(parameters=0,inputs=0,activations=0)
    if state_names:arenas['state']=0
    slots=[];regions={}
    for name in sorted(used,key=lambda n:(birth[n],n)):
        capacity=(sizes[name]+255)//256*256;arena=roles[name]
        free=[s for s in slots if s['last_use']<birth[name] and s['capacity']>=capacity] if arena=='activations' else []
        if free:
            slot=min(free,key=lambda s:(s['capacity'],s['offset']));slot['last_use']=death[name]
        else:
            slot=dict(offset=arenas[arena]+256,capacity=capacity,last_use=death[name])
            arenas[arena]+=capacity+512
            if arena=='activations':slots.append(slot)
        regions[name]=dict(arena=arena,offset=slot['offset'],bytes=sizes[name],capacity=slot['capacity'],
                           guard_before=slot['offset']-256,guard_after=slot['offset']+slot['capacity'],guard_bytes=256,
                           birth=birth[name],last_use=death[name],readonly=arena in ('parameters','inputs'),dtype=descriptors[name]['dtype'])
    stages=[]
    for stage in lowering['stages']:
        stages.append(dict(stage,bindings=[dict(b,region=regions[b['name']]) for b in stage['bindings']]))
    return dict(format='g17-native-resident-layout-v1',gpu_admitted=False,status='byte_offsets_not_published',
                arenas=arenas,regions=regions,stages=stages,activation_slots=len(slots),
                guard_policy='slot outer guards always preserved; active tensor tail padding must be reseeded and checked on slot reuse',
                pending=['GPU addresses/publication','active guard and readonly execution checks','persistent request protocol'])
