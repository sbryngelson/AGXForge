"""Admission for sequential bounded batches sharing one allocation lifetime.

This is a host scheduling contract, not native worker or numerical admission.
The old graph entry point retains its own limits. No buffer is relabelled as an
input at a batch boundary: coverage is established from the complete sequence.
"""
import copy
import g17buffergraph as G

MAX_BATCHES=8
MAX_STAGES=128


def partition(graph, batch_stages=MAX_STAGES):
    if type(batch_stages) is not int or not 1<=batch_stages<=MAX_STAGES:
        raise ValueError('batch size must be 1..128')
    if not isinstance(graph,dict) or graph.get('binding_windows')!=G.DISJOINT_V1:
        raise ValueError('batch plan requires disjoint-v1 windows')
    stages=graph.get('stages')
    if not isinstance(stages,list) or not stages or len(stages)>MAX_BATCHES*batch_stages:
        raise ValueError('batch plan requires 1..8 nonempty batches')
    plan=copy.deepcopy(graph);del plan['stages']
    plan['format']='g17-resident-batches-v1'
    plan['batches']=[copy.deepcopy(stages[i:i+batch_stages]) for i in range(0,len(stages),batch_stages)]
    return plan


def flatten(plan):
    if not isinstance(plan,dict) or plan.get('format')!='g17-resident-batches-v1' or 'stages' in plan:
        raise ValueError('expected unambiguous resident batch plan')
    if plan.get('binding_windows')!=G.DISJOINT_V1:
        raise ValueError('batch plan requires disjoint-v1 windows')
    batches=plan.get('batches')
    if not isinstance(batches,list) or not 1<=len(batches)<=MAX_BATCHES:
        raise ValueError('batch plan requires 1..8 nonempty batches')
    if any(not isinstance(b,list) or not 1<=len(b)<=MAX_STAGES for b in batches):
        raise ValueError('each batch requires 1..128 stages')
    graph=copy.deepcopy(plan);del graph['batches']
    graph['format']='g17-attention-graph-v1'
    graph['stages']=[s for batch in batches for s in batch]
    return graph


def validate(plan,requirements):
    try:graph=flatten(plan)
    except ValueError as error:return [str(error)]
    return G._validate(graph,requirements,batch_stage_count=len(graph['stages']))
