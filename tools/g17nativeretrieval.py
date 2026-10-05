"""Sentence retrieval with a complete pretrained encoder running below Metal."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from g17textinput import prepare
from g17inferencesession import Session, ROOT, R


def run(bundle, tokenizer, query, documents, destination):
    if not 1 <= len(documents) <= 14:
        raise ValueError('refused: bounded session supports 1 through 14 documents plus query and repeat')
    texts = [query, *documents, query]
    encoded = [prepare(ROOT/'evidence/g17-inference-models-v1/minilm', tokenizer, text, limit=32)
               for text in texts]
    arrays = []
    for item in encoded:
        length = len(item['token_ids'])
        arrays.append(dict(token_ids=np.array(item['token_ids']+[0]*(32-length), np.int32),
                           position_ids=np.arange(32, dtype=np.int32),
                           token_type_ids=np.zeros(32, np.int32),
                           attention_mask=np.array([1]*length+[0]*(32-length), np.int32)))
    started = time.perf_counter()
    session = Session(bundle)
    verification_seconds = time.perf_counter()-started
    outputs = []; requests = []; error = None
    try:
        with session:
            for text, inputs, item in zip(texts, arrays, encoded, strict=True):
                rows = session.execute(inputs, readback=False)
                if any(row['data'] for row in rows[:-1]):
                    raise RuntimeError('production inference unexpectedly returned host intermediates')
                output = np.frombuffer(rows[-1]['data'], '<f4').copy()
                if output.shape != (384,) or not np.isfinite(output).all():
                    raise RuntimeError('invalid final sentence embedding')
                outputs.append(output)
                requests.append(dict(text=text, token_ids=item['token_ids'], output_sha256=R.sha(rows[-1]['data']),
                                     submissions=len(rows), intermediate_readback_bytes=0,
                                     final_readback_bytes=len(rows[-1]['data']),
                                     client_roundtrip_ns=session.roundtrip_ns,
                                     submit_call_ns=sum(row['submit_ns'] for row in rows),
                                     submit_to_completion_ns=sum(row['completion_ns'] for row in rows)))
    except (RuntimeError, TimeoutError, EOFError) as failure:
        error = type(failure).__name__ + ': ' + str(failure)
    repeat = len(outputs) == len(texts) and outputs[0].tobytes() == outputs[-1].tobytes()
    healthy = (getattr(session, 'native_returncode', None) == 0
               and getattr(session, 'before_recovery', None) == getattr(session, 'after_recovery', None)
               and getattr(session, 'new_events', None) == [])
    ranking = []
    if error is None and repeat and healthy:
        for index, output in enumerate(outputs[1:-1]):
            score = float(np.dot(outputs[0].astype(np.float64), output.astype(np.float64)) /
                          (np.linalg.norm(outputs[0].astype(np.float64))*np.linalg.norm(output.astype(np.float64))))
            ranking.append(dict(document_index=index, text=documents[index], cosine_similarity=score))
        ranking.sort(key=lambda row: row['cosine_similarity'], reverse=True)
    report = dict(format='g17-native-retrieval-v1', passed=error is None and repeat and healthy,
                  error=error, scope='complete pretrained MiniLM; 32-token bound; CPU tokenizer and ranking only',
                  runtime='persistent AGX/IOGPU below Metal; no Metal execution or Apple native body',
                  model_revision=encoded[0]['checkpoint_revision'], tokenizer=encoded[0]['tokenizer'],
                  tokenizer_engine=encoded[0]['tokenizer_engine'], requests=requests, ranking=ranking,
                  repeat_exact=repeat, embeddings=[array.tolist() for array in outputs],
                  graph_initializations=1, parameter_uploads=1, program_uploads=1,
                  bundle_manifest_sha256=R.sha((Path(bundle)/'manifest.json').read_bytes()),
                  native_binary_sha256=getattr(session, 'binary_sha256', None),
                  runtime_source_sha256=R.sha(R.SOURCE.read_bytes()),
                  runtime_header_sha256=R.sha((ROOT/'spike/agxsub/g17workload_inference.h').read_bytes()),
                  application_sha256=R.sha(Path(__file__).read_bytes()),
                  bundle_verification_seconds=verification_seconds,
                  native_build_seconds=getattr(session, 'build_seconds', None),
                  driver_prepare_seconds=getattr(session, 'prepare_seconds', None),
                  resident_allocation_bytes=sum(row['bytes'] for row in session.manifest['resource']['rows']
                                                if row['kind'] != 1) if 'rows' in session.manifest['resource'] else None,
                  before_recovery=getattr(session, 'before_recovery', None),
                  after_recovery=getattr(session, 'after_recovery', None),
                  new_events=getattr(session, 'new_events', None), native_log=getattr(session, 'native_log', ''),
                  numerical_validation='See independent complete-encoder receipt; ranking itself is not a numerical oracle',
                  timing_scope='Host elapsed time with guards enabled; diagnostic stage headers remain; not GPU timestamps or a performance comparison')
    Path(destination).write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('tokenizer', type=Path)
    parser.add_argument('--query', required=True)
    parser.add_argument('--document', action='append', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.bundle, args.tokenizer, args.query, args.document, args.receipt)
    print(json.dumps({key: report[key] for key in ('passed', 'error', 'ranking')}, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
