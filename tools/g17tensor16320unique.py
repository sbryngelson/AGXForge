#!/usr/bin/env python3
"""Field-build long 16-body graphs with a distinct B address per body.

The authored image and logical B stream are new. The 64-1024 MiB B
apertures, request/page shifts, and low binding-metadata carrier extend
the tested 34 MiB graph; every changed field is checked before a pure
IOGPU Submit. Interior-marker probes retain the code and graph geometry.
"""
import hashlib
from pathlib import Path
import struct

import g17authoredtensorsg4h12loop8160runtime as parent
import g17structuredrequests as requests_base
import g17tensoraliaswitness as alias_witness

# Make agxforge.g17 importable when this tool runs from outside the checkout.
import sys as _g17_sys
from pathlib import Path as _G17Path
_g17_root = str(_G17Path(__file__).resolve().parents[1])
if _g17_root not in _g17_sys.path:
    _g17_sys.path.insert(0, _g17_root)
from agxforge.g17 import cc, tensorlife

TRIPS = 1020
BODIES = 16
B_BYTES = TRIPS * BODIES * 4096
B_SIZE = 64 << 20
SHIFT = B_SIZE - parent.B_SIZE
CODE_SHA256 = 'afae7236bcf9e6afeb07e7d035e80a5350e1e09df072cd5045efbb519123a5c3'
# Fixed after offline construction/reference and before the first Submit.
A_SHA256 = '704925258311cda5ba7385ec934a65d9a0e33b0129e2482dde16b9e17daa9587'
B_SHA256 = '107459f5610ed77176cf5f2b0da5d40a4f44741992ec9fa272fb43607e121508'
PAGE_SHA256 = 'bfc09f3397a242001e0b40419b5eee3e55b5a641799238199fb54d8c49a3a80a'
REQUEST_SHA256 = 'aa94c00cc34f6e4a78ecf6f1d3d511b8b758d37693189dd24f0231615aba2611'
PAYLOAD_SHA256 = '55383b530719ed6c0f272adeaa9f4da41ccd689a119174752eecc0d8c21b2bb8'
OUTPUT_SHA256 = 'e91da36c62db53c33369791cdf610c78675c1ae73ec0cc7dc7f482f29ddbdd7e'
ONE_TRIP_SHORT_SHA256 = '6caffb49e8b1f7186fd1d23d0d5e768b7603b2ac90a6e0b67ad516bc39699a94'
VARIANTS = {
    1020: dict(b_size=B_SIZE, code=CODE_SHA256, a=A_SHA256, b=B_SHA256,
               page=PAGE_SHA256, request=REQUEST_SHA256,
               payload=PAYLOAD_SHA256, output=OUTPUT_SHA256,
               one_trip_short=ONE_TRIP_SHORT_SHA256, marker=68),
    1530: dict(b_size=96 << 20,
               code='fde7c64045550ac2f070358c618a7b59391de768714379c2f9517fc07db0a92e',
               a='139018207e524de406a830afa3502edf64d7f358a26a52bc3458e741719c4a5b',
               b='6b9f4bfa9732a35ca22660ed3b69dbff4ce680842b351c7528a9964b82e4e590',
               page='fe2f17989fd7fa528e41bd0f2da9ff52e7d4edba7664da85d83bc224e144ee05',
               request='a897eeda651ce91fc4f57ea4e505a23227e873abbe716f4a2ff6e7d16290d9d5',
               payload='c82f07442e697abb84e882577911dadd4cae40d79003bce34f6f640c8d81f169',
               output='d32ff50cb4d8cdfd5a266dd13ff1be2ae2e36bb99a70bdb4d9fe25260aa72d3b',
               one_trip_short='6caffb49e8b1f7186fd1d23d0d5e768b7603b2ac90a6e0b67ad516bc39699a94',
               marker=102),
    2040: dict(b_size=128 << 20,
               code='a5f323ebfeea9a4e3fffa00ee2acde19dbf76e1e992fce6ee108fe660f4f27b1',
               a='a6a3011d9c51633000bf0dd4271b37150a36d0a62554b91911008f33a7efb720',
               b='2f000dab190ae6f6bf171f8707814c50b70b9f5e8ba5fa1840962274bd59a173',
               page='8e52f1ff70271dd4137e87f5222ac411f671c33488348c5b5bcfceebc263f334',
               request='ee69f6606bb4779aaa29800c63eb0948c44075059ec0adc4de3af2edff29eec4',
               payload='8f0b6ebb136da196821812d99f88c0636b05db777116467222b9860a698f30c5',
               output='47240417702761c4426aaf2f8dd2676350aee5ed08bb581eefd95332765d4d9a',
               one_trip_short='6caffb49e8b1f7186fd1d23d0d5e768b7603b2ac90a6e0b67ad516bc39699a94',
               marker=136,
               probes={
                   0: dict(b='5a1724440104f292fd7068764a6558d851e407577a0255636e93db689e2eee80',
                           payload='efd033dc6633e1c980af25f3c4f8c3dceec97969ea098959db9f508e725609b4',
                           output='0d41ef23a6dbddf190925412776f77d11a51ccda3538a334944abcca6a558950'),
                   8: dict(b='2f3a6526a5b4577b93e93ba50c963eb86a96a981755dd3ecce2e617083f5f624',
                           payload='6fafc6a874fbfd12d448b21d86028e913bf798b1edb76596b105599aa6dc4994',
                           output='6fc6347a80ab19b9e9527c0ed0d49932703f46d03a31a6a1a09bf547c221518b'),
               }),
    4080: dict(b_size=256 << 20,
               code='ca139fe3694aa1e27c646be4108741cf3b926d968e6c6092d8a3a3329b7a8d18',
               a='448cbeb4c4131c909ac3e9df9a5f300bf7410e465a774df32223a4544fb5b207',
               b='a138e5bf5947c8583dc84f7a00f265e4651c83261f48ae92b6b9f494b40a5055',
               page='e6a6a240d5a7e892e689b277d94f26538ebb0e9419e77e4ce9b1bac5bc72dcb9',
               request='70710e20a6dc0da63f6a9f959367564786cbe16e8cf1f65a0343376c8423055a',
               payload='177fb7b5174eec8afacc6d3ec8a9860fca3146b64af1bf638f6eb35954dcf0db',
               output='4970dde34647d885f3bee387296eef10443cdd4bcab04578703f8e50f48e7c37',
               one_trip_short='6caffb49e8b1f7186fd1d23d0d5e768b7603b2ac90a6e0b67ad516bc39699a94',
               marker=272),
    8160: dict(b_size=512 << 20,
               code='517a41da7e955665b5770869bdf81b56683cf2375879eb12dfc351d6ab3bfa5f',
               a='b1bf5b8cce35e252a60bd6fbf827408d6a3df5bbbe6ce61929a206c025ec9520',
               b='f295d1217f58506a9b2f47a638d09f448a75ad2c73e2a509d92ffa3bf42114a5',
               page='8ffe49c8014814197363a837ed24fd026f3e9e3a2df5264d23ff0b4b1aa4db8c',
               request='0963bd116a7d6b5763e5b857505dfc4a9b5ca7e4e9729541023b805a9edcb189',
               payload='dd85872b73412d45d6f8f8af26b20166637ef0139f7f40fc80c3e4e7262ffbee',
               output='f30f686fe950d06c112212b65398cca6f3b7305bb44d59a20dedac59440c5bb6',
               one_trip_short='6caffb49e8b1f7186fd1d23d0d5e768b7603b2ac90a6e0b67ad516bc39699a94',
               marker=544),
    16320: dict(b_size=1024 << 20,
                code='742ce5c87905637092808531ef1d5d00c10b3fb7c82b5dcbcd4453670a66a5cc',
                a='9fd8bbb52a4e54feb29d4de8d6c5129507ecbddb03d0df177ce84417ac6443b0',
                b='edc5846e6d7e109977dd51391f68ab37bff585715143d258e472973fb364d335',
                page='35849195510a45c987cff8b5c3787ceac7016453ee069b40d83fe7906c16b6f0',
                request='5b2ea16cfc14a3edf43cdc62b8730f3069e497b8c23775e529c858ddfb42f921',
                payload='87cff5ca995dbebbadc59af34c1b97505e5f1258030208a2b76f35ef8d292240',
                output='cf73118376c27c9e7d4f381fecbfd5292d82fe2398adc55aac2c88c274beb50a',
                one_trip_short='6caffb49e8b1f7186fd1d23d0d5e768b7603b2ac90a6e0b67ad516bc39699a94',
                marker=1088,
                mirror_payload='0b7e8c72ee0c55488c18bf3d543318d10bd94df6a7ff10fae7523ea34688114f',
                b_tail_payload='201becddd369a9f13ba599ff4099fa6870c06e54ba3964b8aba441709301d5f9',
                alias_payload='5a201df0940f25af22eb6edd9e8a45f8826be2dd6c61c0bdb8dccc2905127a5f',
                alias_b='87cd73ec60722ea48c020c658e6b20940a428dd51306e878212dc96bc05d6a37',
                alias_output='400a5eccdc5d93fffbd1a7d83e657aaceeae681b4bd8c3e1e621b1eaa025cb6f',
                alias_one_trip_short='7f437821908bc83421a2fd90f08db8dd843c611a166a11de86cc0261d4ecc44d',
                alias_zero_shader_payload='3aca99d5c785b332f73d6e1d7a5f5d4ce0d7fd2ee0fa65ed3924c8574817ccbc',
                alias_zero_shader_b='60c13536affc82b6ee887f72fd43b6c04db07f96fe74ed91031c627b9362637c',
                alias_zero_shader_output='400a5eccdc5d93fffbd1a7d83e657aaceeae681b4bd8c3e1e621b1eaa025cb6f',
                alias_zero_shader_one_trip_short='7f437821908bc83421a2fd90f08db8dd843c611a166a11de86cc0261d4ecc44d',
                alias_witness_payload='4c4313e8d38444a631c628550c2049fbc5be1e5e6581d613d232a0cd6da23a71',
                alias_witness_b='460c3da2056759f0b714fe567172fc586c1a5e7115d0a390671af1f022718818',
                alias_witness_high_output='400a5eccdc5d93fffbd1a7d83e657aaceeae681b4bd8c3e1e621b1eaa025cb6f',
                alias_witness_high_one_trip_short='ebd99d32317354cca5b2a2e164e74aa67df1fbebc3d3866d35631aeb52b10784',
                alias_witness_low_output='cc7439198fdcf16f6ef9136c7b2d94e3f46382346b914d575f90e675ed19dbd7',
                tailffff_witness_payload='9cadcafe81f73886fe74a01a4cf7d13a9efb807d97f4d18b694a167e38fa0ce3',
                tailfffe_witness_payload='fa380997871c87f787d5f456ab8d2820e8731d85af08bb88aa82c547edb866fd',
                taildual_fffd_tensor_payload='6978362eaa6eca4a5f66419162657b2ed449b77887df27b05a6cc8261bdec554',
                taildual_ffff_witness_payload='8e07fef091c36899028bd4813bb3c39f429b2927b39260494d63baeb930afee6'),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def code(trips=TRIPS):
    variant = VARIANTS[trips]
    raw = bytes(cc.compile_function(parent.function(cap=trips)).code)
    proof = tensorlife.counted_loop_check(
        raw, trips, carried=tuple(cc._TENSOR_INDEX_USED), runtime=True)
    if (len(raw) != 56240 or (variant['code'] and sha(raw) != variant['code']) or
            proof['back_edges'] != 1 or proof['trips'] != trips or
            proof['advances'] != {'R125': BODIES} or proof['hazards'] or
            proof['latch_releases']):
        raise ValueError('unique-address machine-code proof changed')
    return raw


def bstream(source, trips=TRIPS, marker_body=BODIES - 1):
    variant = VARIANTS[trips]
    if marker_body != BODIES - 1 and (trips != 2040 or marker_body not in variant['probes']):
        raise ValueError('unsupported interior marker body')
    if len(source) < 16384:
        raise ValueError('short source B')
    marker = b''.join(struct.pack('<e', variant['marker'] * struct.unpack_from('<e', source, i)[0])
                      for i in range(0, 4096, 2))
    raw = source[:16384] + source[:4096] * (trips * BODIES - 5) + marker
    if marker_body != BODIES - 1:
        raw = bytearray(raw)
        start = (trips * BODIES - BODIES + marker_body) * 4096
        raw[-4096:] = source[:4096]
        raw[start:start + 4096] = marker
        raw = bytes(raw)
    pinned_b = (variant['probes'][marker_body]['b'] if marker_body != BODIES - 1
                else variant['b'])
    if len(raw) != trips * BODIES * 4096 or (pinned_b and sha(raw) != pinned_b):
        raise ValueError('unique-address logical B stream changed')
    return raw


def graph(program, a, b, c, trips=TRIPS, marker_body=BODIES - 1,
          lookup_arm='original'):
    variant = VARIANTS[trips]
    if lookup_arm != 'original' and (trips != 16320 or lookup_arm not in ('mirror','b_tail','alias0048','alias0048_zero_shader','alias0048_witness','tailffff_witness','tailfffe_witness','taildual_fffd_tensor','taildual_ffff_witness')):
        raise ValueError('unsupported allocation-25 lookup arm')
    if marker_body != BODIES - 1 and (trips != 2040 or marker_body not in variant['probes']):
        raise ValueError('unsupported interior marker body')
    pinned = variant['probes'][marker_body] if marker_body != BODIES - 1 else variant
    expected_payload = (variant['mirror_payload'] if lookup_arm == 'mirror' else
                        variant['b_tail_payload'] if lookup_arm == 'b_tail' else
                        variant['alias_payload'] if lookup_arm == 'alias0048' else
                        variant['alias_zero_shader_payload'] if lookup_arm == 'alias0048_zero_shader' else
                        variant['alias_witness_payload'] if lookup_arm == 'alias0048_witness' else
                        variant['tailffff_witness_payload'] if lookup_arm == 'tailffff_witness' else
                        variant['tailfffe_witness_payload'] if lookup_arm == 'tailfffe_witness' else
                        variant['taildual_fffd_tensor_payload'] if lookup_arm == 'taildual_fffd_tensor' else
                        variant['taildual_ffff_witness_payload'] if lookup_arm == 'taildual_ffff_witness' else
                        pinned['payload'])
    b_size = variant['b_size']
    shift = b_size - parent.B_SIZE
    if (program != code(trips) or len(a) != 98304 or
            struct.unpack_from('<I', a, 0x17ff0)[0] != trips or
            len(b) != trips * BODIES * 4096 or
            (variant['a'] and sha(a) != variant['a']) or
            sha(b) != pinned['b']):
        raise ValueError('exact long-loop code and A/B input required')
    a510 = bytearray(a)
    struct.pack_into('<I', a510, 0x17ff0, 510)
    old_b = parent.bstream(b[:16384])
    alloc = parent.allocations(parent.code(), bytes(a510), old_b, c)
    raw = bytearray(alloc[0])
    raw[0x6c0:0x6c0 + len(program)] = program
    if lookup_arm in ('alias0048_witness','tailffff_witness','tailfffe_witness','taildual_fffd_tensor','taildual_ffff_witness'):
        alternate = alias_witness.code()
        if any(raw[alias_witness.CODE_OFFSET:alias_witness.CODE_OFFSET + len(alternate)]):
            raise ValueError('alternate code slot is not empty')
        raw[alias_witness.CODE_OFFSET:alias_witness.CODE_OFFSET + len(alternate)] = alternate
    alloc[0] = bytes(raw)
    raw = bytearray(alloc[19])
    struct.pack_into('<I', raw, 0x80 + 0x17ff0, trips)
    alloc[19] = bytes(raw)
    raw = bytearray(b_size)
    raw[:0x80] = b'\xa5' * 0x80
    raw[0x80:0x80 + len(b)] = b
    raw[0x80 + len(b):0x100 + len(b)] = b'\xa5' * 0x80
    alloc[20] = raw

    rows = [list(requests_base.ROW.unpack_from(parent.requests(),
            i * requests_base.ROW.size)) for i in range(29)]
    rows[20][6] = b_size
    rows[20][7] = requests_base.request(0x10001, 0x470, b_size,
                                       base_buffer=True)
    for index in range(21, 29):
        rows[index][5] += shift
    request = b''.join(requests_base.ROW.pack(*row) for row in rows)
    page = bytearray(parent.pages())
    for offset in parent.PAGE_POINTERS:
        struct.pack_into('<Q', page, offset,
                         struct.unpack_from('<Q', page, offset)[0] + shift)
    page = bytes(page)
    c_binding = parent.C_BINDING + shift
    raw = bytearray(alloc[28])
    struct.pack_into('<Q', raw, 0x1bb0, c_binding)
    alloc[28] = bytes(raw)
    if len(alloc[17]) != 0x20000 or any(alloc[17]) or len(alloc[28]) != 0xc000:
        raise ValueError('low binding-metadata carrier changed')
    alloc[17] = alloc[28] + bytes(len(alloc[17]) - len(alloc[28]))
    raw = bytearray(alloc[23])
    metadata_coordinate = (rows[17][5] >> 13) & 0xffff
    struct.pack_into('<H', raw, 6, metadata_coordinate)
    alloc[23] = bytes(raw)
    raw = bytearray(alloc[25])
    record_coordinate = (rows[23][5] >> 14) & 0xffff
    if lookup_arm == 'mirror':
        mirror_offset = 0x10000
        if len(alloc[23]) != 0x8000 or mirror_offset + len(alloc[23]) > len(alloc[17]):
            raise ValueError('allocation-23 record does not fit low carrier')
        low_va = rows[17][5] + mirror_offset
        record_coordinate = (low_va >> 14) & 0xffff
        if record_coordinate != 0x001a:
            raise ValueError('low record lookup coordinate changed')
        low = bytearray(alloc[17])
        low[mirror_offset:mirror_offset + len(alloc[23])] = alloc[23]
        alloc[17] = bytes(low)
    elif lookup_arm in ('taildual_fffd_tensor','taildual_ffff_witness'):
        record_coordinate = 0xfffd if lookup_arm == 'taildual_fffd_tensor' else 0xffff
        tensor_offset = rows[0][5] + (0xfffd << 14) - rows[20][5]
        witness_offset = rows[0][5] + (0xffff << 14) - rows[20][5]
        if (tensor_offset != 0x3ff54000 or witness_offset != 0x3ff5c000 or
                witness_offset - tensor_offset != len(alloc[23]) or
                tensor_offset < 0x100 + len(b) or
                witness_offset + len(alloc[23]) > len(alloc[20])):
            raise ValueError('dual records are not disjoint in unused B tail')
        tensor_record = alloc[23]
        witness_record = bytearray(tensor_record)
        address = alias_witness.CODE_VA
        struct.pack_into('<H', witness_record, 0x40, (address & 0xffff) | 7)
        struct.pack_into('<H', witness_record, 0x46, (address >> 16) & 0xffff)
        struct.pack_into('<H', witness_record, 0x48, (address >> 32) & 0xffff)
        if bytes(witness_record[0x40:0x4a]) != bytes.fromhex('07f0580e000000000001'):
            raise ValueError('alternate LoadShader address changed')
        alloc[20][tensor_offset:tensor_offset+len(tensor_record)] = tensor_record
        alloc[20][witness_offset:witness_offset+len(witness_record)] = witness_record
    elif lookup_arm in ('b_tail','tailffff_witness','tailfffe_witness'):
        record_coordinate = (0xfffe if lookup_arm == 'tailfffe_witness' else
                             0xffff if lookup_arm == 'tailffff_witness' else 0xfff0)
        target_va = rows[0][5] + (record_coordinate << 14)
        tail_offset = target_va - rows[20][5]
        expected_offset = (0x3ff58000 if lookup_arm == 'tailfffe_witness' else
                           0x3ff5c000 if lookup_arm == 'tailffff_witness' else 0x3ff20000)
        if (tail_offset != expected_offset or
                tail_offset < 0x100 + len(b) or
                tail_offset + len(alloc[23]) > len(alloc[20])):
            raise ValueError('record copy is not in unused B tail')
        tail_record = bytearray(alloc[23])
        if lookup_arm in ('tailffff_witness','tailfffe_witness'):
            address = alias_witness.CODE_VA
            struct.pack_into('<H', tail_record, 0x40, (address & 0xffff) | 7)
            struct.pack_into('<H', tail_record, 0x46, (address >> 16) & 0xffff)
            struct.pack_into('<H', tail_record, 0x48, (address >> 32) & 0xffff)
            if bytes(tail_record[0x40:0x4a]) != bytes.fromhex('07f0580e000000000001'):
                raise ValueError('alternate LoadShader address changed')
        alloc[20][tail_offset:tail_offset + len(tail_record)] = tail_record
    elif lookup_arm in ('alias0048','alias0048_zero_shader','alias0048_witness'):
        record_coordinate = 0x0048
        target_va = rows[0][5] + (record_coordinate << 14)
        alias_offset = target_va - rows[20][5]
        if alias_offset != 0x80000 or alias_offset + len(alloc[23]) > 0x80 + len(b):
            raise ValueError('low alias does not overlap the predicted B stream')
        low_record = bytearray(alloc[23])
        if lookup_arm == 'alias0048_zero_shader':
            low_record[0x40:0x4a] = bytes(10)
        elif lookup_arm == 'alias0048_witness':
            address = alias_witness.CODE_VA
            struct.pack_into('<H', low_record, 0x40, (address & 0xffff) | 7)
            struct.pack_into('<H', low_record, 0x46, (address >> 16) & 0xffff)
            struct.pack_into('<H', low_record, 0x48, (address >> 32) & 0xffff)
            if bytes(low_record[0x40:0x4a]) != bytes.fromhex('07f0580e000000000001'):
                raise ValueError('alternate LoadShader address changed')
        alloc[20][alias_offset:alias_offset + len(low_record)] = low_record
    struct.pack_into('<H', raw, 9, record_coordinate)
    alloc[25] = bytes(raw)
    alloc[20] = bytes(alloc[20])
    effective_b_sha = sha(memoryview(alloc[20])[0x80:0x80+len(b)])
    payload = b''.join(alloc[i] for i in parent.INDICES)
    if (len(payload) != 36880384 + shift or
            metadata_coordinate != 0x002c or
            (variant['page'] and sha(page) != variant['page']) or
            (variant['request'] and sha(request) != variant['request']) or
            (lookup_arm == 'alias0048' and variant['alias_b'] and
             effective_b_sha != variant['alias_b']) or
            (lookup_arm == 'alias0048_zero_shader' and variant['alias_zero_shader_b'] and
             effective_b_sha != variant['alias_zero_shader_b']) or
            (lookup_arm == 'alias0048_witness' and variant['alias_witness_b'] and
             effective_b_sha != variant['alias_witness_b']) or
            (expected_payload and sha(payload) != expected_payload)):
        raise ValueError('long-loop graph geometry/hash changed')
    return request, page, payload, alloc, (metadata_coordinate,
                                           record_coordinate, c_binding)


def build(request_path, page_path, payload_path, program, a, b, c, trips=TRIPS,
          marker_body=BODIES - 1, lookup_arm='original'):
    request, page, payload, alloc, fields = graph(program, a, b, c, trips,
                                                   marker_body, lookup_arm)
    peer_payload_sha = None
    coordinate_delta_offsets = None
    if lookup_arm in ('taildual_fffd_tensor','taildual_ffff_witness'):
        peer = 0xffff if lookup_arm == 'taildual_fffd_tensor' else 0xfffd
        old = struct.pack('<H', fields[1])
        new = struct.pack('<H', peer)
        coordinate_delta_offsets = [9+i for i,(x,y) in enumerate(zip(old,new)) if x != y]
        if coordinate_delta_offsets != [9]:
            raise ValueError('dual record selector is not a one-byte intervention')
        offset = sum(len(alloc[i]) for i in parent.INDICES[:parent.INDICES.index(25)]) + 9
        digest = hashlib.sha256()
        digest.update(memoryview(payload)[:offset])
        digest.update(new)
        digest.update(memoryview(payload)[offset+2:])
        peer_payload_sha = digest.hexdigest()
        expected_peer = (VARIANTS[trips]['taildual_ffff_witness_payload']
                         if lookup_arm == 'taildual_fffd_tensor' else
                         VARIANTS[trips]['taildual_fffd_tensor_payload'])
        if peer_payload_sha != expected_peer:
            raise ValueError('dual record peer payload changed')
    for path, data in ((request_path, request), (page_path, page),
                       (payload_path, payload)):
        Path(path).write_bytes(data)
    return {
        'scope': f'{VARIANTS[trips]["b_size"] >> 20} MiB physical B graph with {trips * BODIES} distinct B addresses',
        'capture_read_at_build': False,
        'program': 'sg4h12loop8160runtime',
        'runtime_trips': trips,
        'marker_body_in_final_trip': marker_body,
        'alloc25_lookup_arm': lookup_arm,
        'logical_b_bytes': len(b), 'b_physical_bytes': VARIANTS[trips]['b_size'],
        'allocation23_metadata_coordinate': hex(fields[0]),
        'allocation25_record_coordinate': hex(fields[1]),
        'allocation25_b_tail_record_offset': '0x3ff54000' if lookup_arm == 'taildual_fffd_tensor' else '0x3ff5c000' if lookup_arm in ('taildual_ffff_witness','tailffff_witness') else '0x3ff58000' if lookup_arm == 'tailfffe_witness' else '0x3ff20000' if lookup_arm == 'b_tail' else None,
        'allocation25_dual_tail_offsets': {'tensor': '0x3ff54000', 'witness': '0x3ff5c000'} if lookup_arm in ('taildual_fffd_tensor','taildual_ffff_witness') else None,
        'dual_coordinate_delta_offsets': coordinate_delta_offsets,
        'dual_peer_payload_sha256': peer_payload_sha,
        'allocation25_low_alias_record_offset': '0x80000' if lookup_arm in ('alias0048','alias0048_zero_shader','alias0048_witness') else None,
        'alternate_witness_code_sha256': alias_witness.CODE_SHA256 if lookup_arm in ('alias0048_witness','tailffff_witness','tailfffe_witness','taildual_fffd_tensor','taildual_ffff_witness') else None,
        'c_binding': hex(fields[2]),
        'requests_sha256': sha(request), 'pages_sha256': sha(page),
        'payload_sha256': sha(payload), 'payload_bytes': len(payload),
        'effective_logical_b_sha256': sha(memoryview(alloc[20])[0x80:0x80+len(b)]),
        'physical_allocations': [
            {'index': i, 'size': len(alloc[i]), 'sha256': sha(alloc[i])}
            for i in parent.INDICES],
    }
