#!/usr/bin/env python3
"""Reproduce the full Metal-free, no-Submit command/record staging control."""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import g17gpulock
import g17stagepayload
import g17commandpages
import g17viewrequest
import g17allviewrequests
import g17special18

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'spike/agxsub/g17pure3_preflight.c'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recovery():
    data=subprocess.check_output(['ioreg','-l','-r','-c','AGXAcceleratorG17X'],text=True)
    values=[int(x) for x in re.findall(r'"recoveryCount"\s*=\s*(\d+)',data)]
    if not values:raise RuntimeError('recoveryCount unavailable')
    return values


def events():
    return set(glob.glob('/Library/Logs/DiagnosticReports/gpuEvent-*.ips'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--notification',action='store_true',
                    help='also create and bind an IOGPU notification queue, without Submit')
    ap.add_argument('--zero-submit',action='store_true',
                    help='call Submit with count=0, a live record, and no GPU command')
    ap.add_argument('--view',action='store_true',
                    help='also register selector-9 output-buffer view #5, without GPU work')
    ap.add_argument('--view-nine',action='store_true',
                    help='register selector-9 view #9 with its pointer field zeroed')
    ap.add_argument('--all-views',action='store_true',
                    help='replay all 17 captured nonzero-aperture views with pointers zeroed')
    ap.add_argument('--rebase-parent',action='store_true',
                    help='for all-views, rebase view #21 parent ordinal from 21 to 4')
    ap.add_argument('--special-18',action='store_true',
                    help='also replay captured selector-9 request #18 with no GPU aperture')
    ap.add_argument('--base-class',action='store_true',
                    help='try IOGPU-only IOGPUMetalBuffer class in backing requests 2/20')
    args=ap.parse_args()
    if args.base_class and not args.view:ap.error('--base-class requires --view')
    if args.rebase_parent and not args.all_views:ap.error('--rebase-parent requires --all-views')
    if args.output.exists():raise ValueError('refusing to overwrite report')
    with tempfile.TemporaryDirectory(prefix='g17',dir='/tmp') as directory:
        temp=Path(directory);binary=temp/'p';payload=temp/'physical.bin';pages=temp/'pages.bin'
        pmanifest=g17stagepayload.build(payload)
        cmanifest=g17commandpages.build(pages)
        vmanifest=g17viewrequest.build(temp/'view5.bin') if args.view else None
        v9manifest=g17viewrequest.build(temp/'view9.bin',9) if args.view_nine else None
        all_manifest=g17allviewrequests.build(temp/'allviews.bin') if args.all_views else None
        s18manifest=g17special18.build(temp/'special18.bin') if args.special_18 else None
        subprocess.run(['xcrun','clang','-Wall','-Wextra','-Werror','-Wno-unused-function',
                        '-fblocks','-DG17_BLOCK_PREFLIGHT',str(SOURCE),'-framework','IOKit',
                        '-framework','CoreFoundation','-o',str(binary)],check=True)
        env=os.environ.copy()
        env.update(LAYOUT_ALLOC_PREFLIGHT='1',STAGE_CAPTURED_PAYLOAD=str(payload),
                   LAYOUT_QUEUE_PREFLIGHT='1',TRACE_ID_PREFLIGHT='1',
                   STAGE_COMMAND_PAGES=str(pages),RECORD_PREFLIGHT='1')
        if args.view:env['STAGE_VIEW5_TEMPLATE']=str(temp/'view5.bin')
        if args.view_nine:env['STAGE_VIEW9_TEMPLATE']=str(temp/'view9.bin')
        if args.all_views:env['STAGE_ALL_VIEWS']=str(temp/'allviews.bin')
        if args.rebase_parent:env['ALL_VIEWS_REBASE_PARENT']='1'
        if args.special_18:env['STAGE_SPECIAL18']=str(temp/'special18.bin')
        if args.base_class:env['VIEW5_BASE_CLASS']='IOGPUMetalBuffer'
        if args.notification or args.zero_submit:env['NOTIFICATION_PREFLIGHT']='1'
        if args.zero_submit:env['ZERO_SUBMIT_PREFLIGHT']='1'
        with g17gpulock.acquire('exclusive',timeout=60):
            before=recovery();prior=events()
            run=subprocess.run([str(binary)],env=env,text=True,capture_output=True,timeout=30)
            after=recovery();new_events=sorted(events()-prior)
        required=('STAGE PASS:','TRACE ID PASS:','COMMAND PAGES PASS:',
                  'RECORD PREFLIGHT PASS:','NOTIFICATION PREFLIGHT PASS:'
                  if (args.notification or args.zero_submit) else 'LAYOUT QUEUE PASS:')
        required+=('LAYOUT CONTROL PASS; zero-count Submit only, no GPU command.'
                   if args.zero_submit else 'LAYOUT CONTROL PASS; no Submit call.',)
        if args.zero_submit:required+=('ZERO SUBMIT returned status=0', 'sentinels=64/64 marks=0/0;')
        if args.view:required+=('VIEW5 CONTROL PASS:',)
        if args.view_nine:required+=('VIEW9 CONTROL PASS:',)
        if args.all_views:required+=('ALL VIEWS CONTROL PASS:',)
        if args.special_18:required+=('SPECIAL18 CONTROL PASS:',)
        if args.base_class:required+=('CLASS CONTROL: IOGPUMetalBuffer=',)
        physical_calls=re.findall(r'^LAYOUT ALLOC original#(\d+) kr=0x([0-9a-f]+)',run.stderr,re.M)
        view_calls=re.findall(r'^ALL VIEWS index=(\d+) kr=0x([0-9a-f]+) bytes=\d+ aperture=0x([0-9a-f]+) expected=0x([0-9a-f]+)',
                              run.stderr,re.M)
        special_calls=re.findall(r'^SPECIAL18 sel-9 kr=0x([0-9a-f]+)',run.stderr,re.M)
        expected_views=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21]
        call_receipt=(len(physical_calls)==11 and all(status=='0' for _,status in physical_calls)
                      and ([int(index) for index,*_ in view_calls]==expected_views
                           if args.all_views else not view_calls)
                      and all(status=='0' and aperture==expected
                              for _,status,aperture,expected in view_calls)
                      and (special_calls==['0'] if args.special_18 else not special_calls))
        passed=(run.returncode==0 and before==after and not new_events and
                call_receipt and all(marker in run.stderr for marker in required))
        report=dict(scope='pure-process CPU staging of authored physical and selector-14 pages, '
                          'fresh trace IDs and locally callable record blocks'
                          + (', notification queue bound' if (args.notification or args.zero_submit) else '')
                          + ('; no Metal, zero-count Submit only, no GPU command'
                             if args.zero_submit else '; no Metal and no Submit'),
                    source_sha256=digest(SOURCE),binary_sha256=digest(binary),
                    physical_payload_sha256=pmanifest['payload_sha256'],
                    command_template_sha256=cmanifest['template_sha256'],
                    view_template_sha256=vmanifest['template_sha256'] if vmanifest else None,
                    view9_template_sha256=v9manifest['template_sha256'] if v9manifest else None,
                    all_views_template_sha256=all_manifest['template_sha256'] if all_manifest else None,
                    parent_rebased=args.rebase_parent,
                    special18_template_sha256=s18manifest['template_sha256'] if s18manifest else None,
                    base_class='IOGPUMetalBuffer' if args.base_class else None,
                    selector9_receipt={'physical_calls':physical_calls,
                                       'view_calls':view_calls,
                                       'special_calls':special_calls,'passed':call_receipt},
                    returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,
                    recovery_before=before,recovery_after=after,new_events=new_events,passed=passed)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print('PASS' if passed else 'FAIL',args.output)
        return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
