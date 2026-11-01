#!/usr/bin/env python3
"""Validate generated sticker geometry, PNG density, SVG loops and previews."""
import argparse
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile
from PIL import Image


def require(ok,message):
    if not ok:raise ValueError(message)


def validate(output):
    output=Path(output)
    report=json.loads((output/'GEOMETRY_AND_CAPTIONS.json').read_text())
    require(bool(report),'No themes in report')
    total=0
    for e in report:
        name=e['theme']; folder=output/name
        with Image.open(folder/(name+'_PRINT.png')) as im:
            im.load()
            require(im.mode=='RGB',f'{name}: expected RGB print image')
            require(list(im.size)==e['print_pixels'],f'{name}: print dimensions differ')
            require(all(abs(d-e['dpi'])<.1 for d in im.info.get('dpi',(0,0))),f'{name}: incorrect dpi metadata')
            require(all(abs(px-mm*e['dpi']/25.4)<=.5 for px,mm in zip(im.size,e['sheet_mm'])),f'{name}: physical size differs')
        root=ET.parse(folder/(name+'_CUTLINES_ONLY.svg')).getroot()
        paths=root.findall('.//{http://www.w3.org/2000/svg}path')
        require(len(paths)==e['stickers']==e['closed_cutlines'],f'{name}: cut count mismatch')
        require(root.attrib['width']==f"{float(e['sheet_mm'][0])}mm",f'{name}: SVG width')
        require(root.attrib['height']==f"{float(e['sheet_mm'][1])}mm",f'{name}: SVG height')
        for p in paths:
            d=p.attrib.get('d','')
            require(len(re.findall('[Mm]',d))==1 and len(re.findall('[Zz]',d))==1 and d.rstrip()[-1] in 'zZ',f'{name}: contour must be one closed loop')
            require(p.attrib.get('stroke')=='#ff0000' and p.attrib.get('fill')=='none',f'{name}: red unfilled contour required')
        boxes=e['bounds_mm']; actual_gap=None
        for j,(lo,hi) in enumerate(boxes):
            require(min(lo)>=0 and hi[0]<=e['sheet_mm'][0] and hi[1]<=e['sheet_mm'][1],f'{name}: outside page')
            for lo2,hi2 in boxes[j+1:]:
                distance=math.hypot(max(0,lo2[0]-hi[0],lo[0]-hi2[0]),max(0,lo2[1]-hi[1],lo[1]-hi2[1]))
                actual_gap=distance if actual_gap is None else min(actual_gap,distance)
        side=min(min(lo[0],e['sheet_mm'][0]-hi[0]) for lo,hi in boxes)
        require(actual_gap is None or actual_gap>=e['required_gap_mm'],f'{name}: gap below requested minimum')
        require(side>=e['required_side_margin_mm'],f'{name}: side margin below minimum')
        require(len(e['captions'])==e['stickers'],f'{name}: captions mismatch')
        for suffix in ['phone_RED','phone_CLEAN']:
            with Image.open(folder/(name+'_'+suffix+'.jpg')) as im:im.load()
        total+=e['stickers']
    archive=output/'STICKER_PRINT_PACK.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as z:require(z.testzip() is None,'Corrupt ZIP')
    print(f'Validated {len(report)} themes, {total} stickers. Visual review still required.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    validate(parser.parse_args().output)
