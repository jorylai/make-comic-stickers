#!/usr/bin/env python3
"""Build millimeter-accurate bitmap sticker sheets with separate vector cutlines."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from trace_cutlines import trace
from extract_grid import extract


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_image(im, path, **options):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        im.save(tmp, **options)
        with Image.open(tmp) as check:
            check.load()
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def silhouette(mask, offset_px):
    work = mask.copy()
    for _ in range(100):
        expanded = ndimage.binary_fill_holes(
            ndimage.distance_transform_edt(~work) <= offset_px)
        labels, count = ndimage.label(expanded)
        if count == 1:
            break
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        largest = int(sizes.argmax())
        dist, near = ndimage.distance_transform_edt(labels != largest, return_indices=True)
        yy, xx = np.where((labels != 0) & (labels != largest))
        if not len(xx):
            raise ValueError('Empty sticker mask')
        index = dist[yy, xx].argmin()
        y0, x0 = int(yy[index]), int(xx[index])
        y1, x1 = near[:, y0, x0]
        steps = max(abs(y1-y0), abs(x1-x0)) + 1
        work[np.rint(np.linspace(y0,y1,steps)).astype(int),
             np.rint(np.linspace(x0,x1,steps)).astype(int)] = True
    else:
        raise ValueError('Too many separate components; regroup artwork and text')
    if expanded[0].any() or expanded[-1].any() or expanded[:,0].any() or expanded[:,-1].any():
        raise ValueError('Contour exceeds local padding')
    path, loops = trace(expanded)
    if loops != 1:
        raise ValueError(f'Expected one closed contour, found {loops}')
    yy, xx = np.where(expanded)
    lo = np.array([xx.min(), yy.min()], dtype=float)
    hi = np.array([xx.max()+1, yy.max()+1], dtype=float)
    edge = expanded & ~ndimage.binary_erosion(expanded, iterations=2)
    return path, edge, lo, hi


def load_asset(path):
    with Image.open(path) as im:
        tile = im.convert('RGBA')
    alpha = tile.getchannel('A')
    if alpha.getextrema()[0] == 255:
        raise ValueError(f'Asset must have transparency: {path}')
    box = alpha.getbbox()
    if not box:
        raise ValueError(f'Empty asset: {path}')
    return tile.crop(box)


def build(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    cfg = json.loads(config_path.read_text())
    width, height = map(float, cfg.get('sheet_mm', [140,210]))
    dpi = int(cfg.get('dpi',600))
    rows, cols = int(cfg.get('rows',4)), int(cfg.get('columns',3))
    offset = float(cfg.get('offset_mm',2.08))
    min_gap = float(cfg.get('min_gap_mm',4))
    margin = float(cfg.get('side_margin_mm',8))
    if min(width,height,dpi,rows,cols,offset) <= 0 or min_gap < 0 or margin < 0:
        raise ValueError('Invalid dimensions or production parameters')
    ppm = dpi/25.4
    pixels = (round(width*ppm),round(height*ppm))
    cw, ch = (width-2*margin)/cols, (height-2*margin)/rows
    # Reserve extra separation for tracing tolerance and raster rounding.
    fit = [cw-min_gap-2*offset-1, ch-min_gap-2*offset-1]
    if 'max_art_mm' in cfg:
        fit = [min(a,float(b)) for a,b in zip(fit,cfg['max_art_mm'])]
    if min(fit) <= 0:
        raise ValueError('Sheet too small for requested grid, gap and border')
    output.mkdir(parents=True,exist_ok=True)
    report, deliverables, previews, seen = [], [], [], set()
    resolve = lambda p: (config_path.parent / p).resolve()
    for theme in cfg['themes']:
        name = theme['id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',name) or name in seen:
            raise ValueError('Theme id must be unique and filesystem-safe')
        seen.add(name)
        folder = output/name
        folder.mkdir(exist_ok=True)
        if bool(theme.get('assets')) == bool(theme.get('source_grid')):
            raise ValueError('Choose exactly one of assets or source_grid')
        if theme.get('assets'):
            assets = [load_asset(resolve(p)) for p in theme['assets']]
        else:
            assets = extract(resolve(theme['source_grid']),
                             theme.get('heading_shift_px',0),rows,cols)
        if not assets or len(assets)>rows*cols:
            raise ValueError('Invalid sticker count for the grid')
        captions = theme.get('captions',['']*len(assets))
        if len(captions)!=len(assets):
            raise ValueError('Caption count differs from asset count')
        page = Image.new('RGB',pixels,'white')
        proof = page.copy()
        paths, boxes, sources = [], [], []
        for i,asset in enumerate(assets):
            box = asset.getchannel('A').getbbox()
            asset = asset.crop(box)
            save_image(asset,folder/'_extracted'/f'{i+1:02d}.png')
            aw, ah = asset.size
            scale = min(fit[0]*ppm/aw,fit[1]*ppm/ah)
            nw, nh = max(1,round(aw*scale)), max(1,round(ah*scale))
            art = asset.resize((nw,nh),Image.Resampling.LANCZOS)
            pad = math.ceil((offset+2)*ppm)
            local = Image.new('RGBA',(nw+2*pad,nh+2*pad))
            local.alpha_composite(art,(pad,pad))
            d, edge, lo, hi = silhouette(np.asarray(local)[:,:,3]>=96,offset*ppm)
            cx, cy = margin+(i%cols+.5)*cw, margin+(i//cols+.5)*ch
            center = (lo+hi)/2
            px,py = round(cx*ppm-center[0]),round(cy*ppm-center[1])
            page.paste(local,(px,py),local)
            proof.paste(local,(px,py),local)
            proof.paste((255,0,0),(px,py),Image.fromarray((edge*255).astype('uint8')))
            paths.append(f'<path id="cut-{i+1:02d}" d="{d}" transform="translate({px/ppm:.9f} {py/ppm:.9f}) scale({1/ppm:.12f})" fill="none" stroke="#ff0000" stroke-width="{.1*ppm:.9f}" stroke-linejoin="round"/>')
            origin = np.array([px,py])
            boxes.append(((lo+origin)/ppm,(hi+origin)/ppm))
            sources.append({'source_pixels':[aw,ah],'scale_to_output':scale})
        gap = None
        for j,(lo,hi) in enumerate(boxes):
            for lo2,hi2 in boxes[j+1:]:
                distance = float(np.linalg.norm(np.maximum(0,np.maximum(lo2-hi,lo-hi2))))
                gap = distance if gap is None else min(gap,distance)
        side = min(min(lo[0],width-hi[0]) for lo,hi in boxes)
        if gap is not None and gap < min_gap:
            raise ValueError(f'{name}: cutline gap {gap:.3f} < {min_gap} mm')
        if side < margin:
            raise ValueError(f'{name}: side margin {side:.3f} < {margin} mm')
        if any(min(lo)<0 or hi[0]>width or hi[1]>height for lo,hi in boxes):
            raise ValueError(f'{name}: cutline outside sheet')
        print_path = folder/(name+'_PRINT.png')
        save_image(page,print_path,dpi=(dpi,dpi),compress_level=4)
        ratio = 1680/pixels[0]
        phone_size = (1680,round(pixels[1]*ratio))
        for suffix,im in [('phone_CLEAN',page),('phone_RED',proof)]:
            path = folder/(name+'_'+suffix+'.jpg')
            save_image(im.resize(phone_size,Image.Resampling.LANCZOS),path,quality=95,subsampling=0)
            deliverables.append(path)
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}"><title>{escape(theme.get("title",name))} — NONPRINTING CUTLINES</title><g id="RED_CUTLINES_NONPRINTING">'+''.join(paths)+'</g></svg>'
        svg_path=folder/(name+'_CUTLINES_ONLY.svg')
        atomic_bytes(svg_path,svg.encode())
        deliverables.extend([print_path,svg_path])
        report.append({'theme':name,'title':theme.get('title',name),'stickers':len(assets),
                       'sheet_mm':[width,height],'print_pixels':list(pixels),'dpi':dpi,
                       'offset_mm':offset,'required_gap_mm':min_gap,'required_side_margin_mm':margin,
                       'min_cutline_gap_mm':gap,'side_margin_mm':side,'closed_cutlines':len(paths),
                       'bounds_mm':[[lo.tolist(),hi.tolist()] for lo,hi in boxes],
                       'sources':sources,'captions':captions})
        previews.append((name,proof.resize((420,round(420*height/width)),Image.Resampling.LANCZOS)))
        print(f'Built {name}: {len(assets)} stickers; gap={gap}; side={side:.3f}mm',flush=True)
    if not report:
        raise ValueError('No themes')
    for filename,data in [
        ('GEOMETRY_AND_CAPTIONS.json',json.dumps(report,ensure_ascii=False,indent=2)),
        ('CAPTIONS.txt','\n\n'.join(e['title']+'\n'+'\n'.join(e['captions']) for e in report)),
        ('PRINT_NOTES.txt','无红线 PRINT.png 用于印刷。phone_RED.jpg 仅校对。CUTLINES_ONLY.svg 为不参与印刷的红色刀模，需由印厂映射刀模通道。\n插画和文字为位图，刀模为闭合矢量。输出 dpi 为画布密度，实际细节受源素材像素限制。RGB 输出，未应用 CMYK 印厂配置。\n')]:
        path=output/filename
        atomic_bytes(path,data.encode())
        deliverables.append(path)
    columns=min(3,len(previews)); th=previews[0][1].height+40
    board=Image.new('RGB',(columns*460,math.ceil(len(previews)/columns)*th),'#f8f4ef')
    draw=ImageDraw.Draw(board)
    for i,(name,im) in enumerate(previews):
        x,y=i%columns*460,i//columns*th
        draw.text((x+20,y+8),name,fill='#49352e')
        board.paste(im,(x+20,y+30))
    overview=output/'ALL_THEMES_PHONE.jpg'
    save_image(board,overview,quality=94)
    deliverables.append(overview)
    from validate_pack import validate
    validate(output)
    fd,tmp=tempfile.mkstemp(suffix='.zip',dir=output)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in deliverables:
                archive.write(path,path.relative_to(output))
        with zipfile.ZipFile(tmp) as archive:
            if archive.testzip():raise ValueError('Corrupt ZIP')
        os.replace(tmp,output/'STICKER_PRINT_PACK.zip')
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    build(args.config,args.output)
