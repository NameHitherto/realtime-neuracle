"""Read-only Unity client audit. Run in .tools-venv (dnfile + dncil)."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import dnfile
from dncil.cil.body.reader import read_method_body_from_bytes
from dncil.clr.token import Token


def assembly(path):
    pe = dnfile.dnPE(str(path))
    owners = {}
    for t in pe.net.mdtables.TypeDef:
        name = f'{t.TypeNamespace}.{t.TypeName}'
        for field in t.FieldList:
            owners[id(field.row)] = name
        for method in t.MethodList:
            owners[id(method.row)] = name

    def resolve(value):
        if not isinstance(value, Token):
            return str(value)
        table, index = value.value >> 24, value.value & 0xffffff
        if table == 0x70:
            return repr(pe.net.user_strings.get(index).value)
        rows = pe.net.mdtables.tables.get(table)
        if rows and 0 < index <= len(rows.rows):
            row = rows.rows[index - 1]
            name = str(getattr(row, 'Name', getattr(row, 'TypeName', value)))
            if hasattr(row, 'Class') and row.Class.row:
                parent = row.Class.row
                name = f'{getattr(parent, "TypeNamespace", "")}.{getattr(parent, "TypeName", "")}.{name}'
            return owners.get(id(row), '') + '::' + name
        return str(value)

    methods = {}
    for t in pe.net.mdtables.TypeDef:
        for ref in t.MethodList:
            m = ref.row
            if not m.Rva:
                continue
            key = f'{t.TypeNamespace}.{t.TypeName}::{m.Name}'
            try:
                methods[key] = '\n'.join(f'{i.offset:04x} {i.opcode.name} {resolve(i.operand)}' for i in read_method_body_from_bytes(pe.get_data(m.Rva)).instructions)
            except Exception as exc:
                methods[key] = f'UNREADABLE: {exc}'
    refs = [f'{r.TypeNamespace}.{r.TypeName}' for r in pe.net.mdtables.TypeRef]
    return methods, refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('base', type=Path)
    parser.add_argument('--out', type=Path, default=Path('outputs/final-debug/client-audit'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = ['虚拟任务竞速赛06251432', '虚拟任务竞速赛_0911_禁键盘_比赛版本']
    inventories, methods = [], []
    for label, name in zip(['old', 'final'], names):
        root = args.base / name
        files = {str(p.relative_to(root)): {'size': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in root.rglob('*') if p.is_file()}
        inventories.append(files)
        result, refs = assembly(root / '虚拟任务竞速赛_Data/Managed/Assembly-CSharp.dll')
        methods.append(result)
        (args.out / f'{label}-inventory.json').write_text(json.dumps(files, ensure_ascii=False, indent=2), encoding='utf-8')
        (args.out / f'{label}-methods.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        (args.out / f'{label}-type-refs.json').write_text(json.dumps(refs, indent=2), encoding='utf-8')
        selected = {k:v for k,v in result.items() if any(t in k for t in ['EEGInlet', 'KartLSLOutlet', '.KartInput::', '.SplineFollowController::'])}
        (args.out / f'{label}-control.il.txt').write_text('\n\n'.join(k+'\n'+v for k,v in selected.items()), encoding='utf-8')
    a,b = inventories
    ma,mb = methods
    diff = {'changed_files':[k for k in a.keys() & b.keys() if a[k]['sha256'] != b[k]['sha256']], 'added_files': sorted(b.keys()-a.keys()), 'removed_files':sorted(a.keys()-b.keys()), 'unchanged_files':sum(a[k]['sha256']==b[k]['sha256'] for k in a.keys() & b.keys()), 'changed_methods': sorted(k for k in ma.keys() & mb.keys() if ma[k]!=mb[k]), 'added_methods':sorted(mb.keys()-ma.keys()), 'removed_methods': sorted(ma.keys()-mb.keys())}
    (args.out/'diff.json').write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding='utf-8')
    with ZipFile(args.base/'附件6.docx') as z:
        doc=ET.fromstring(z.read('word/document.xml'))
        ns='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        text='\n'.join(''.join(t.text or '' for t in p.iter(ns+'t')) for p in doc.iter(ns+'p'))
        (args.out/'attachment6.txt').write_text(text, encoding='utf-8')
    print(json.dumps(diff, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
