"""Persistent operator fault latch independent of the inspection SQLite database."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4
from mes_vision.training.data import require


def fault_code(message):
    text=str(message).lower()
    if any(word in text for word in ("out of memory","memoryerror","cuda error","cublas","메모리")): return "MODEL_MEMORY"
    if any(word in text for word in ("disk","sqlite","database","permission","space","저장","공간","원본")): return "STORAGE_FAILURE"
    return "ENGINE_FAILURE"


class FaultJournal:
    def __init__(self,root):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.path=self.root/"operator-faults.json"
        self.entries=[]; self.persistence_error=None; self.corrupt=False
        if self.path.exists():
            try:
                require(self.path.stat().st_size<=4*1024**2,"오류 기록 파일이 너무 큽니다.")
                value=json.loads(self.path.read_text(encoding="utf-8"))
                require(value["schema_version"]==1 and isinstance(value["entries"],list) and len(value["entries"])<=1000,"오류 기록 형식 오류")
                for entry in value["entries"]:
                    require(isinstance(entry,dict) and all(isinstance(entry[k],str) for k in ("id","code","detail")) and type(entry["opened"]) in {int,float}
                        and (entry["cleared"] is None or type(entry["cleared"]) in {int,float}),"오류 기록 항목 손상")
                self.entries=value["entries"]
            except Exception as exc:
                self.corrupt=True; self.persistence_error=str(exc)
                self.entries=[self.entry("FAULT_RECORD_DAMAGED","이전 오류 기록을 읽을 수 없습니다. 진단을 보존하고 복구 확인이 필요합니다.")]

    @staticmethod
    def entry(code,detail): return {"id":uuid4().hex,"code":code,"detail":str(detail)[:4000],"opened":time.time(),"cleared":None,"operator":None,"note":None}

    @property
    def active(self): return [e for e in self.entries if e["cleared"] is None]

    def recover_from_store(self,store):
        # Recover a fault if the independent file could not be written but SQLite survived.
        with store.connect() as db:
            rows=db.execute("""SELECT e.data FROM events e WHERE e.type='OPERATOR_FAULT'
                AND NOT EXISTS(SELECT 1 FROM events a,json_each(a.data,'$.ids') ids
                  WHERE a.type='OPERATOR_FAULT_ACK' AND ids.value=json_extract(e.data,'$.id')) ORDER BY e.id""").fetchall()
        changed=False
        for row in rows:
            entry=json.loads(row[0]); existing=next((e for e in self.entries if e['id']==entry['id']),None)
            if existing and existing['cleared'] is None: continue
            if existing: existing['cleared']=None
            else: self.entries.append(entry)
            changed=True
        if changed:
            try: self.save()
            except Exception as exc: self.persistence_error=str(exc)

    def save(self):
        require(not self.corrupt,"손상된 오류 기록은 확인 절차에서 보존한 뒤 복구하세요.")
        # Retain all unresolved faults; trim only old acknowledged history.
        active=self.active; cleared=[e for e in self.entries if e["cleared"] is not None]
        require(len(active)<=1000,"미해결 오류 기록 수 초과")
        self.entries=cleared[-(1000-len(active)):] + active if len(active)<1000 else active
        temporary=self.path.with_name(self.path.name+".partial-"+uuid4().hex)
        try:
            with temporary.open("x",encoding="utf-8") as stream:
                json.dump({"schema_version":1,"entries":self.entries},stream,ensure_ascii=False,allow_nan=False); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary,self.path); self.persistence_error=None
        except Exception as exc:
            self.persistence_error=str(exc); raise

    def raise_fault(self,code,detail):
        existing=next((e for e in self.active if e["code"]==code),None)
        if existing: return existing,False
        entry=self.entry(code,detail); self.entries.append(entry)
        try: self.save()
        except Exception as exc: self.persistence_error=str(exc)
        return entry,True

    def acknowledge(self,operator,note):
        require(operator.strip() and note.strip(),"확인자와 조치 내용을 입력하세요.")
        original=deepcopy(self.entries); was_corrupt=self.corrupt
        if self.corrupt:
            target=self.path.with_name("operator-faults-damaged-"+uuid4().hex+".json")
            shutil.copyfile(self.path,target)
            from mes_vision.training.data import sha256
            require(sha256(self.path)==sha256(target),"손상된 기록의 진단 사본을 보존하지 못했습니다.")
            self.corrupt=False
        for entry in self.active: entry.update(cleared=time.time(),operator=operator.strip(),note=note.strip())
        try: self.save()
        except Exception:
            self.entries=original; self.corrupt=was_corrupt; raise
