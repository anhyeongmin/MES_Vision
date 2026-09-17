"""Finish the evaluation automatically after a specified local training process exits."""
import argparse,ctypes,json,subprocess,sys,time,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1]
status=a.run/'pipeline.json'
def save(state,**kw):status.write_text(json.dumps(dict(status=state,**kw),indent=2),encoding='utf-8')
try:
 save('WAITING_FOR_TRAINING',pid=a.pid)
 kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p;kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.CloseHandle.argtypes=[ctypes.c_void_p]
 handle=kernel.OpenProcess(0x00100000,False,a.pid)
 if handle:
  try:
   while kernel.WaitForSingleObject(handle,1000)==258:pass
  finally:kernel.CloseHandle(handle)
 m=json.loads((a.run/'run.json').read_text(encoding='utf-8'));assert m['status']=='COMPLETED_UNVALIDATED',m['status']
 save('COMPARING')
 with (a.run/'comparison.log').open('w',encoding='utf-8') as log:
  subprocess.run([sys.executable,'-X','utf8',str(root/'scripts/compare_vlm_lora.py'),'--run',str(a.run)],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
 save('COMPLETED',report=str(a.run/'comparison/report.html'),production_ready=False)
except BaseException as e:save('FAILED',error=repr(e));traceback.print_exc();raise
