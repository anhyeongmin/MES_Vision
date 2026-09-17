"""Render diagnostic outputs without calling text similarity production accuracy."""
import argparse,json,html,statistics
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);args=p.parse_args();run=args.run
 read=lambda p:json.loads(p.read_text(encoding='utf8'))
 before=read(run/'before.json');after=read(run/'after.json');lookup={r['id']:r for r in before}
 h=html.escape;cards=[]
 for row in after:
  b=lookup[row['id']]
  cards.append(f'<article><img src="{Path(row["image"]).as_uri()}"><div><h2>{h(row["label"])}</h2><p>{h(row["question"])}</p><p>참고 설명: {h(row["target"]["observation"])}</p><p>이전: {h(b["raw"])}</p><p>추가 학습: {h(row["raw"])}</p><small>{row["seconds"]:.2f}초 · JSON {row["valid"]}</small></div></article>')
 summary=dict(count=len(after),valid_outputs=sum(r['valid'] for r in after),mean_seconds=statistics.mean(r['seconds'] for r in after),
  independent_evaluation=False,note='Same specimens, reused prior sessions and assistant draft labels; inspect individual observations. Timing includes unmerged LoRA.')
 (run/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
 (run/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>VLM 전체 종류 확장 비교</title><style>body{font:16px sans-serif;margin:32px;background:#eef1f5}article{display:flex;gap:24px;background:white;padding:20px;margin:18px 0}img{width:240px;height:240px;object-fit:contain}small{color:#555}</style><h1>VLM 전체 종류 확장 비교</h1><p>기존 사진으로 확인한 진단 결과입니다. 독립 성능 검증이 아닙니다.</p><pre>'+h(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards),encoding='utf8')
 print(run/'report.html')
if __name__=='__main__':main()
