"""Compare completed benchmark runs with matching inputs, weights and collection threshold."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from mes_vision.training.data import read_json,write_json,sha256,require
from mes_vision.validation.inference_comparison import compare_predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True); parser.add_argument('--candidate',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    require(not args.output.exists(),'Refusing to overwrite comparison')
    baseline=read_json(args.baseline); require(baseline['status']=='COMPLETED','Incomplete baseline')
    result={'baseline':str(args.baseline),'baseline_sha256':sha256(args.baseline),'comparisons':{},'product_accuracy_validated':False}
    for path in args.candidate:
        candidate=read_json(path); require(candidate['status']=='COMPLETED','Incomplete candidate')
        for key in ('sources','input_shapes','weights_sha256','threshold','gpu','torch','rfdetr'):
            require(candidate[key]==baseline[key],'Comparison mismatch: '+key)
        result['comparisons'][path.parent.name]={'report_sha256':sha256(path),'profile':candidate['profile'],
            'median_speedup':baseline['timing_ms']['median']/candidate['timing_ms']['median'],
            'p95_speedup':baseline['timing_ms']['p95']/candidate['timing_ms']['p95'],
            'outputs':[compare_predictions(baseline['predictions'],candidate['predictions'],threshold=t) for t in (.05,.4,.7)]}
    write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
