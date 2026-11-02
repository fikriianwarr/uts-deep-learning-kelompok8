"""Exercise the real Gradio API and compare outputs with saved test predictions."""
from pathlib import Path
import os,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)

def main():
    import torch
    torch.set_num_threads(4)
    from app import demo,predictor,classify
    from gradio_client import Client,handle_file
    if predictor is None:raise RuntimeError('Final model or matching threshold could not be loaded')
    examples=json.loads((ROOT/'artifacts/eksperimen/demo_examples.json').read_text())
    if len(examples)<2:raise RuntimeError('Missing real test examples')
    outputs=[]
    try:
        _,url,_=demo.launch(server_name='127.0.0.1',server_port=7861,prevent_thread_lock=True,quiet=True)
        c=Client(url,verbose=False)
        for row in examples:
            labels,heatmap,status=c.predict(handle_file(str(ROOT/row['image'])),api_name='/classify')
            if labels['label']!=row['predicted_label']:raise AssertionError('API and offline label differ')
            if row['accepted']!=('DITERIMA' in status):raise AssertionError('API threshold decision differs')
            if not heatmap:raise AssertionError('Grad-CAM output is missing')
            outputs.append({'kind':row['kind'],'true_label':row['true_label'],'predicted_label':labels['label'],
                            'status':status,'gradcam_produced':True})
        assert 'Unggah citra' in classify(None)[2]
    finally:demo.close()
    result={'status':'PASSED','api_cases':len(outputs),'outputs':outputs,
            'checkpoint_sha256':hashlib.sha256((ROOT/'artifacts/eksperimen/efficientnet/best.pt').read_bytes()).hexdigest(),
            'public_hosting_verified':False,'browser_visual_test_verified':False}
    target=ROOT/'artifacts/eksperimen/app_integration.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
