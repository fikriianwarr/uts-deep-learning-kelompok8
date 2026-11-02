"""Gradio demonstration. Displays no fake results without a trained checkpoint."""
import os
from pathlib import Path
import gradio as gr
from src.infer import Predictor

ROOT=Path(__file__).resolve().parent
ck=ROOT/'artifacts/eksperimen/efficientnet/best.pt'
th=ROOT/'artifacts/eksperimen/efficientnet/threshold.json'
predictor=None
init_error=''
try:
    if ck.is_file():predictor=Predictor(ck,th)
    else:init_error='Model terlatih belum ditemukan. Jalankan notebook pelatihan final dan salin best.pt ke artifacts/eksperimen/efficientnet/.'
except Exception as ex:init_error=f'Gagal memuat checkpoint: {ex}'

def classify(im):
    if predictor is None:return {},None,init_error
    if im is None:return {},None,'Unggah citra sebelum melakukan prediksi.'
    data=predictor.predict(im)
    status=(f'Prediksi: {data["label"]} | confidence {data["confidence"]:.4%}; '
            f'ambang {data["threshold"]:.4%}. '
            + ('DITERIMA' if data['accepted'] else 'DITOLAK: kepercayaan model rendah'))
    return data['probabilities'],data['gradcam'],status

with gr.Blocks(title='Kelompok 8 | Klasifikasi Citra Satelit EuroSAT') as demo:
    gr.Markdown('# Klasifikasi Tutupan Lahan Sentinel-2\n**EfficientNet-B0 · Grad-CAM · Confidence Threshold**')
    gr.Markdown('Demo akademis menggunakan citra RGB dengan karakteristik serupa EuroSAT. Tidak untuk keputusan tata ruang operasional.')
    if predictor is None:gr.Markdown('**Status:** '+init_error)
    with gr.Row():
        input_image=gr.Image(label='Citra RGB',type='pil',sources=['upload'],height=360)
        with gr.Column():
            result=gr.Label(label='Probabilitas 10 kelas',num_top_classes=10)
            heatmap=gr.Image(label='Grad-CAM',type='pil',height=300)
            gr.Markdown('Persentase pada diagram dibulatkan. Skor lebih rinci dan ambang keputusan ditampilkan pada status prediksi. Pembesaran gambar tidak menambah resolusi citra asli 64×64 piksel.')
    import json
    sample_file=ROOT/'artifacts/eksperimen/demo_examples.json'
    if sample_file.is_file():
        samples=json.loads(sample_file.read_text())
        gr.Examples(examples=[[str(ROOT/x['image'])] for x in samples if (ROOT/x['image']).is_file()],inputs=input_image,label='Contoh asli dari data testing',cache_examples=False)
    status=gr.Textbox(label='Status prediksi',interactive=False)
    btn=gr.Button('Analisis citra',variant='primary')
    btn.click(classify,inputs=[input_image],outputs=[result,heatmap,status],concurrency_limit=1)
    gr.Markdown('**Catatan:** Confidence Threshold mengukur keyakinan model, bukan menjamin prediksi benar. Grad-CAM menyoroti area yang berkontribusi, bukan peta segmentasi.')
if __name__=='__main__':demo.launch(server_name='0.0.0.0',server_port=int(os.getenv('PORT','7860')))
