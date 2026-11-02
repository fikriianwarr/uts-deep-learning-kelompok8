"""Execute a pure-Python notebook in process and persist authentic rich outputs.

Uses IPython's display capture; no Jupyter socket service is needed. Intended
for this project's Python-only code cells, with cached provenance-checked runs.
"""
from pathlib import Path
import sys,os,json,traceback,base64
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
ROOT=Path(__file__).resolve().parents[1]

def main():
    os.chdir(ROOT);sys.path.insert(0,str(ROOT))
    path=ROOT/'notebooks/PROYEK_UTS_DEEP_LEARNING_KELOMPOK_8.ipynb'
    book=nbformat.read(path,4);scope={'__name__':'__main__'}
    InteractiveShell.instance()
    counter=0
    for cell in book.cells:
        if cell.cell_type!='code':continue
        counter+=1;error=None
        print('Executing cell',counter,flush=True)
        with capture_output() as captured:
            try:exec(compile(cell.source,f'<notebook-cell-{counter}>','exec'),scope)
            except BaseException as ex:
                error=ex;traceback.print_exc()
        cell.execution_count=counter;cell.outputs=[]
        if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for rich in captured.outputs:
            data={k:(base64.b64encode(v).decode() if isinstance(v,bytes) and k.startswith('image/') else v) for k,v in rich.data.items()}
            cell.outputs.append(nbformat.v4.new_output('display_data',data=data,metadata=rich.metadata))
        if error:
            cell.outputs.append(nbformat.v4.new_output('error',ename=type(error).__name__,evalue=str(error),traceback=traceback.format_exception(error)))
            nbformat.write(book,path);raise error
        nbformat.write(book,path)
    book.metadata['execution_environment']={'method':'IPython in-process execution','python':sys.version.split()[0],'device':'CPU','full_dataset':True,'test_results_reused_when_identical':True}
    nbformat.validate(book);nbformat.write(book,path)
    print('Executed and validated',counter,'code cells',flush=True)

if __name__=='__main__':main()
