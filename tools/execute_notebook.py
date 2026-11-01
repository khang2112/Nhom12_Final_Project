"""Execute read-only notebook cells in-process and save genuine rich outputs."""
from pathlib import Path
import os
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

root=Path(__file__).resolve().parents[1]
os.chdir(root)
path=root/'notebooks/Project_Stage1_Stage2_NHOM12.ipynb'
nb=nbformat.read(path,as_version=4)
shell=InteractiveShell.instance()
count=0
for cell in nb.cells:
    if cell.cell_type != 'code':continue
    count+=1
    with capture_output() as output:
        result=shell.run_cell(cell.source,store_history=True)
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError(f'Notebook cell {count} failed: {result.error_in_exec or result.error_before_exec}')
    cell.execution_count=count
    cell.outputs=[]
    if output.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=output.stdout))
    if output.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=output.stderr))
    for item in output.outputs:
        cell.outputs.append(nbformat.v4.new_output('display_data',data=item.data,metadata=item.metadata))
nbformat.write(nb,path)
print(f'NOTEBOOK PASS: executed {count} cells in-process, 0 errors')
