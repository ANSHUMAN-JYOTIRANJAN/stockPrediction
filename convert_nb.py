import json

with open("Multi_Stock_ML_Prediction.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

code_cells = []
for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        if isinstance(source, list):
            code_cells.append("".join(source))
        else:
            code_cells.append(source)

with open("Multi_Stock_ML_Prediction.py", "w", encoding="utf-8") as f:
    f.write("\n\n".join(code_cells))
