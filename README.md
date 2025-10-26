# Install and activate virtual environment
- Init virt env:
```bash
python3 -m venv venv
virtualenv -p /Library/Frameworks/Python.framework/Versions/3.9/bin/python3.9 venv
```
- Activate virtual environment:
```bash
source venv/bin/activate
```
- Install `requirements.txt` for the required libraries:
```bash
pip install -r requirements.txt
```
- To delete any unnecessary libraries:
```bash
pip uninstall <package-name>
```
- To deactivate virtual environment:
```bash
deactivate
rm -rf venv
```