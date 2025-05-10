# Step to reproduce the results using QWen 2.5VL
1. Checkout to the branch "leihan_chen/qwen2.5vl" to get the code for the evaluation.
2. Please check the [QWen 2.5VL](https://github.com/QwenLM/Qwen2.5-VL) GitHub repository for the model and environment setup.
3. Open the evaluation folder.
```bash
cd reason_evaluation
```
4. Run the following python script.
```python
python3 qwen2.5vl_3d_test.py 
```
5. Convert generated results to the format used for evaluation script.
```python
python3 convert_json_files.py 
```
6. Run the evaluation script.
```bash
cd ..
bash scripts/eval/sqa3d_distributed.sh
```

# NOTE:
- The generated files from QWen can be problematic for JSON output format.
- Convert script cannot always conver all json entries due to the inconsistent format of the generated files by QWen.
