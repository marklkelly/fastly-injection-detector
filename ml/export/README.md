# ml/export — Model Export and Quantization

## Export to ONNX
```bash
python ml/export/export_onnx.py --model-path ml/models/bert-tiny-pi-v1 --output-dir service/assets/
```

## Quantize to INT8
```bash
python ml/export/quantize_int8.py service/assets/student_1x128_f32.onnx --output service/assets/student_1x128_int8.onnx
```
