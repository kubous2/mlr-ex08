# mlr-ex08 – Leaf Classifier

Live leaf classification in the browser from the phone's rear camera, using `leaf_model.onnx` with onnxruntime-web (works in Safari on iPhone). A photo upload is available as a fallback.

**Live page:** https://kubous2.github.io/mlr-ex08/

Classes: apple, cherry, chestnut, maple. Preprocessing follows `model_info.json`: crop the central square, resize to 96×96, scale RGB to 0–1, standardize with mean/std, NCHW input 1×3×96×96, softmax over the logits.

The "Run self-test" button on the page classifies `check_image.jpg` and compares the result with `check_expected.json`.
