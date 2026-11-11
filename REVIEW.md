# REVIEW – mlr-ex08

Page: https://kubous2.github.io/mlr-ex08/
Tested on: Chrome, Samsung A54 5G

## 1. It runs on my phone
**What I did:** I opened the page on my phone Android Samsung A54 5G in Google Chrome,
both from the link and by scanning the QR code. I allowed camera access, pointed the
rear camera at real leaves and also tested the photo upload.

**What I found:** The page loaded without errors and the probabilities of all four
classes updated live. Maple and chestnut leaves were classified correctly. Apple leaves
were classified as cherry, the same mistake as in the confusion matrix in the notebook
(all 8 validation apples predicted as cherry), so a "cherry" result cannot be trusted.
Safari on iPhone was not tested.

**What was fixed:** Nothing, the page worked on the first try.

## 2. The same numbers as the notebook
**What I did:**I uploaded check_image.jpg on the page on my PC
and compared the probabilities with check_expected.json, which was saved by the notebook.

| class    | check_expected.json | page |
|----------|---------------------|------|
| apple    | 4.950804921577401e-08                   | 4.9508e-8    |
| cherry   | 0.9999963045120239                   | 1.0000e+0    |
| chestnut | 3.613846729422221e-06                   | 3.6139e-6    |
| maple    | 1.350733924709857e-07                   | 1.3507e-7    |

**What I found:** …
The page and the notebook agree to three decimals – the largest
difference is 1.03e-7 (self-test: PASS). So the page runs the same model with the same
preprocessing as the notebook. check_image.jpg is an apple from the validation set,
but both the notebook and the page predict cherry with almost 100 % – the same
apple → cherry confusion as in the confusion matrix.

**What was fixed:** Nothing.

## 3. The preprocessing matches the model
**What I did:** …
I read the preprocess function in index.html and compared
each step with model_info.json and the transform in the notebook.

| step | index.html (lines) | model_info.json / notebook |
|------|--------------------|----------------------------|
| crop central square | 124–125, 129: `s = min(w, h)`, `sx = (w−s)/2`, `sy = (h−s)/2` | the dataset photos are centre-cropped |
| resize to 96 × 96 | 113, 129: canvas size = `input_size`, `drawImage(…, 0, 0, size, size)` | `input_size: 96` |
| divide by 255 | 135–137: `px[…] / 255` | `ToTensor` (0–1) |
| (x − mean) / std per channel | 133, 135–137: `(… − mean[c]) / std[c]` | `mean`, `std` in model_info.json = `Normalize(mean, std)` |
| channel order | 134–138: RGBA from `getImageData` → R plane, G plane, B plane; alpha dropped | `ToTensor` gives CHW (3, H, W) |
| tensor shape | 139: `[1, 3, size, size]`, float32 | 1×3×96×96 |

**What I found:** …
**What I found:** Every step matches. mean, std and the input size are read from
model_info.json, not hard-coded. The canvas gives the pixels interleaved as RGBA, the loop
on lines 134–138 rearranges them into three planes R, G, B (CHW, like ToTensor) and drops
the alpha channel. The model outputs logits, so the page applies softmax (lines 142–147),
like `.softmax(dim=1)` in the notebook.

One difference: the notebook `transform` does no crop and no resize, because the photos
in the ZIP were already centre-cropped and resized to 96 × 96 in advance. For
check_image.jpg (already 96 × 96) the crop and resize on the page therefore change
nothing, which is why the numbers agree to about 1e-7. For the live camera the page
resizes a frame of up to 1280 px in the browser (`imageSmoothingQuality = "high"`),
which may differ slightly from how the training photos were resized. This is a possible
source of small differences between the camera and the notebook.

**What was fixed:** Nothing 

## 4. Nothing leaves the phone
**What I did:** …
I opened the page in Chrome on my PC with DevTools → Network
(cache disabled) and reloaded it. Then I uploaded a photo and ran the live camera
and watched for new requests, also with the Fetch/XHR filter.
I also checked index.html for any code that sends data.

| URL | size | why the page loads it |
|-----|------|-----------------------|
| kubous2.github.io/mlr-ex08/ (index.html) | 4.5 kB | the page itself |
| cdn.jsdelivr.net/…/onnxruntime-web@1.20.1/dist/ort.min.js | 105 kB | ONNX Runtime Web – runs the model in the browser (line 79) |
| cdn.jsdelivr.net/…/qrcode-generator@1.4.4/qrcode.js | 13.5 kB | generates the QR code in the browser (line 80) |
| cdn.jsdelivr.net/…/ort-wasm-simd-threaded.mjs + .wasm | 11 kB + 2.5 MB | WebAssembly backend of ONNX Runtime (line 95; run with 1 thread) |
| kubous2.github.io/mlr-ex08/model_info.json | 0.3 kB | classes, input size, mean, std (line 111) |
| kubous2.github.io/mlr-ex08/leaf_model.onnx | 5.7 MB | the model (line 115) |

**What I found:** …
All 7 requests are downloads (status 200) made once when the page loads.
The only fetch/XHR requests are model_info.json and leaf_model.onnx. Uploading a photo
and running the live camera created no new requests: the uploaded photo is opened only as
a local blob URL and the inference runs locally in
WebAssembly. index.html contains no POST request and no code that sends the image anywhere.
So nothing leaves the phone. (check_image.jpg and check_expected.json are downloaded
only when the self-test button is clicked.)

**What was fixed:** Nothing.

## What the agent could not check
Agent tested only in headless Chrome with a simulated camera. I also verified that the app works on a phone Android Samsung A54 5G in Google Chrome. 