// perturber.js — Pure Headless AI Perturbation Model (No UI)
let _model = null;

async function getModel() {
  if (!_model) {
    _model = await mobilenet.load({ version: 2, alpha: 1.0 });
  }
  return _model;
}

/**
 * Takes an uploaded File (from <input type="file">) and returns a perturbed PNG File
 * @param {File} imageFile - The original image uploaded by the user
 * @param {number} strength - Perturbation strength (default: 16 out of 255)
 * @returns {Promise<{file: File, dataUrl: string}>}
 */
export async function perturbUploadedFile(imageFile, strength = 16) {
  const net = await getModel();

  // 1. Load uploaded File into an off-screen Image object
  const img = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = reject;
      el.src = reader.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(imageFile);
  });

  const origW = img.naturalWidth || 224;
  const origH = img.naturalHeight || 224;

  // 2. Run 15-step PGD adversarial gradient optimization
  const dataUrl = tf.tidy(() => {
    const raw = tf.browser.fromPixels(img).toFloat();
    const resized = tf.image.resizeBilinear(raw, [224, 224]);
    const base = resized.div(127.5).sub(1.0).expandDims(0);

    const origLogits = net.model.predict(base);
    const origClass = origLogits.argMax(1).dataSync().at(0);
    const oneHot = tf.oneHot(tf.tensor1d([origClass], 'int32'), origLogits.shape.at(1));

    let delta = tf.zerosLike(base);
    const eps = strength / 127.5;
    const alpha = (strength / 5) / 127.5;

    const lossFn = (d) => {
      const adv = base.add(d).clipByValue(-1, 1);
      return tf.losses.softmaxCrossEntropy(oneHot, net.model.predict(adv));
    };
    const gradFn = tf.grad(lossFn);

    for (let i = 0; i < 15; i++) {
      const grad = gradFn(delta);
      delta = delta.add(grad.sign().mul(alpha)).clipByValue(-eps, eps);
      delta = base.add(delta).clipByValue(-1, 1).sub(base);
    }

    // 3. Apply perturbation back onto original rectangular resolution
    const deltaFull = tf.image.resizeBilinear(delta.squeeze(0), [origH, origW]);
    const finalImg = raw
      .div(127.5).sub(1.0).add(deltaFull)
      .clipByValue(-1, 1).add(1.0).mul(127.5)
      .clipByValue(0, 255).toInt();

    const canvas = document.createElement('canvas');
    canvas.width = origW;
    canvas.height = origH;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(origW, origH);
    const pixels = finalImg.dataSync();

    for (let i = 0, j = 0; i < pixels.length; i += 3, j += 4) {
      imgData.data.set([pixels.at(i), pixels.at(i + 1), pixels.at(i + 2), 255], j);
    }
    ctx.putImageData(imgData, 0, 0);
    return canvas.toDataURL('image/png');
  });

  // 4. Convert back into a standard File object ready for your website
  const res = await fetch(dataUrl);
  const blob = await res.blob();
  const perturbedFile = new File([blob], 'perturbed_' + imageFile.name.replace(/\.[^/.]+$/, '') + '.png', {
    type: 'image/png'
  });

  return { file: perturbedFile, dataUrl };
}
