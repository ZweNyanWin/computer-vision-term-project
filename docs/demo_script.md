# Four-minute demonstration — 31 August 2026 checkpoint

Run `./demo.sh` and press Enter between steps. Total compute is about 16 seconds;
the rest of the time is talking. Every slow stage was run beforehand and is read
back from the file it wrote, and the script prints the command that produced it.

**Before you start:** terminal font large and `demo.sh` already `cd`-ed into the
project. While online, warm the depth-model cache once with
`DEMO_ALLOW_DOWNLOAD=1 ./demo.sh 3`; a cold cache downloads about 100 MB. The
normal demo runs step 3 in cache-only mode so a poor venue connection cannot
stall it with network retries. Then run the complete demo once through.

---

## 0:00 – 0:30 · What changed

> "At the last checkpoint everything ran on a synthetic proxy, because we had no
> photographs. Since then we have bought the frog, photographed it properly, run
> the learned depth model, and measured the result against photographs the
> pipeline never saw. So this time the numbers are about the frog."

**Run step 1.** It prints 36 ring photographs and 9 elevated ones.

> "A closed turntable ring: the camera stays fixed and the frog turns ten degrees
> at a time, so the lighting belongs to the room rather than to the object. Nine
> more from about sixty degrees above."

---

## 0:30 – 1:10 · Segmentation had to change

**Run step 2.** → `36 frames checked … frames with background in the corners: 0`

> "Segmentation was thresholding on brightness, and on real photographs it broke.
> A real backdrop is never evenly lit — across these frames the corners span 55
> to 199 grey levels, so a global threshold cuts through the background instead
> of around the frog. On one frame it inverted completely: it selected the
> backdrop, and reported a perfectly plausible seventy per cent foreground while
> doing it.
>
> It now thresholds on saturation, because a neutral backdrop stays desaturated
> under any lighting while the wood keeps its hue. Otsu separates it at 0.82 to
> 0.93 against 0.60 to 0.73 for brightness. All thirty-six frames come out clean."

*If asked how you knew it was wrong:* the foreground percentage looked fine — we
only caught it by looking at the mask.

---

## 1:10 – 1:50 · Learned depth, then views nobody photographed

**Run step 3.** Point at the line `depth map was inverse depth; flipped …`

> "Depth Anything V2 on one real photograph — three seconds, no GPU.
>
> That line matters. The mesh uses the depth value directly as a height, and our
> camera sits on the low side, so a bigger number means further away. This family
> of networks predicts the opposite. Used raw, every bump becomes a dent and the
> frog renders inside out. Rather than trust the checkpoint's convention we
> measure it: the frog stands in front of its backdrop, so whichever side of the
> mask holds the smaller values is the near side."

**Run step 4.** Five views from one photograph.

---

## 1:50 – 2:40 · The measurement, and what it does not show

**Run step 5.**

> "We withhold photographs, synthesise the view at each withheld angle from the
> nearest photograph we kept, and score it against the one we withheld. Then we
> score a second thing against the same photograph: the nearest kept photograph
> itself, unaltered — what you would see if you just snapped to the closest
> frame. That second column is what makes the first mean anything.
>
> One result clears its confidence interval: at sixty-degree spacing the
> synthesised view is structurally closer to the withheld photograph, by 0.024
> SSIM. No PSNR difference at any spacing is distinguishable from zero.
>
> We had a larger claim here and withdrew it. An earlier run showed a gain at
> forty degrees; adding one more photograph turned it from plus 0.03 to minus
> 0.23 decibels. The effect was smaller than the noise, so it is gone, and every
> number now carries its interval."

*This paragraph is the strongest thing you will say. Do not cut it.*

---

## 2:40 – 3:20 · Why one photograph is not enough

**Run step 6.** → relief keeps 0.8% behind; mesh keeps 43.1%.

> "Single-image reconstruction gives a relief, not a solid. Swing the camera
> behind it and almost nothing survives — under one per cent of its triangles.
>
> The same photographs through Apple's Object Capture give a closed surface:
> twenty-five thousand vertices, eighty seconds, no GPU and no cloud. Behind it,
> forty-three per cent of the surface is still there, because there is a back."

**Run step 7.** The `.usdz` opens — drag it live. Then the turntable video.

> "That is our own renderer, not Apple's viewer — the same projection, culling
> and shading we wrote for the assignment, now with geometry worth rendering."

---

## 3:20 – 4:00 · Honest limits, and what is next

> "Three limits. The advantage over simply showing the nearest photograph is
> small — at the spacings a dense capture actually gives you, frame-switching is
> nearly as good. Depth is relative, not metric, so nothing here is in
> millimetres. And rendering is not yet interactive, which matters because the
> workshop station is the remaining deliverable.
>
> One question for you: the reconstruction is photogrammetry. Does the project
> need a 3D model produced specifically by neural methods — Gaussian splatting or
> a radiance field — or does the learned depth already in the pipeline satisfy
> that?"

**Ending on that question is deliberate — it is the one decision that changes
what you build over the next three weeks.**

---

## If you run long

Cut in this order:

1. Step 4 (the five novel views — step 6 makes the same point better)
2. Step 1 (say the numbers instead of printing them)
3. The segmentation detail in step 2 — keep the fact that it broke and was fixed

**Never cut:** the withdrawn claim at 2:40, the relief-versus-mesh numbers, or
the closing question.

## Likely questions

**"Why is PSNR so low?"**
> Both columns are silhouette-cropped comparisons between a render and a real
> photograph, which is conservative by construction. The gain column is the
> claim; the absolute column is context. A comparable project on the same course
> reports 13.9 dB by a similar protocol.

**"Why does SSIM disagree with PSNR?"**
> PSNR rewards sharp pixels, and the baseline is a real photograph — just at the
> wrong angle. SSIM compares structure, and our render is at the right angle. For
> novel-view synthesis structure is what matters, but we report both.

**"Is Object Capture allowed? Isn't it a rendering engine?"**
> It reconstructs geometry, it does not render. Everything shown is drawn by our
> own projection, hidden-surface and shading code. The mesh is reported as a
> comparison, not as a replacement for the pipeline.

**"Can you compare the mesh's score with the relief's?"**
> No, and we say so in the report. Every photograph scored against the mesh also
> went into building it, so it measures fit rather than prediction. The
> comparison we can make is coverage: 43.1 per cent against 0.8.
