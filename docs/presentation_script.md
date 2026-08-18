# 4-minute progress presentation — speaking script

Plain walkthrough of how the project will be done, step by step.

---

## 0:00 – 0:30 · The topic

> "Our topic is the Thai wooden frog.
>
> It's carved from one piece of hardwood, hollow inside, with a row of ridges cut
> across the back. You rub the wooden stick along the ridges and it croaks. So
> it's a Thai handicraft and a simple instrument at the same time.
>
> One note first — our original topic, the Khon masks, was already taken by
> another team, so we switched to this."

---

## 0:30 – 2:45 · How we're doing it, step by step

> "Here's the plan, step by step.
>
> **Step one — get the frog.** We buy one. They're cheap and sold in any
> handicraft market. We need the real object in our hands, not pictures off the
> internet, and I'll explain why in a moment.
>
> **Step two — photograph it properly.** We put it on a turntable, fix the camera
> on a tripod so it never moves, and rotate the frog in small steps — about ten
> degrees at a time. We repeat that at three different camera heights. That gives
> us roughly a hundred photos covering the frog from every angle, with the
> lighting and exposure locked the whole time.
>
> This is the step that decides everything. If the photos are bad, no amount of
> code fixes it later.
>
> And this is why we need the real object. You can't do this with images
> downloaded from the web, because those are photographs of *different* frogs —
> not the same frog from different angles. The software needs the same physical
> object each time.
>
> **Step three — build the 3D model.** Those photos go into reconstruction
> software that works out where the camera was for every shot and rebuilds the
> shape from that. The output is a 3D model of our frog with the real wood
> texture on it.
>
> We also have a backup here. If the turntable session doesn't work out, we can
> build a rougher 3D model from a single photograph instead. That code is already
> written and tested.
>
> **Step four — render it.** This is the main deliverable. We take the 3D model
> and generate views of it from angles we never actually photographed — spin it
> around, look at it from above, produce a turntable video. Our renderer is
> already built and working.
>
> **Step five — the classifier.** Separately, we collect photos of Thai wooden
> handicrafts and train a model to recognise which one it's looking at, so the
> system can name the object before it reconstructs it.
>
> **Step six — the workshop.** Put it all together into something a visitor can
> actually use at the table on the day."

---

## 2:45 – 3:30 · Where we are now

**→ point at the status table in the report**

> "Where we actually are:
>
> Step four is **done** — the renderer is built and tested, and we have measured
> numbers for it in the report.
>
> Step three is **half done** — the single-photo version is written and working.
>
> Steps one, two, five and six **haven't started**, because we only fixed the
> topic recently. We're not claiming any results on the frog itself yet."

---

## 3:30 – 4:00 · Next, then a question

> "So the immediate next thing is to buy a frog and shoot the turntable set.
> That one step unblocks everything after it.
>
> One question for you: for the workshop, would you rather we focus on doing one
> frog really well, or on something faster that works on whatever object a
> visitor brings to the table?"

---

## If you run long

Cut in this order:

1. The backup plan in step three
2. Step five (classifier) — say "and separately we train a classifier"
3. The detail in step two — keep only "turntable, fixed camera, about a hundred
   photos"

Never cut: why the real object is needed, step four, or the closing question.

## Likely questions

**"Why can't you use internet photos?"**
> "They're different frogs. Reconstruction needs many views of the same physical
> object — matching one frog against a different one gives nothing."

**"How accurate will the model be?"**
> "The turntable version should be good. The single-photo backup only recovers
> the side facing the camera, not the back — that's why we want the real capture."

**"Why write your own renderer instead of using Blender?"**
> "We wanted the projection and shading to be our own implementation rather than
> a library call. It's about 200 lines on top of OpenCV."

**"Where's the training part?"**
> "Step five, not started. The scripts carried over from our previous topic. We'd
> like to confirm the categories with you before we spend a day cleaning a
> dataset."
