import gradio as gr
import torch
from PIL import Image, ImageDraw
from ultralytics import SAM

# 1. Indlæs din lokale sam3.pt model
model = SAM("modelData/sam3/sam3.pt")


def draw_points(image_np, points):
    """Tegner røde markører for hvert klik-punkt, til visuel feedback."""
    preview = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(preview)
    radius = 6
    for x, y in points:
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill="red",
            outline="white",
        )
    return preview


def on_upload(image):
    """Nyt billede uploadet: nulstil punkter."""
    if image is None:
        return None, []
    return image, []


def on_select(image_np, points, evt: gr.SelectData):
    """Bruger har klikket på billedet: tilføj punktet og vis feedback."""
    if image_np is None:
        return points, None

    x, y = evt.index
    points = points + [[x, y]]

    return points, draw_points(image_np, points)


def clear_points(image_np):
    """Nulstil alle klik-punkter."""
    if image_np is None:
        return [], None
    return [], Image.fromarray(image_np)


def segment_by_click(image_np, points):
    if image_np is None:
        return None

    if not points:
        results = model.predict(source=image_np, verbose=False)
        annotated_img = results[0].plot(labels=False, boxes=False)
        return Image.fromarray(annotated_img[..., ::-1])

    labels = [1] * len(points)

    # multimask_output isn't exposed through the high-level predict() API, so
    # get a predictor instance first, then call it directly. A single-mask
    # point prompt is often ambiguous - it can lock onto just the stud the
    # click landed on instead of the whole brick - so multimask_output=True
    # asks SAM for several whole/part candidates per point instead of one.
    if model.predictor is None:
        model.predict(source=image_np, points=points, labels=labels, verbose=False)
    else:
        model.predictor.reset_image()
    results = model.predictor(
        source=image_np, points=points, labels=labels, multimask_output=True
    )
    res = results[0]

    # Each point yields several candidates (whole object, part, sub-part), and
    # on a crowded scene some candidates span multiple bricks at once. Keep,
    # per point, only the best-scoring candidate whose mask covers exactly
    # that one clicked point and no other - this rules out both sub-object
    # slivers and merged multi-object blobs.
    masks = res.masks.data
    best_idx = {}
    for i in range(len(masks)):
        covering = [j for j, (x, y) in enumerate(points) if masks[i, int(y), int(x)] > 0.5]
        if len(covering) != 1:
            continue
        j = covering[0]
        if j not in best_idx or res.boxes.conf[i] > res.boxes.conf[best_idx[j]]:
            best_idx[j] = i

    if best_idx:
        res = res[torch.tensor(sorted(best_idx.values()))]

    # Tegn masken oven på det originale billede
    annotated_img = res.plot(labels=False, boxes=False)

    return Image.fromarray(annotated_img[..., ::-1])


# 2. Opbyg Web-UI
with gr.Blocks() as demo:
    gr.Markdown("# 🖱️ SAM 3 - Klik for at segmentere")
    gr.Markdown(
        "Upload et billede, **klik direkte på de objekter** du vil vælge, og tryk på knappen."
    )

    # State: originalbillede samt de klikkede punkter
    state_image = gr.State(None)
    state_points = gr.State([])

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(
                label="Input Billede (Klik for at sætte punkter)",
                type="numpy",
                sources=["upload", "webcam"],
            )
            with gr.Row():
                submit_btn = gr.Button("Segmentér med SAM 3", variant="primary")
                clear_btn = gr.Button("Ryd punkter")

        with gr.Column():
            output_img = gr.Image(label="SAM 3 Output", type="pil")

    input_img.upload(
        fn=on_upload,
        inputs=[input_img],
        outputs=[state_image, state_points],
    )

    input_img.select(
        fn=on_select,
        inputs=[state_image, state_points],
        outputs=[state_points, input_img],
    )

    clear_btn.click(
        fn=clear_points,
        inputs=[state_image],
        outputs=[state_points, input_img],
    )

    submit_btn.click(
        fn=segment_by_click,
        inputs=[state_image, state_points],
        outputs=output_img,
    )

if __name__ == "__main__":
    demo.launch()
