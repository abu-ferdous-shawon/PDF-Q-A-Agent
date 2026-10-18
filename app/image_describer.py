from langchain_core.messages import HumanMessage

IMAGE_DESCRIPTION_PROMPT = (
    "Describe this image in detail. If it is a chart, graph, diagram, "
    "flowchart, or a table rendered as an image, extract all visible "
    "data points, labels, axes, categories, and numeric values as "
    "precisely as possible. If it is a photo or illustration, describe "
    "what it depicts factually. Do not speculate about anything not "
    "visible in the image."
)


def describe_image(llm, image_b64: str) -> str:
    message = HumanMessage(
        content=[
            {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
            {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"},
        ]
    )
    response = llm.invoke([message])
    content = response.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content.strip()