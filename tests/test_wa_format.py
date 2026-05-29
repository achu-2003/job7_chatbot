"""WhatsApp payload builders — confirm replies carry no interactive buttons."""
from app.chatbot import wa_format as wa


def test_image_message_has_caption_and_no_buttons():
    msg = wa.image_message("*Red Silk Saree*\n₹1,899", "https://cdn.example.com/a.jpg")
    assert msg["type"] == "image"
    assert msg["image"]["link"] == "https://cdn.example.com/a.jpg"
    assert msg["image"]["caption"].startswith("*Red Silk Saree*")
    # no interactive action block → no buttons
    assert "interactive" not in msg
    assert "action" not in msg.get("image", {})


def test_text_message_is_plain():
    assert wa.text_message("hello") == {"type": "text", "text": {"body": "hello"}}
