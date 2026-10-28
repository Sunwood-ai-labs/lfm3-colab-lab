from __future__ import annotations

import base64

from IPython.display import Video

from lfm3_lab import display


def test_video_display_embeds_the_selected_file_in_the_notebook(tmp_path, monkeypatch):
    path = tmp_path / "selected.mp4"
    contents = b"test-file-transport"
    path.write_bytes(contents)
    rendered = []
    monkeypatch.setattr(display, "_display", rendered.append)
    display._display_media(path)
    video = next(item for item in rendered if isinstance(item, Video))
    assert base64.b64encode(contents).decode("ascii") in video._repr_html_()
