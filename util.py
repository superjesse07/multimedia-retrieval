from pathlib import Path

from moderngl import Context


def load_program(file_path: Path, ctx: Context):
    with open(file_path.with_suffix(".vert"), "r") as f:
        vertex_shader = f.read()
    with open(file_path.with_suffix(".frag"), "r") as f:
        fragment_shader = f.read()
    return ctx.program(
        vertex_shader, fragment_shader
    )