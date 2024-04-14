import dearpygui.dearpygui as dpg
import detection as dt
import asyncio
import threading as th
import random
import numpy as np

MATRIX_WIDTH = 32
MATRIX_HEIGHT = 32
PIXEL_SIZE = 16


class LEDMatrixSimulator:

    def __init__(self, eye_type="left"):
        self.eye_type = eye_type
        self.width = MATRIX_WIDTH
        self.height = MATRIX_HEIGHT
        self.matrix = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.detector = dt.Detector()

        try:
            self.load_image("./textures/white.png", "white_texture")
            self.load_image("./textures/black.png", "black_texture")
        except:
            pass

    def load_image(self, path, name):
        width, height, channels, data = dpg.load_image(path)
        with dpg.texture_registry():
            dpg.add_static_texture(width, height, data, tag=name)

    def draw_pixels(self, pixels):
        try:
            for row in range(self.height):
                for col in range(self.width):
                    index = row * self.width + col
                    texture_tag = (
                        "white_texture" if pixels[index] == 1 else "black_texture"
                    )
                    dpg.configure_item(
                        f"pixel_{row}_{col}_{self.eye_type}", texture_tag=texture_tag
                    )
        except:
            pass

    def update(self):
        while True:
            # Get the eye image from the detector
            result = self.detector.detect(eye_type=self.eye_type)
            if result is not None:
                pixels = result.flatten().tolist()
                self.draw_pixels(pixels)


async def main():
    dpg.create_context()
    dpg.create_viewport(
        title="LED Matrix Simulator",
        width=(MATRIX_WIDTH * PIXEL_SIZE * 2) + 32,
        height=MATRIX_HEIGHT * PIXEL_SIZE,
        resizable=False,
    )

    left_simulator = LEDMatrixSimulator(eye_type="left")
    right_simulator = LEDMatrixSimulator(eye_type="right")

    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            # column spacing
            dpg.add_theme_color(dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 0))

    with dpg.window(label="eye_left"):
        for row in range(left_simulator.height):
            with dpg.group(horizontal=True):
                for col in range(left_simulator.width):
                    dpg.add_image(
                        "black_texture",
                        width=PIXEL_SIZE,
                        height=PIXEL_SIZE,
                        user_data=(row, col),
                        tag=f"pixel_{row}_{col}_left",
                    )

    with dpg.window(label="eye_right", pos=((MATRIX_HEIGHT * PIXEL_SIZE) + 16, 0)):
        for row in range(right_simulator.height):
            with dpg.group(horizontal=True):
                for col in range(right_simulator.width):
                    dpg.add_image(
                        "black_texture",
                        width=PIXEL_SIZE,
                        height=PIXEL_SIZE,
                        user_data=(row, col),
                        tag=f"pixel_{row}_{col}_right",
                    )

    dpg.bind_theme(global_theme)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    # dpg.show_style_editor()

    th.Thread(target=left_simulator.update).start()
    th.Thread(target=right_simulator.update).start()

    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    asyncio.run(main())
