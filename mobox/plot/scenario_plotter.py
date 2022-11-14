import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from matplotlib import patches, colors
from mobox.utils.geometry import rotate
from mobox.utils.plot import get_color, get_style_map


class ScenarioPlotter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.style_map = get_style_map()
        self.clear()

    def clear(self):
        plt.cla()
        plt.clf()
        plt.close("all")
        self.new_plot()

    def new_plot(self):
        dpi = self.cfg.PLOT.DPI
        s = self.cfg.PLOT.IMG_SIZE
        w, h = self.cfg.PLOT.MAP_SIZE
        a = w / h
        if w > h:
            w, h = a * s, s
        else:
            w, h = s, s / a

        fig, ax = plt.subplots(facecolor="black")
        fig.set_size_inches([w/dpi, h/dpi])
        fig.set_dpi(dpi)

        # Minimize plot margins and make axes invisible.
        plt.axis("off")
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, right=1,
                            left=0, hspace=0, wspace=0)

    def connect_points(self, points, line_stype="-", color="grey"):
        xs, ys = points[:, 0], points[:, 1]
        plt.plot(xs, ys, line_stype, color=color, linewidth=0.5, zorder=1)

    def plot_elem(self, df_elem):
        # Map elem type to line style.
        xy = df_elem[["px", "py"]]
        style, color = self.style_map[(df_elem[0, "type"], df_elem[0, "sub_type"])]
        self.connect_points(xy.to_numpy(), line_stype=style, color=color)
        return df_elem

    def plot_map(self, df_map):
        df_map.groupby("id").apply(self.plot_elem)

    def plot_box(self, row):
        x, y = row[0, "px"], row[0, "py"]
        w, h = row[0, "length"], row[0, "width"]
        angle = row[0, "yaw"]
        track_id = row[0, "track_id"]

        color = get_color(track_id)
        anchor = np.array([[x-w/2, y-h/2],  # left-bottom
                           [x+w/2, y],      # right-middle
                           [x-w/4, y-h/4]])
        pivot = np.array([[x, y]])
        rotated = rotate(anchor[None, :, :], np.array([angle]), pivot)[0]
        box = patches.Rectangle(rotated[0], w, h, fill=True, alpha=0.5,
                                color=color, linewidth=0.5, angle=angle*180/np.pi)
        plt.gca().add_patch(box)

        # Plot heading arrow.
        plt.plot([x, rotated[1][0]], [y, rotated[1][1]], "-", color=color, linewidth=0.5)

        # Plot track_id.
        plt.text(rotated[2][0], rotated[2][1], track_id, fontsize=6,
                 color=color, rotation=angle*180/np.pi, ha="center", va="center")
        return row

    def plot_track(self, df_track):
        H = self.cfg.TRACK.HISTORY_SIZE
        xys = df_track[["px", "py"]].to_numpy()
        color = get_color(df_track[0, "track_id"])
        plt.gca().scatter(xys[:, 0], xys[:, 1], s=0.1, color=color)
        self.plot_box(df_track[H])

    def get_viewport(self, scenario):
        df_ego = scenario.ego_track
        H = self.cfg.TRACK.HISTORY_SIZE
        w = h = self.cfg.PLOT.VIEWPORT_SIZE
        x, y = df_ego[H, "px"], df_ego[H, "py"]
        return (x-w/2, x+w/2, y-h/2, y+h/2)

    def plot(self, scenario):
        # Set viewport.
        viewport = self.get_viewport(scenario)
        plt.gca().axis(viewport)
        plt.gca().set_aspect("equal")

        # Plot map.
        self.plot_map(scenario.map)

        # Plot tracks.
        for track in scenario.focused_tracks:
            self.plot_track(track)

    def save(self, save_name="z.svg"):
        plt.savefig(save_name)


if __name__ == "__main__":
    from config.defaults import get_cfg
    from projects.wayformer.scenario_generator import WaymoScenarioGenerator
    cfg = get_cfg()
    plotter = ScenarioPlotter(cfg)
    gen = WaymoScenarioGenerator(cfg)

    for i, scenario in enumerate(gen.scenarios):
        print(scenario)
        plotter.plot(scenario)
        plotter.save(f"./img/{i}.svg")
        plotter.clear()
