import numpy as np

from mobox.utils.geometry import rotate


class CoordinateTransform:
    """Transform global points & vectors to local coordinate frame."""

    @staticmethod
    def transform_points(points, pos):
        """Convert points to another coordinate frame.

        Args:
          points: (np.array) points in global coordinate frame, sized [N,M,2].
          pos: (np.array) local coordinate pose of (origin_x, origin_y, heading), sized [N,3].

        Returns:
          (np.array) transformed points coordinates, sized [N,M,2].
        """
        origin_xy = pos[:, :2]
        heading = pos[:, 2]
        points = rotate(points-origin_xy, np.pi/2-heading)
        return points

    @classmethod
    def transform_angles(cls, angles, pos):
        """Convert angles to another coordinate frame.

        Args:
          angles: (np.array) angles in global coordinate frame, sized [N,M].
          pos: (np.array) local coordinate pose of (origin_x, origin_y, heading), sized [N,3].

        Returns:
          (np.array) transformed angles, sized [N,M].
        """
        dx = np.cos(angles)
        dy = np.sin(angles)
        v = np.stack([dx, dy], axis=2)
        v = cls.transform_vector(v, pos)
        angles = np.arctan2(v[:, :, 1], v[:, :, 0])
        return angles

    @classmethod
    def transform_vector(cls, vector, pos):
        """Convert vector to another coordinate frame.

        Args:
          vector: (np.array) vector in global coordinate frame, sized [N,M,2].
          pos: (np.array) local coordinate pose of (origin_x, origin_y, heading), sized [N,3].

        Returns:
          (np.array) transformed points coordinates, sized [N,M,2].
        """
        v1 = cls.transform_points(vector, pos)
        v0 = cls.transform_points(np.zeros_like(vector), pos)
        return v1-v0


def test_transform_points():
    points = np.array([[0, 1.], [1., 0], [1., 1.]])
    pos = np.array([[1, 1, np.pi/2]])
    ret = CoordinateTransform.transform_points(points[None, :, :], pos)
    print(ret)


def test_transform_angles():
    angle = np.array([[0, np.pi/4, np.pi/2, np.pi]])
    pos = np.array([[0, 0, -np.pi/2]])
    ret = CoordinateTransform.transform_angles(angle, pos)
    print(np.rad2deg(ret))


def test_transform_vector():
    vector = np.array([[1, 1.]])
    pos = np.array([[0, 0, -np.pi/2]])
    ret = CoordinateTransform.transform_vector(vector[None, :, :], pos)
    print(ret)


if __name__ == "__main__":
    # test_transform_points()
    # test_transform_angles()
    test_transform_vector()
