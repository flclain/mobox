import numpy as np


def normalize_heading(yaw):
    """Normalize heading to [-PI, PI]."""
    yaw = np.mod(yaw, 2*np.pi)
    return np.where(yaw < np.pi, yaw, yaw - 2*np.pi)


def vector_angle_diff(a, b):
    """Get angle between two vectors, between [0,PI].

    Args:
      a: (np.array) input vector, sized [N,2].
      b: (np.array) input vector, sized [N,2].

    Returns:
      (np.array) angle diff in radian, sized [N,].
    """
    inner = (a * b).sum(axis=-1)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))  # [0,PI]
    return rad


def rotate(points, radian, pivot=np.array([[0, 0]])):
    """Rotate batch of points.

    Args:
      points: (np.array) input points to rotate, sized [N,M,2].
      radian: (np.array) rotate angle, sized [N,].
      pivot: (np.array) rotate pivot, sized [N,2].

    Returns:
      (np.array) rotated points, sized [N,M,2].
    """
    points = points - pivot
    sin = np.sin(radian)
    cos = np.cos(radian)
    R = np.array([[cos, -sin],
                  [sin, cos]]).transpose((2, 0, 1))  # [N,2,2]
    return (R @ points.transpose(0, 2, 1)).transpose(0, 2, 1) + pivot[:, None, :]


def rotate_vector(vector, radian, pivot=np.array([[0, 0]])):
    """Rotate vector around pivot.

    Args:
      vector: (np.array) input vector to rotate, sized [N,M,2].
      radian: (np.array) rotate angle, sized [N,].
      pivot: (np.array) rotate pivot, sized [N,2].

    Returns:
      (np.array) rotated points, sized [N,M,2].
    """
    v1 = rotate(vector, radian, pivot)
    v0 = rotate(np.zeros_like(vector), radian, pivot)
    return v1 - v0


if __name__ == "__main__":
    # points = np.array([[0, 0], [0, 1], [1, 0]]).reshape(1, 3, 2)
    # radian = np.array([np.pi/2])
    # ret = rotate(points, radian)
    # print(ret.shape)
    # a = np.array([[0, 1.], [1, 1]])
    # b = np.array([[1, 1.], [1, 1]])
    # print(vector_angle_diff(b, a))
    a = np.array([[1, 0.]])
    print(rotate_vector(a[None, :, :], [np.pi/2]))
