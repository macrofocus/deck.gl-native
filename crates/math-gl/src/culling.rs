//! Port of the parts of `@math.gl/culling` that the tile traversal needs: planes, a culling
//! volume, axis aligned and oriented bounding boxes, and the oriented bounding box that fits a
//! set of points (math.gl's port of Cesium's, Apache 2).

use glam::{DMat3, DVec3};

/// `INTERSECTION`: where a bounding volume lies relative to a plane or a culling volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intersection {
    /// Not contained within the frustum
    Outside = -1,
    /// Crosses one of the frustum's planes
    Intersecting = 0,
    /// Fully within the frustum
    Inside = 1,
}

/// `Plane`: the points `p` with `normal . p + distance == 0`; the side the normal points to is
/// the positive one. The normal is expected to be normalized.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    pub normal: DVec3,
    pub distance: f64,
}

impl Plane {
    pub fn new(normal: DVec3, distance: f64) -> Self {
        Self { normal, distance }
    }

    /// The signed shortest distance of a point to the plane.
    pub fn distance_to(&self, point: DVec3) -> f64 {
        self.normal.dot(point) + self.distance
    }
}

/// A volume `CullingVolume` can test against its planes.
pub trait BoundingVolume {
    /// Which side of `plane` the volume is on.
    fn intersect_plane(&self, plane: &Plane) -> Intersection;
    /// The distance from the closest point of the volume to `point`.
    fn distance_to(&self, point: DVec3) -> f64;
}

/// `CullingVolume`: planes whose positive sides enclose the visible space, a frustum
/// typically.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CullingVolume {
    pub planes: Vec<Plane>,
}

impl CullingVolume {
    pub fn new(planes: Vec<Plane>) -> Self {
        Self { planes }
    }

    /// `computeVisibility`: outside as soon as the volume is outside one plane.
    pub fn compute_visibility(&self, volume: &impl BoundingVolume) -> Intersection {
        let mut intersect = Intersection::Inside;
        for plane in &self.planes {
            match volume.intersect_plane(plane) {
                Intersection::Outside => return Intersection::Outside,
                Intersection::Intersecting => intersect = Intersection::Intersecting,
                Intersection::Inside => {}
            }
        }
        intersect
    }
}

/// `AxisAlignedBoundingBox`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisAlignedBoundingBox {
    pub minimum: DVec3,
    pub maximum: DVec3,
    pub center: DVec3,
    pub half_diagonal: DVec3,
}

impl AxisAlignedBoundingBox {
    pub fn new(minimum: DVec3, maximum: DVec3) -> Self {
        let center = (minimum + maximum) * 0.5;
        Self {
            minimum,
            maximum,
            center,
            half_diagonal: maximum - center,
        }
    }

    pub fn distance_squared_to(&self, point: DVec3) -> f64 {
        let offset = point - self.center;
        let mut distance_squared = 0.0;
        for i in 0..3 {
            let d = offset[i].abs() - self.half_diagonal[i];
            if d > 0.0 {
                distance_squared += d * d;
            }
        }
        distance_squared
    }
}

impl BoundingVolume for AxisAlignedBoundingBox {
    fn intersect_plane(&self, plane: &Plane) -> Intersection {
        let h = self.half_diagonal;
        let n = plane.normal;
        let e = h.x * n.x.abs() + h.y * n.y.abs() + h.z * n.z.abs();
        // signed distance from the center
        let s = self.center.dot(n) + plane.distance;
        if s - e > 0.0 {
            return Intersection::Inside;
        }
        // Not in front because normals point inward
        if s + e < 0.0 {
            return Intersection::Outside;
        }
        Intersection::Intersecting
    }

    fn distance_to(&self, point: DVec3) -> f64 {
        self.distance_squared_to(point).sqrt()
    }
}

/// `OrientedBoundingBox`: a cuboid around `center` whose columns of `half_axes` are its half
/// edges.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrientedBoundingBox {
    pub center: DVec3,
    pub half_axes: DMat3,
}

impl Default for OrientedBoundingBox {
    fn default() -> Self {
        Self {
            center: DVec3::ZERO,
            half_axes: DMat3::ZERO,
        }
    }
}

impl OrientedBoundingBox {
    pub fn new(center: DVec3, half_axes: DMat3) -> Self {
        Self { center, half_axes }
    }

    /// The lengths of the three half axes.
    pub fn half_size(&self) -> DVec3 {
        DVec3::new(
            self.half_axes.x_axis.length(),
            self.half_axes.y_axis.length(),
            self.half_axes.z_axis.length(),
        )
    }

    /// `distanceSquaredTo`, Geometric Tools for Computer Graphics 10.4.2.
    pub fn distance_squared_to(&self, point: DVec3) -> f64 {
        let offset = point - self.center;
        let mut distance_squared = 0.0;
        for axis in [
            self.half_axes.x_axis,
            self.half_axes.y_axis,
            self.half_axes.z_axis,
        ] {
            let half = axis.length();
            // math.gl's normalize leaves a zero vector at zero
            let d = offset.dot(axis.normalize_or_zero()).abs() - half;
            if d > 0.0 {
                distance_squared += d * d;
            }
        }
        distance_squared
    }
}

impl BoundingVolume for OrientedBoundingBox {
    fn intersect_plane(&self, plane: &Plane) -> Intersection {
        let n = plane.normal;
        let h = &self.half_axes;
        // The plane is used as if it is its normal; its first three components are assumed to
        // be normalized
        let rad_effective = n.dot(h.x_axis).abs() + n.dot(h.y_axis).abs() + n.dot(h.z_axis).abs();
        let distance_to_plane = n.dot(self.center) + plane.distance;
        if distance_to_plane <= -rad_effective {
            // The entire box is on the negative side of the plane normal
            Intersection::Outside
        } else if distance_to_plane >= rad_effective {
            // The entire box is on the positive side of the plane normal
            Intersection::Inside
        } else {
            Intersection::Intersecting
        }
    }

    fn distance_to(&self, point: DVec3) -> f64 {
        self.distance_squared_to(point).sqrt()
    }
}

/// `makeOrientedBoundingBoxFromPoints`: Stefan Gottschalk's fit, the box along the principal
/// axes of the points' covariance.
pub fn make_oriented_bounding_box_from_points(positions: &[DVec3]) -> OrientedBoundingBox {
    if positions.is_empty() {
        return OrientedBoundingBox::default();
    }
    let inv_length = 1.0 / positions.len() as f64;
    let mean = positions.iter().copied().sum::<DVec3>() * inv_length;

    let (mut exx, mut exy, mut exz, mut eyy, mut eyz, mut ezz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for position in positions {
        let p = *position - mean;
        exx += p.x * p.x;
        exy += p.x * p.y;
        exz += p.x * p.z;
        eyy += p.y * p.y;
        eyz += p.y * p.z;
        ezz += p.z * p.z;
    }
    let covariance = DMat3::from_cols_array(&[exx, exy, exz, exy, eyy, eyz, exz, eyz, ezz]) * inv_length;

    let rotation = compute_eigen_decomposition(&covariance).unitary;
    let (v1, v2, v3) = (rotation.x_axis, rotation.y_axis, rotation.z_axis);

    let mut upper = DVec3::splat(-f64::MAX);
    let mut lower = DVec3::splat(f64::MAX);
    for position in positions {
        let along = DVec3::new(position.dot(v1), position.dot(v2), position.dot(v3));
        upper = upper.max(along);
        lower = lower.min(along);
    }
    let middle = (lower + upper) * 0.5;
    let center = v1 * middle.x + v2 * middle.y + v3 * middle.z;
    let scale = (upper - lower) * 0.5;
    OrientedBoundingBox {
        center,
        half_axes: rotation * DMat3::from_diagonal(scale),
    }
}

/// The result of [`compute_eigen_decomposition`]: `matrix = unitary * diagonal * unitaryᵀ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EigenDecomposition {
    /// Columns are the eigenvectors
    pub unitary: DMat3,
    /// The diagonal holds the eigenvalues
    pub diagonal: DMat3,
}

const EIGEN_TOLERANCE: f64 = 1e-20;
const EIGEN_MAX_SWEEPS: u32 = 10;
const SHUR_TOLERANCE: f64 = 1e-15;
const ROW_VAL: [usize; 3] = [1, 0, 0];
const COL_VAL: [usize; 3] = [2, 2, 1];

/// `computeEigenDecomposition`: eigenvectors and eigenvalues of a symmetric matrix by the
/// classical Jacobi algorithm (Golub and Van Loan, Matrix Computations, 8.4.3).
pub fn compute_eigen_decomposition(matrix: &DMat3) -> EigenDecomposition {
    let mut count = 0;
    let mut sweep = 0;
    let mut unitary = DMat3::IDENTITY;
    let mut diagonal = *matrix;
    let epsilon = EIGEN_TOLERANCE * frobenius_norm(&diagonal);
    while sweep < EIGEN_MAX_SWEEPS && off_diagonal_frobenius_norm(&diagonal) > epsilon {
        let j = shur_decomposition(&diagonal);
        diagonal = j.transpose() * (diagonal * j);
        unitary *= j;
        count += 1;
        if count > 2 {
            sweep += 1;
            count = 0;
        }
    }
    EigenDecomposition { unitary, diagonal }
}

fn element(matrix: &DMat3, row: usize, col: usize) -> f64 {
    matrix.col(col)[row]
}

fn frobenius_norm(matrix: &DMat3) -> f64 {
    matrix.to_cols_array().iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// The "off-diagonal" Frobenius norm of a symmetric matrix.
fn off_diagonal_frobenius_norm(matrix: &DMat3) -> f64 {
    (0..3)
        .map(|i| {
            let temp = element(matrix, COL_VAL[i], ROW_VAL[i]);
            2.0 * temp * temp
        })
        .sum::<f64>()
        .sqrt()
}

/// The Jacobi rotation that reduces the largest off-diagonal term of a symmetric matrix (Golub
/// and Van Loan, 8.4.2 The 2by2 Symmetric Schur Decomposition).
///
/// math.gl took this from Cesium, whose `getElementIndex(column, row)` has its arguments the
/// other way round to math.gl's `(row, col)`, so the rotation is written transposed and the
/// decomposition does not converge to the eigenvalues. It is kept as math.gl has it: the
/// result is still orthonormal, so boxes built on it still hold their points, and the tile
/// traversal then picks the tiles deck.gl picks.
fn shur_decomposition(matrix: &DMat3) -> DMat3 {
    let mut max_diagonal = 0.0;
    let mut rot_axis = 1;
    // pivot on the largest off-diagonal term
    for i in 0..3 {
        let temp = element(matrix, COL_VAL[i], ROW_VAL[i]).abs();
        if temp > max_diagonal {
            rot_axis = i;
            max_diagonal = temp;
        }
    }
    let p = ROW_VAL[rot_axis];
    let q = COL_VAL[rot_axis];
    let mut c = 1.0;
    let mut s = 0.0;
    if element(matrix, q, p).abs() > SHUR_TOLERANCE {
        let qq = element(matrix, q, q);
        let pp = element(matrix, p, p);
        let qp = element(matrix, q, p);
        let tau = (qq - pp) / 2.0 / qp;
        let t = if tau < 0.0 {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        } else {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        };
        c = 1.0 / (1.0 + t * t).sqrt();
        s = t * c;
    }
    let mut result = DMat3::IDENTITY.to_cols_array();
    // math.gl's getElementIndex(row, col) is col * 3 + row
    let index = |row: usize, col: usize| col * 3 + row;
    result[index(p, p)] = c;
    result[index(q, q)] = c;
    result[index(q, p)] = s;
    result[index(p, q)] = -s;
    DMat3::from_cols_array(&result)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_POINTS: [DVec3; 4] = [
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(2.0, 0.0, 0.0),
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 1.0, 2.0),
    ];

    #[test]
    fn oriented_bounding_box_of_no_points_is_empty() {
        let b = make_oriented_bounding_box_from_points(&[]);
        assert_eq!(b.center, DVec3::ZERO);
        assert_eq!(b.half_size(), DVec3::ZERO);
    }

    #[test]
    fn oriented_bounding_box_of_one_point_is_the_point() {
        let point = DVec3::new(1.0, 2.0, 3.0);
        let b = make_oriented_bounding_box_from_points(&[point]);
        assert!(b.center.abs_diff_eq(point, 1e-12), "{:?}", b.center);
        assert_eq!(b.half_size(), DVec3::ZERO);
    }

    #[test]
    fn oriented_bounding_box_contains_its_points() {
        let b = make_oriented_bounding_box_from_points(&TEST_POINTS);
        for point in TEST_POINTS {
            assert!(b.distance_to(point) < 1e-12, "{point} is inside the bounding box");
        }
    }

    #[test]
    fn oriented_bounding_box_of_a_cube_has_its_half_axes() {
        let b = make_oriented_bounding_box_from_points(&[
            DVec3::new(2.0, -2.0, -2.0),
            DVec3::new(-2.0, 2.0, 2.0),
            DVec3::new(-2.0, -2.0, -2.0),
            DVec3::new(2.0, 2.0, 2.0),
            DVec3::new(2.0, -2.0, 2.0),
            DVec3::new(-2.0, 2.0, -2.0),
            DVec3::new(2.0, 2.0, -2.0),
            DVec3::new(-2.0, -2.0, 2.0),
        ]);
        assert_eq!(b.half_axes, DMat3::from_diagonal(DVec3::splat(2.0)));
        assert_eq!(b.half_size(), DVec3::splat(2.0));
        assert_eq!(b.half_size().element_product(), 8.0);
    }

    #[test]
    fn eigen_decomposition_is_an_orthonormal_similarity() {
        // row major 4 -1 1 / -1 3 -2 / 1 -2 3, symmetric so the same column major
        let a = DMat3::from_cols_array(&[4.0, -1.0, 1.0, -1.0, 3.0, -2.0, 1.0, -2.0, 3.0]);
        let EigenDecomposition { unitary, diagonal } = compute_eigen_decomposition(&a);
        // math.gl's own test leaves the eigenvalue checks out ("not yet quite right", see
        // `shur_decomposition`); what holds is that `unitary` is a rotation relating the two
        assert!(
            (unitary * unitary.transpose()).abs_diff_eq(DMat3::IDENTITY, 1e-12),
            "{unitary:?}"
        );
        let b = unitary * diagonal * unitary.transpose();
        assert!(b.abs_diff_eq(a, 1e-12), "{b:?}");
    }

    /// math.gl's `intersectPlaneTestCornersEdgesFaces` for the untransformed unit box: planes
    /// just outside, just inside and just through the faces, edges and corners.
    #[test]
    fn intersect_plane_classifies_faces_edges_and_corners() {
        let b = OrientedBoundingBox::new(DVec3::ZERO, DMat3::IDENTITY * 0.5);
        let plane = |n: DVec3, dist: f64| -> Option<Plane> {
            let arb = DVec3::new(357.0, 924.0, 258.0);
            let p0 = n.normalize() * -dist;
            let tang = n.cross(arb).normalize();
            let binorm = n.cross(tang).normalize();
            let n = tang.cross(binorm);
            if n.length() == 0.0 {
                return None;
            }
            let n = n.normalize();
            let d = -p0.dot(n);
            (d.abs() > 0.0001 && n.length_squared() > 0.0001).then(|| Plane::new(n, d))
        };
        let mut normals: Vec<(DVec3, f64)> = Vec::new();
        for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
            normals.push((axis, 0.5));
            normals.push((-axis, 0.5));
        }
        for (a, b) in [(DVec3::X, DVec3::Y), (DVec3::X, DVec3::Z), (DVec3::Y, DVec3::Z)] {
            for (sa, sb) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                normals.push((a * sa + b * sb, 0.5f64.sqrt()));
            }
        }
        for sx in [1.0, -1.0] {
            for sy in [1.0, -1.0] {
                for sz in [1.0, -1.0] {
                    normals.push((DVec3::new(sx, sy, sz), 0.75f64.sqrt()));
                }
            }
        }
        let eps = 0.00001;
        for (n, r) in normals {
            for (dist, expected) in [
                (r + eps, Intersection::Inside),
                (r - eps, Intersection::Intersecting),
                (-r + eps, Intersection::Intersecting),
                (-r - eps, Intersection::Outside),
            ] {
                if let Some(pl) = plane(n, dist) {
                    assert_eq!(b.intersect_plane(&pl), expected, "normal {n}, distance {dist}");
                }
            }
        }
    }

    #[test]
    fn distance_squared_to_a_rotated_box() {
        let rotation_z = |a: f64| {
            let (s, c) = a.sin_cos();
            DMat3::from_cols_array(&[c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0])
        };
        let rotation_y = |a: f64| {
            let (s, c) = a.sin_cos();
            DMat3::from_cols_array(&[c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c])
        };
        let rotation = rotation_y(45f64.to_radians()) * rotation_z(-45f64.to_radians());
        let scale = DVec3::new(2.0, 3.0, 4.0);
        let center = DVec3::new(4.0, 3.0, 2.0);
        let obb = OrientedBoundingBox::new(center, rotation * DMat3::from_diagonal(scale));
        // from the positive x direction
        let point = obb.half_axes.x_axis * 2.0 + center;
        let d = point.distance(center) - scale.x;
        assert!((obb.distance_squared_to(point) - d * d).abs() < 1e-10);
    }

    #[test]
    fn culling_volume_is_outside_as_soon_as_one_plane_rejects() {
        let unit_box = AxisAlignedBoundingBox::new(DVec3::splat(-1.0), DVec3::splat(1.0));
        let volume = CullingVolume::new(vec![Plane::new(DVec3::X, 5.0), Plane::new(-DVec3::X, 5.0)]);
        assert_eq!(volume.compute_visibility(&unit_box), Intersection::Inside);
        let volume = CullingVolume::new(vec![Plane::new(DVec3::X, 0.5), Plane::new(DVec3::X, -5.0)]);
        assert_eq!(volume.compute_visibility(&unit_box), Intersection::Outside);
        let volume = CullingVolume::new(vec![Plane::new(DVec3::X, 0.5)]);
        assert_eq!(volume.compute_visibility(&unit_box), Intersection::Intersecting);
    }
}
