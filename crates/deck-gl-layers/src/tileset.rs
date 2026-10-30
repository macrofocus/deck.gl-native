//! Port of `@deck.gl/geo-layers/src/tileset-2d`: which tiles a viewport needs (a quadtree walk
//! with frustum culling for maps, a grid for cartesian views), tile bounds, URL templates, a
//! cache with parent and child links, and the "best available" refinement that shows loaded
//! ancestors or children while a tile loads.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use deck_gl::glam::{DMat4, DVec3, DVec4};
use deck_gl::{ProjectionMode, Viewport};
use math_gl::culling::{
    make_oriented_bounding_box_from_points, AxisAlignedBoundingBox, BoundingVolume, CullingVolume,
    Intersection, OrientedBoundingBox, Plane,
};
use math_gl::web_mercator::lng_lat_to_world;

pub const TILE_SIZE: f64 = 512.0;

/// OSM style tile index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileIndex {
    pub x: i64,
    pub y: i64,
    pub z: u32,
}

impl TileIndex {
    pub fn new(x: i64, y: i64, z: u32) -> Self {
        Self { x, y, z }
    }

    /// The tile one level up that contains this one.
    pub fn parent(&self) -> Option<TileIndex> {
        (self.z > 0).then(|| TileIndex::new(self.x.div_euclid(2), self.y.div_euclid(2), self.z - 1))
    }

    pub fn children(&self) -> [TileIndex; 4] {
        let (x, y, z) = (self.x * 2, self.y * 2, self.z + 1);
        [
            TileIndex::new(x, y, z),
            TileIndex::new(x + 1, y, z),
            TileIndex::new(x, y + 1, z),
            TileIndex::new(x + 1, y + 1, z),
        ]
    }

    /// `"x-y-z"`, the id deck.gl gives a tile.
    pub fn id(&self) -> String {
        format!("{}-{}-{}", self.x, self.y, self.z)
    }
}

/// The area a tile covers: longitude and latitude for maps, world units otherwise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TileBounds {
    Geo {
        west: f64,
        north: f64,
        east: f64,
        south: f64,
    },
    Cartesian {
        left: f64,
        top: f64,
        right: f64,
        bottom: f64,
    },
}

impl TileBounds {
    /// `[min_x, min_y, max_x, max_y]` (west, south, east, north for geo bounds).
    pub fn as_array(&self) -> [f64; 4] {
        match *self {
            TileBounds::Geo {
                west,
                north,
                east,
                south,
            } => [west, south, east, north],
            TileBounds::Cartesian {
                left,
                top,
                right,
                bottom,
            } => [left, top.min(bottom), right, top.max(bottom)],
        }
    }
}

/// How selected tiles that are not loaded yet are covered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RefinementStrategy {
    /// Show the nearest loaded ancestor, else loaded descendants
    #[default]
    BestAvailable,
    /// Only draw a tile once every selected tile overlapping it is loaded
    NoOverlap,
    /// Draw selected tiles as soon as they load, never placeholders
    Never,
}

impl RefinementStrategy {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "best-available" => Some(Self::BestAvailable),
            "no-overlap" => Some(Self::NoOverlap),
            "never" => Some(Self::Never),
            _ => None,
        }
    }
}

/// Options of [`get_tile_indices`] and [`Tileset`].
#[derive(Clone, Debug, PartialEq)]
pub struct TilesetOptions {
    /// Pixel size of a tile at the zoom it is drawn; 512 matches the viewport zoom
    pub tile_size: f64,
    pub min_zoom: Option<u32>,
    pub max_zoom: Option<u32>,
    /// Added to the viewport zoom before choosing the tile zoom
    pub zoom_offset: f64,
    /// Bounds of the data, `[min_x, min_y, max_x, max_y]` (west, south, east, north)
    pub extent: Option<[f64; 4]>,
    /// Elevation range of the tiles in meters, for culling pitched views
    pub z_range: Option<[f64; 2]>,
    pub refinement_strategy: RefinementStrategy,
    /// Tiles kept once no longer selected; five times the selection when unset
    pub max_cache_size: Option<usize>,
}

impl Default for TilesetOptions {
    fn default() -> Self {
        Self {
            tile_size: TILE_SIZE,
            min_zoom: Some(0),
            max_zoom: None,
            zoom_offset: 0.0,
            extent: None,
            z_range: None,
            refinement_strategy: RefinementStrategy::BestAvailable,
            max_cache_size: None,
        }
    }
}

/// Sample points of a tile, in tile units, for its bounding volume in a projection that is not
/// Web Mercator: deck.gl's `REF_POINTS_5`, the four corners and the center.
const REF_POINTS_5: [[f64; 2]; 5] = [[0.5, 0.5], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
/// `REF_POINTS_9`: the corners, the center and the four mid points.
const REF_POINTS_9: [[f64; 2]; 9] = [
    [0.5, 0.5],
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 0.5],
    [0.5, 0.0],
    [1.0, 0.5],
    [0.5, 1.0],
];
/// `REF_POINTS_11`: two more points on the equator, for the top tile.
const REF_POINTS_11: [[f64; 2]; 11] = [
    [0.5, 0.5],
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 0.5],
    [0.5, 0.0],
    [1.0, 0.5],
    [0.5, 1.0],
    [0.25, 0.5],
    [0.75, 0.5],
];

/// The six planes of a viewport's frustum in common space, normals pointing inside.
fn frustum_planes(view_projection: &DMat4) -> CullingVolume {
    let m = view_projection;
    let row = |i: usize| DVec4::new(m.x_axis[i], m.y_axis[i], m.z_axis[i], m.w_axis[i]);
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    let plane = |v: DVec4| {
        let n = v.truncate();
        let len = n.length().max(f64::EPSILON);
        Plane::new(n / len, v.w / len)
    };
    CullingVolume::new(vec![
        plane(r3 + r0),
        plane(r3 - r0),
        plane(r3 + r1),
        plane(r3 - r1),
        plane(r3 + r2),
        plane(r3 - r2),
    ])
}

/// A tile's bounding volume: a box in Web Mercator common space, or one fitted to sample points
/// on the globe.
enum TileVolume {
    Aligned(AxisAlignedBoundingBox),
    Oriented(OrientedBoundingBox),
}

impl TileVolume {
    fn visibility(&self, culling: &CullingVolume) -> Intersection {
        match self {
            Self::Aligned(b) => culling.compute_visibility(b),
            Self::Oriented(b) => culling.compute_visibility(b),
        }
    }

    fn distance_to(&self, point: DVec3) -> f64 {
        match self {
            Self::Aligned(b) => b.distance_to(point),
            Self::Oriented(b) => b.distance_to(point),
        }
    }
}

struct Traversal<'a> {
    viewport: &'a Viewport,
    /// Whether the viewport is a globe, whose common space is not Web Mercator
    globe: bool,
    culling: CullingVolume,
    elevation: [f64; 2],
    min_z: u32,
    max_z: u32,
    /// Extent in OSM world units (y down)
    bounds: Option<[f64; 4]>,
    world_offset: f64,
}

/// deck.gl's `OSMNode.getBoundingVolume`.
fn bounding_volume(node: TileIndex, t: &Traversal<'_>) -> TileVolume {
    if t.globe {
        // At low zoom levels more samples are needed to bound the tile correctly
        let ref_points: &[[f64; 2]] = match node.z {
            0 => &REF_POINTS_11,
            1 => &REF_POINTS_9,
            _ => &REF_POINTS_5,
        };
        let [z_min, z_max] = t.elevation;
        let mut positions = Vec::with_capacity(ref_points.len() * 2);
        for [dx, dy] in ref_points {
            // deck.gl passes the common space elevation range here as if it were meters
            let [lng, lat] = osm_tile_to_lng_lat_f(node.x as f64 + dx, node.y as f64 + dy, node.z);
            positions.push(t.viewport.project_position(DVec3::new(lng, lat, z_min)));
            if z_min != z_max {
                // Account for the elevation volume
                positions.push(t.viewport.project_position(DVec3::new(lng, lat, z_max)));
            }
        }
        return TileVolume::Oriented(make_oriented_bounding_box_from_points(&positions));
    }
    let extent = TILE_SIZE / 2f64.powi(node.z as i32);
    let origin_x = node.x as f64 * extent + t.world_offset * TILE_SIZE;
    // deck.gl's common space is y up while OSM tiles count rows from the top
    let origin_y = TILE_SIZE - (node.y as f64 + 1.0) * extent;
    TileVolume::Aligned(AxisAlignedBoundingBox::new(
        DVec3::new(origin_x, origin_y, t.elevation[0]),
        DVec3::new(origin_x + extent, origin_y + extent, t.elevation[1]),
    ))
}

/// deck.gl's `OSMNode.beyondHorizon`: on the globe, whether the point of the tile closest to
/// the camera faces away from it, hidden by the near hemisphere.
fn beyond_horizon(node: TileIndex, t: &Traversal<'_>, elevation: f64) -> bool {
    let c = t.viewport.camera_position;
    let camera_lng = c.x.atan2(-c.y).to_degrees();
    let camera_lat = (c.z / c.length()).asin().to_degrees();
    let [west, north] = osm_tile_to_lng_lat(node.x, node.y, node.z);
    let [east, south] = osm_tile_to_lng_lat(node.x + 1, node.y + 1, node.z);
    let tile_center_lng = (west + east) / 2.0;
    // JavaScript's `%` keeps the sign of the dividend, as Rust's does
    let wrapped_camera_lng = tile_center_lng + ((camera_lng - tile_center_lng + 540.0) % 360.0 - 180.0);
    let closest_lng = wrapped_camera_lng.clamp(west, east);
    let closest_lat = camera_lat.clamp(south, north);
    let closest = t
        .viewport
        .project_position(DVec3::new(closest_lng, closest_lat, elevation));
    closest.dot(c) <= closest.length_squared()
}

/// One node of the OSM quadtree walk, deck.gl's `OSMNode.update`.
fn visit(node: TileIndex, t: &Traversal<'_>, out: &mut Vec<TileIndex>) -> bool {
    if let Some([min_x, min_y, max_x, max_y]) = t.bounds {
        let extent = TILE_SIZE / 2f64.powi(node.z as i32);
        let (x0, y0) = (node.x as f64 * extent, node.y as f64 * extent);
        if x0 >= max_x || x0 + extent <= min_x || y0 >= max_y || y0 + extent <= min_y {
            return false;
        }
    }
    let volume = bounding_volume(node, t);
    if volume.visibility(&t.culling) == Intersection::Outside {
        return false;
    }
    // Globe: reject tiles occluded by the near hemisphere
    if t.globe && beyond_horizon(node, t, t.elevation[1]) {
        return false;
    }
    let mut z = node.z;
    if z < t.max_z && z >= t.min_z {
        // Level of detail from the distance to the camera, in viewport heights
        let distance = volume.distance_to(t.viewport.camera_position) * t.viewport.scale / t.viewport.height;
        if distance > 1.0 {
            z += distance.log2().floor() as u32;
        }
    }
    if z >= t.max_z {
        out.push(node);
        return true;
    }
    for child in node.children() {
        visit(child, t, out);
    }
    true
}

/// deck.gl's `getOSMTileIndices`.
fn osm_tile_indices(viewport: &Viewport, z: u32, opts: &TilesetOptions) -> Vec<TileIndex> {
    let globe = viewport.projection_mode() == ProjectionMode::Globe;
    let units_per_meter = viewport.distance_scales.units_per_meter.z;
    let elevation = opts
        .z_range
        .map(|[min, max]| [min * units_per_meter, max * units_per_meter])
        .unwrap_or([0.0, 0.0]);
    // Always load the current zoom on a map unless it is pitched far, then coarser far away
    let min_z = if !globe && viewport.pitch <= 60.0 { z } else { 0 };
    let bounds = opts.extent.map(|[min_lng, min_lat, max_lng, max_lat]| {
        let top_left = lng_lat_to_world([min_lng, max_lat]);
        let bottom_right = lng_lat_to_world([max_lng, min_lat]);
        [
            top_left[0],
            TILE_SIZE - top_left[1],
            bottom_right[0],
            TILE_SIZE - bottom_right[1],
        ]
    });
    let mut result = Vec::new();
    let mut traversal = Traversal {
        viewport,
        globe,
        culling: frustum_planes(&viewport.view_projection_matrix),
        elevation,
        min_z,
        max_z: z,
        bounds,
        world_offset: 0.0,
    };
    visit(TileIndex::new(0, 0, 0), &traversal, &mut result);
    // Repeated world copies across the antimeridian, on a map only
    let view_bounds = viewport.get_bounds(0.0);
    if !globe && (view_bounds[0] < -180.0 || view_bounds[2] > 180.0) {
        let min_offset = ((view_bounds[0] + 180.0) / 360.0).floor().max(-3.0) as i32;
        let max_offset = ((view_bounds[2] - 180.0) / 360.0).ceil().min(3.0) as i32;
        for offset in min_offset..=max_offset {
            if offset != 0 {
                traversal.world_offset = offset as f64;
                visit(TileIndex::new(0, 0, 0), &traversal, &mut result);
            }
        }
    }
    result.sort();
    result.dedup();
    result
}

fn identity_tile_indices(viewport: &Viewport, z: u32, opts: &TilesetOptions) -> Vec<TileIndex> {
    let extent = opts
        .extent
        .unwrap_or([f64::NEG_INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY]);
    let b = viewport.get_bounds(0.0);
    let bbox = [
        b[0].min(extent[2]).max(extent[0]),
        b[1].min(extent[3]).max(extent[1]),
        b[2].max(extent[0]).min(extent[2]),
        b[3].max(extent[1]).min(extent[3]),
    ];
    let scale = 2f64.powi(z as i32) * TILE_SIZE / opts.tile_size;
    let to_index = |v: f64| v * scale / TILE_SIZE;
    let (min_x, min_y, max_x, max_y) = (
        to_index(bbox[0]),
        to_index(bbox[1]),
        to_index(bbox[2]),
        to_index(bbox[3]),
    );
    let mut indices = Vec::new();
    if !(min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite()) {
        return indices;
    }
    let mut x = min_x.floor() as i64;
    while (x as f64) < max_x {
        let mut y = min_y.floor() as i64;
        while (y as f64) < max_y {
            indices.push(TileIndex::new(x, y, z));
            y += 1;
        }
        x += 1;
    }
    indices
}

/// The tiles a viewport needs, port of deck.gl's `getTileIndices`: at the rounded viewport
/// zoom for maps (like maplibre and Google Maps), clamped to `min_zoom` and `max_zoom`.
pub fn get_tile_indices(viewport: &Viewport, opts: &TilesetOptions) -> Vec<TileIndex> {
    let raw = if viewport.is_geospatial {
        (viewport.zoom + (TILE_SIZE / opts.tile_size).log2() + opts.zoom_offset).round()
    } else {
        (viewport.zoom + opts.zoom_offset).ceil()
    };
    let mut z = raw.max(0.0) as u32;
    if let Some(min) = opts.min_zoom {
        if z < min {
            if opts.extent.is_none() {
                return Vec::new();
            }
            z = min;
        }
    }
    if let Some(max) = opts.max_zoom {
        z = z.min(max);
    }
    if viewport.is_geospatial {
        osm_tile_indices(viewport, z, opts)
    } else {
        identity_tile_indices(viewport, z, opts)
    }
}

/// Longitude and latitude of the top left corner of an OSM tile.
pub fn osm_tile_to_lng_lat(x: i64, y: i64, z: u32) -> [f64; 2] {
    osm_tile_to_lng_lat_f(x as f64, y as f64, z)
}

/// deck.gl's `osmTile2lngLat` at a fractional tile position.
fn osm_tile_to_lng_lat_f(x: f64, y: f64, z: u32) -> [f64; 2] {
    let scale = 2f64.powi(z as i32);
    let lng = x / scale * 360.0 - 180.0;
    let n = std::f64::consts::PI - 2.0 * std::f64::consts::PI * y / scale;
    let lat = 180.0 / std::f64::consts::PI * (0.5 * (n.exp() - (-n).exp())).atan();
    [lng, lat]
}

/// The bounds of a tile for a viewport, port of `tileToBoundingBox`.
pub fn tile_bounds(viewport: &Viewport, index: TileIndex, tile_size: f64) -> TileBounds {
    if viewport.is_geospatial {
        let [west, north] = osm_tile_to_lng_lat(index.x, index.y, index.z);
        let [east, south] = osm_tile_to_lng_lat(index.x + 1, index.y + 1, index.z);
        TileBounds::Geo {
            west,
            north,
            east,
            south,
        }
    } else {
        let scale = 2f64.powi(index.z as i32) * TILE_SIZE / tile_size;
        let to = |v: i64| v as f64 / scale * TILE_SIZE;
        TileBounds::Cartesian {
            left: to(index.x),
            top: to(index.y),
            right: to(index.x + 1),
            bottom: to(index.y + 1),
        }
    }
}

/// Fill `{x}`, `{y}`, `{z}` and `{-y}` in a URL template; a list of templates is chosen by a
/// hash of the tile id, like deck.gl's `getURLFromTemplate`.
pub fn url_from_template(templates: &[String], index: TileIndex) -> Option<String> {
    if templates.is_empty() {
        return None;
    }
    let id = index.id();
    let hash = id
        .bytes()
        .fold(0i32, |a, b| {
            a.wrapping_shl(5).wrapping_sub(a).wrapping_add(b as i32)
        })
        .unsigned_abs() as usize;
    let template = &templates[hash % templates.len()];
    Some(
        template
            .replace("{x}", &index.x.to_string())
            .replace("{y}", &index.y.to_string())
            .replace("{z}", &index.z.to_string())
            .replace("{-y}", &((1i64 << index.z) - index.y - 1).to_string()),
    )
}

/// Whether a string is a tile URL template (`{z}`, `{x}` and `{y}` or `{-y}`).
pub fn is_url_template(s: &str) -> bool {
    s.contains("{z}") && s.contains("{x}") && (s.contains("{y}") || s.contains("{-y}"))
}

/// Fetch a URL's bytes through the shared [`Fetcher`](crate::fetch::Fetcher): cached
/// responses come back at once and one request serves every layer asking for the URL. Needs
/// the `fetch` feature to reach the network.
pub fn fetch_bytes(url: &str) -> std::result::Result<Arc<Vec<u8>>, String> {
    crate::fetch::Fetcher::global().fetch_blocking(url)
}

/// Like [`fetch_bytes`], giving up when `cancel` is set.
pub fn fetch_bytes_with(
    url: &str,
    cancel: &crate::fetch::CancelToken,
) -> std::result::Result<Arc<Vec<u8>>, String> {
    crate::fetch::Fetcher::global().fetch_blocking_with(url, cancel)
}

/// Loading state of a cached tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileStatus {
    Pending,
    Loading,
    /// Loaded, with or without content
    Loaded,
    Failed,
}

/// A tile in the cache, deck.gl's `Tile2DHeader` without the content.
#[derive(Clone, Debug)]
pub struct TileHeader {
    pub index: TileIndex,
    pub bounds: TileBounds,
    pub status: TileStatus,
    pub is_selected: bool,
    pub is_visible: bool,
    /// Whether the loaded tile has content to draw
    pub has_content: bool,
    last_selected_frame: u64,
}

/// Which tiles are selected, loading and visible for the current viewport.
pub struct Tileset {
    pub opts: TilesetOptions,
    tiles: HashMap<TileIndex, TileHeader>,
    selected: Vec<TileIndex>,
    frame: u64,
}

impl Tileset {
    pub fn new(opts: TilesetOptions) -> Self {
        Self {
            opts,
            tiles: HashMap::new(),
            selected: Vec::new(),
            frame: 0,
        }
    }

    pub fn tiles(&self) -> impl Iterator<Item = &TileHeader> {
        self.tiles.values()
    }

    pub fn tile(&self, index: TileIndex) -> Option<&TileHeader> {
        self.tiles.get(&index)
    }

    pub fn selected(&self) -> &[TileIndex] {
        &self.selected
    }

    /// Selected tiles that have not started loading, in selection order.
    pub fn pending(&self) -> Vec<TileIndex> {
        self.selected
            .iter()
            .copied()
            .filter(|i| self.tiles.get(i).is_some_and(|t| t.status == TileStatus::Pending))
            .collect()
    }

    /// Visible tiles, drawn in order (coarser first).
    pub fn visible(&self) -> Vec<TileIndex> {
        let mut visible: Vec<TileIndex> = self
            .tiles
            .values()
            .filter(|t| t.is_visible && t.has_content)
            .map(|t| t.index)
            .collect();
        visible.sort();
        visible
    }

    /// Whether every selected tile has finished loading.
    pub fn is_loaded(&self) -> bool {
        self.selected.iter().all(|i| {
            self.tiles
                .get(i)
                .is_some_and(|t| matches!(t.status, TileStatus::Loaded | TileStatus::Failed))
        })
    }

    pub fn set_status(&mut self, index: TileIndex, status: TileStatus, has_content: bool) {
        if let Some(tile) = self.tiles.get_mut(&index) {
            tile.status = status;
            tile.has_content = has_content;
        }
    }

    /// Select the tiles for a viewport; returns true when the selection changed.
    pub fn update(&mut self, viewport: &Viewport) -> bool {
        self.frame += 1;
        let indices = get_tile_indices(viewport, &self.opts);
        let changed = indices != self.selected;
        for tile in self.tiles.values_mut() {
            tile.is_selected = false;
        }
        for &index in &indices {
            let bounds = tile_bounds(viewport, index, self.opts.tile_size);
            let tile = self.tiles.entry(index).or_insert(TileHeader {
                index,
                bounds,
                status: TileStatus::Pending,
                is_selected: false,
                is_visible: false,
                has_content: false,
                last_selected_frame: 0,
            });
            tile.is_selected = true;
            tile.last_selected_frame = self.frame;
        }
        self.selected = indices;
        self.update_visibility();
        self.prune();
        changed
    }

    /// Recompute visibility after loads finished.
    pub fn update_visibility(&mut self) {
        let loaded = |t: &TileHeader| t.status == TileStatus::Loaded && t.has_content;
        let mut visible: HashSet<TileIndex> = HashSet::new();
        match self.opts.refinement_strategy {
            RefinementStrategy::Never => {
                for tile in self.tiles.values() {
                    if tile.is_selected && loaded(tile) {
                        visible.insert(tile.index);
                    }
                }
            }
            RefinementStrategy::NoOverlap => {
                // Selected and loaded tiles show; a selected tile that is not loaded shows its
                // nearest loaded ancestor only when no sibling under that ancestor is shown
                let mut shown_ancestors = HashSet::new();
                for tile in self.tiles.values().filter(|t| t.is_selected) {
                    if loaded(tile) {
                        visible.insert(tile.index);
                    } else if let Some(ancestor) = self.loaded_ancestor(tile.index) {
                        shown_ancestors.insert(ancestor);
                    }
                }
                for ancestor in shown_ancestors {
                    let covered = visible.iter().any(|v| self.is_descendant(*v, ancestor));
                    if !covered {
                        visible.insert(ancestor);
                    }
                }
            }
            RefinementStrategy::BestAvailable => {
                for tile in self.tiles.values().filter(|t| t.is_selected) {
                    match self.loaded_ancestor(tile.index) {
                        Some(ancestor) => {
                            visible.insert(ancestor);
                        }
                        None => self.loaded_descendants(tile.index, &mut visible),
                    }
                }
            }
        }
        for tile in self.tiles.values_mut() {
            tile.is_visible = visible.contains(&tile.index);
        }
    }

    /// The tile itself when loaded, else its nearest loaded ancestor in the cache.
    fn loaded_ancestor(&self, mut index: TileIndex) -> Option<TileIndex> {
        loop {
            if self
                .tiles
                .get(&index)
                .is_some_and(|t| t.status == TileStatus::Loaded && t.has_content)
            {
                return Some(index);
            }
            index = index.parent()?;
        }
    }

    fn loaded_descendants(&self, index: TileIndex, out: &mut HashSet<TileIndex>) {
        for child in index.children() {
            match self.tiles.get(&child) {
                Some(t) if t.status == TileStatus::Loaded && t.has_content => {
                    out.insert(child);
                }
                Some(_) => self.loaded_descendants(child, out),
                None => {}
            }
        }
    }

    fn is_descendant(&self, mut index: TileIndex, ancestor: TileIndex) -> bool {
        while let Some(parent) = index.parent() {
            if parent == ancestor {
                return true;
            }
            index = parent;
        }
        false
    }

    /// Drop the least recently selected tiles beyond the cache size.
    fn prune(&mut self) {
        let max = self
            .opts
            .max_cache_size
            .unwrap_or(self.selected.len() * 5)
            .max(self.selected.len());
        if self.tiles.len() <= max {
            return;
        }
        let mut candidates: Vec<(u64, TileIndex)> = self
            .tiles
            .values()
            .filter(|t| !t.is_selected && !t.is_visible && t.status != TileStatus::Loading)
            .map(|t| (t.last_selected_frame, t.index))
            .collect();
        candidates.sort();
        let excess = self.tiles.len() - max;
        for (_, index) in candidates.into_iter().take(excess) {
            self.tiles.remove(&index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deck_gl::{ViewState, WebMercatorViewportOptions};

    fn map(zoom: f64, pitch: f64) -> Viewport {
        Viewport::web_mercator(&WebMercatorViewportOptions {
            width: 512.0,
            height: 512.0,
            longitude: 0.0,
            latitude: 0.0,
            zoom,
            pitch,
            ..Default::default()
        })
    }

    #[test]
    fn tile_zoom_rounds_like_maplibre_and_respects_limits() {
        let opts = TilesetOptions::default();
        let tiles = get_tile_indices(&map(0.0, 0.0), &opts);
        assert_eq!(tiles, [TileIndex::new(0, 0, 0)]);
        // A 512 px view at zoom 1 centred on the equator sees all four tiles
        let mut tiles = get_tile_indices(&map(1.0, 0.0), &opts);
        tiles.sort();
        assert_eq!(tiles.len(), 4, "{tiles:?}");
        assert!(tiles.iter().all(|t| t.z == 1));
        // Zoom 1.5 rounds up to 2
        assert!(get_tile_indices(&map(1.5, 0.0), &opts).iter().all(|t| t.z == 2));
        let capped = TilesetOptions {
            max_zoom: Some(1),
            ..Default::default()
        };
        assert!(get_tile_indices(&map(5.0, 0.0), &capped).iter().all(|t| t.z == 1));
        let floor = TilesetOptions {
            min_zoom: Some(3),
            ..Default::default()
        };
        assert!(
            get_tile_indices(&map(1.0, 0.0), &floor).is_empty(),
            "below minZoom without extent"
        );
        let extent = TilesetOptions {
            extent: Some([-10.0, -10.0, 10.0, 10.0]),
            ..Default::default()
        };
        let tiles = get_tile_indices(&map(3.0, 0.0), &extent);
        assert!(
            !tiles.is_empty()
                && tiles
                    .iter()
                    .all(|t| (3..=4).contains(&t.x) && (3..=4).contains(&t.y)),
            "{tiles:?}"
        );
        // Pitched views load coarser tiles far away
        let pitched = get_tile_indices(&map(6.0, 70.0), &opts);
        assert!(
            pitched.iter().any(|t| t.z < 6) && pitched.iter().any(|t| t.z == 6),
            "{pitched:?}"
        );
    }

    #[test]
    fn globe_tiles_follow_deck_gl() {
        // deck.gl's `getTileIndices` "globe" case: on the globe the tiles are culled against
        // boxes fitted to points on the sphere, the far side is dropped behind the horizon, and
        // tiles away from the camera come at a coarser zoom
        let viewport = deck_gl::View::Globe(deck_gl::GlobeViewProps::default()).make_viewport(
            &deck_gl::AnyViewState::Globe(ViewState {
                longitude: -6.0,
                latitude: 58.0,
                zoom: 2.0,
                pitch: 0.0,
                bearing: 0.0,
            }),
            400.0,
            400.0,
        );
        assert_eq!(viewport.projection_mode(), ProjectionMode::Globe);
        let mut ids: Vec<String> = get_tile_indices(&viewport, &TilesetOptions::default())
            .iter()
            .map(|t| format!("{},{},{}", t.x, t.y, t.z))
            .collect();
        ids.sort();
        assert_eq!(
            ids,
            ["0,1,1", "0,1,2", "1,0,2", "1,1,1", "1,1,2", "2,0,2", "2,1,2", "3,1,2"]
        );
    }

    #[test]
    fn globe_tiles_beyond_the_horizon_are_dropped() {
        let viewport = deck_gl::View::Globe(deck_gl::GlobeViewProps::default()).make_viewport(
            &deck_gl::AnyViewState::Globe(ViewState {
                longitude: 0.0,
                latitude: 0.0,
                zoom: 2.0,
                pitch: 0.0,
                bearing: 0.0,
            }),
            400.0,
            400.0,
        );
        let tiles = get_tile_indices(&viewport, &TilesetOptions::default());
        assert!(!tiles.is_empty());
        for tile in &tiles {
            let [west, _] = osm_tile_to_lng_lat(tile.x, tile.y, tile.z);
            let [east, _] = osm_tile_to_lng_lat(tile.x + 1, tile.y + 1, tile.z);
            assert!(
                east > -90.0 && west < 90.0,
                "tile {tile:?} ({west}..{east}) is on the far side of the globe"
            );
        }
    }

    #[test]
    fn tile_bounds_and_urls() {
        let v = map(0.0, 0.0);
        match tile_bounds(&v, TileIndex::new(0, 0, 1), TILE_SIZE) {
            TileBounds::Geo {
                west,
                north,
                east,
                south,
            } => {
                assert!((west + 180.0).abs() < 1e-9 && (east - 0.0).abs() < 1e-9);
                assert!((north - 85.0511).abs() < 1e-3 && south.abs() < 1e-9);
            }
            other => panic!("{other:?}"),
        }
        let url = url_from_template(
            &["https://a/{z}/{x}/{y}.png".to_string()],
            TileIndex::new(3, 1, 2),
        )
        .unwrap();
        assert_eq!(url, "https://a/2/3/1.png");
        let flipped = url_from_template(&["{z}/{x}/{-y}".to_string()], TileIndex::new(3, 1, 2)).unwrap();
        assert_eq!(flipped, "2/3/2");
        assert!(is_url_template("https://a/{z}/{x}/{y}.png") && !is_url_template("data.json"));
        assert_eq!(TileIndex::new(5, 3, 3).parent(), Some(TileIndex::new(2, 1, 2)));
    }

    #[test]
    fn best_available_shows_ancestors_while_loading() {
        let mut set = Tileset::new(TilesetOptions::default());
        set.update(&map(0.0, 0.0));
        assert_eq!(set.pending(), [TileIndex::new(0, 0, 0)]);
        set.set_status(TileIndex::new(0, 0, 0), TileStatus::Loaded, true);
        set.update_visibility();
        assert_eq!(set.visible(), [TileIndex::new(0, 0, 0)]);
        // Zoom in: the four children are selected, the root stays visible until they load
        set.update(&map(1.0, 0.0));
        assert_eq!(set.selected().len(), 4);
        assert_eq!(set.visible(), [TileIndex::new(0, 0, 0)]);
        assert!(!set.is_loaded());
        for index in set.pending() {
            set.set_status(index, TileStatus::Loaded, true);
        }
        set.update_visibility();
        assert_eq!(set.visible().len(), 4);
        assert!(set.is_loaded());
        // Zoom out again: the root is loaded, children stay cached
        set.update(&map(0.0, 0.0));
        assert_eq!(set.visible(), [TileIndex::new(0, 0, 0)]);
        assert_eq!(set.tiles().count(), 5);
        let _ = ViewState::default();
    }
}
