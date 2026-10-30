//! Port of `@deck.gl/extensions/src/terrain`'s `TerrainEffect` and shader module: layers
//! whose operation is `terrain` draw their ground elevation into a map covering the view, and
//! layers with the terrain extension either read it to sit on that ground (offset) or are
//! drawn into a cover texture per terrain layer, which the terrain layer then shows (drape).
//!
//! Maps and covers are laid out in flat (Mercator) common space, also under a globe.

use std::collections::HashMap;
use std::sync::Arc;

use glam::{DVec2, DVec3, Vec4};
use luma_gl::device::create_render_texture;
use luma_gl::{Model, RenderTarget, ShaderField, ShaderInjection, ShaderModuleSource, COVER_FORMAT};
use math_gl::web_mercator::lng_lat_to_world;

use crate::constants::ProjectionMode;
use crate::extension::ExtensionShaders;
use crate::layer::{LayerContext, LAYER_INDEX_STRIDE};
use crate::viewport::{Viewport, WebMercatorViewportOptions};
use crate::Result;

/// The `terrain` shader module.
pub const TERRAIN: ShaderModuleSource = ShaderModuleSource {
    name: "terrain",
    source: include_str!("shaderlib/wgsl/terrain.wgsl"),
};

/// Side of the height map in pixels.
pub const HEIGHT_MAP_SIZE: u32 = 1024;

/// Format of the height map: metres in the red channel, filterable so the ground reads
/// smoothly between texels.
pub const HEIGHT_MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Attachments of the height map pass.
pub const HEIGHT_MAP_TARGET: RenderTarget = RenderTarget {
    color_format: HEIGHT_MAP_FORMAT,
    depth_format: None,
    sample_count: 1,
};

/// Attachments of a terrain cover pass.
pub const COVER_TARGET: RenderTarget = RenderTarget {
    color_format: COVER_FORMAT,
    depth_format: None,
    sample_count: 1,
};

/// The largest side of a terrain cover, deck.gl's `MAX_VIEWPORT_SIZE`.
pub const MAX_COVER_SIZE: u32 = 2048;

/// What every layer needs to take part: the module and the hooks it injects.
pub fn terrain_shaders() -> ExtensionShaders {
    ExtensionShaders {
        modules: vec![TERRAIN],
        injections: vec![
            ShaderInjection::new(
                "vs:DECKGL_FILTER_GL_POSITION",
                "position = terrain_setVertexPosition(position);",
            ),
            ShaderInjection::new("fs:DECKGL_FILTER_COLOR", "color = terrain_filterColor(color);"),
        ],
        attributes: Vec::new(),
        varyings: vec![
            ShaderField {
                name: "terrain_height",
                ty: "f32",
            },
            ShaderField {
                name: "terrain_flatPosition",
                ty: "vec2<f32>",
            },
        ],
    }
}

/// deck.gl's `terrainDrawMode`: how a layer with the terrain extension meets the ground.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainDrawMode {
    /// Every object moves up by the ground's height under its anchor (icons, columns, 3D)
    Offset,
    /// The layer is drawn into the terrain covers and the terrain shows it (paths, polygons)
    Drape,
}

/// What the draped layers rendered for one terrain layer: a texture over part of the world.
#[derive(Clone, Debug)]
pub struct TerrainCover {
    pub view: wgpu::TextureView,
    /// Origin and size of the cover in absolute flat common space: `[x, y, width, height]`
    pub bounds: [f32; 4],
}

/// The ground of the current frame: a map of elevations in metres over a part of the world.
#[derive(Clone, Debug)]
pub struct TerrainMap {
    /// The map, once it has been drawn
    pub view: Option<wgpu::TextureView>,
    /// Bound where there is none
    pub dummy: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    /// Origin and size of the map in absolute flat common space: `[x, y, width, height]`
    pub bounds: [f32; 4],
    /// The cover of each terrain layer that draws, by its index in the deck's layer list
    pub covers: HashMap<usize, TerrainCover>,
}

impl TerrainMap {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let dummy = luma_gl::create_rgba8_texture(device, queue, "terrain fallback", 1, 1, &[0; 4]);
        Self {
            view: None,
            dummy: dummy.create_view(&Default::default()),
            sampler: luma_gl::default_sampler(device),
            bounds: [0.0; 4],
            covers: HashMap::new(),
        }
    }

    /// The cover of the top level layer `layer_index` belongs to.
    pub fn cover(&self, layer_index: u32) -> Option<&TerrainCover> {
        self.covers.get(&((layer_index / LAYER_INDEX_STRIDE) as usize))
    }

    /// The map to bind, the dummy when there is none.
    pub fn texture(&self) -> &wgpu::TextureView {
        self.view.as_ref().unwrap_or(&self.dummy)
    }
}

/// A height map texture of the standard size.
pub fn create_height_map(device: &wgpu::Device) -> wgpu::Texture {
    create_render_texture(
        device,
        "terrain height map",
        HEIGHT_MAP_SIZE,
        HEIGHT_MAP_SIZE,
        HEIGHT_MAP_FORMAT,
    )
}

/// The part of the world in view, `[min x, min y, max x, max y]` in absolute flat common
/// space. Under a globe, where the view's corners may miss the sphere, a grid of pixels is
/// unprojected onto it (pixels past the limb land on the limb).
pub fn visible_flat_bounds(viewport: &Viewport) -> [f64; 4] {
    let mut points = Vec::new();
    if viewport.projection_mode() == ProjectionMode::Globe {
        const STEPS: u32 = 8;
        for i in 0..=STEPS {
            for j in 0..=STEPS {
                let pixel = DVec2::new(
                    viewport.width * i as f64 / STEPS as f64,
                    viewport.height * j as f64 / STEPS as f64,
                );
                let lng_lat = viewport.unproject(pixel, None, true, Some(0.0));
                points.push(lng_lat_to_world([lng_lat.x, lng_lat.y]));
            }
        }
    } else {
        let bounds = viewport.get_bounds(0.0);
        for (x, y) in [(bounds[0], bounds[1]), (bounds[2], bounds[3])] {
            let p = viewport.project_position(DVec3::new(x, y, 0.0));
            points.push([p.x, p.y]);
        }
    }
    let mut bounds = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for [x, y] in points
        .into_iter()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
    {
        bounds = [
            bounds[0].min(x),
            bounds[1].min(y),
            bounds[2].max(x),
            bounds[3].max(y),
        ];
    }
    bounds
}

/// The part of the world the map covers, in flat common space: the view's bounds with a
/// margin, so geometry a little outside the view still finds ground under it.
pub fn height_map_bounds(viewport: &Viewport) -> [f32; 4] {
    let [min_x, min_y, max_x, max_y] = visible_flat_bounds(viewport);
    let (width, height) = (
        (max_x - min_x).max(f64::EPSILON),
        (max_y - min_y).max(f64::EPSILON),
    );
    // An eighth of the view on every side
    let margin = 0.125;
    [
        (min_x - width * margin) as f32,
        (min_y - height * margin) as f32,
        (width * (1.0 + 2.0 * margin)) as f32,
        (height * (1.0 + 2.0 * margin)) as f32,
    ]
}

/// The bounds of the cover of a terrain layer whose geometry spans `layer_bounds` (absolute
/// flat common space, `None` when unknown), deck.gl's `TerrainCover._updateViewport`: the
/// layer, cut down to twice the view where it is larger. Under a globe deck.gl renders the
/// whole layer; here the visible part of the globe stands in for the view, so tiled terrain
/// spanning the world still gets a cover of useful resolution.
pub fn cover_bounds(layer_bounds: Option<[f64; 4]>, viewport: &Viewport) -> [f64; 4] {
    let view = visible_flat_bounds(viewport);
    if viewport.projection_mode() != ProjectionMode::Globe {
        return crate::mask::render_bounds(layer_bounds, view);
    }
    match layer_bounds {
        Some(b) => [
            b[0].max(view[0]),
            b[1].max(view[1]),
            b[2].min(view[2]),
            b[3].min(view[3]),
        ],
        None => view,
    }
}

/// deck.gl's `makeViewport` given a zoom: an unpitched Web Mercator viewport centred on
/// `bounds` (absolute flat common space) whose pixels are `2^zoom` per common unit, shrunk to
/// at most [`MAX_COVER_SIZE`] on a side, and the flat bounds its texture covers. `None` for
/// empty bounds.
pub fn cover_viewport(bounds: [f64; 4], zoom: f64, reference: &Viewport) -> Option<(Viewport, [f64; 4])> {
    let [x0, y0, x1, y1] = bounds;
    if x1 <= x0 || y1 <= y0 || !bounds.iter().all(|v| v.is_finite()) {
        return None;
    }
    let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let mut zoom = zoom;
    let scale = 2f64.powf(zoom);
    let (mut width, mut height) = (((x1 - x0) * scale).round(), ((y1 - y0) * scale).round());
    let max_size = MAX_COVER_SIZE as f64;
    if width > max_size || height > max_size {
        let r = max_size / width.max(height);
        width = (width * r).round();
        height = (height * r).round();
        zoom += r.log2();
    }
    let (width, height) = (width.max(1.0), height.max(1.0));
    // Unprojected through Mercator: a globe would give sphere coordinates
    let [longitude, latitude] = math_gl::web_mercator::world_to_lng_lat([cx, cy]);
    let viewport = Viewport::web_mercator(&WebMercatorViewportOptions {
        id: format!("{}-terrain-cover", reference.id),
        width,
        height,
        longitude,
        latitude,
        zoom,
        ..Default::default()
    });
    let scale = 2f64.powf(zoom);
    let (half_w, half_h) = (width / scale / 2.0, height / scale / 2.0);
    Some((viewport, [cx - half_w, cy - half_h, cx + half_w, cy + half_h]))
}

/// What a model does with the terrain module this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainMode {
    /// Draw as usual
    None,
    /// Draw the ground elevation into the map
    WriteHeightMap,
    /// Sit on the ground the map holds
    UseHeightMap,
    /// A terrain layer on screen: its own colour under the cover the draped layers drew
    UseCover,
    /// A terrain layer showing only its cover
    UseCoverOnly,
    /// Not drawn: a draped layer on screen, or an offset layer in a cover
    Skip,
}

impl TerrainMode {
    fn shader_value(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::WriteHeightMap => 1.0,
            Self::UseHeightMap => 2.0,
            Self::UseCover => 3.0,
            Self::UseCoverOnly => 4.0,
            Self::Skip => 5.0,
        }
    }
}

/// Give a model the terrain uniforms for `mode`. Models without the module are left alone,
/// so this is safe to call for every layer.
pub fn write_terrain_uniforms(model: &mut Model, ctx: &LayerContext, mode: TerrainMode) -> Result<()> {
    if !model.has_uniforms("terrain") {
        model.set_terrain_mode(false);
        model.set_cover_mode(false);
        return Ok(());
    }
    model.set_terrain_mode(mode == TerrainMode::WriteHeightMap);
    model.set_cover_mode(ctx.terrain_cover_pass);
    let terrain = ctx.terrain.clone();
    let cover = terrain.as_ref().and_then(|map| map.cover(ctx.layer_index));
    let (bounds, has_map) = match (&terrain, mode) {
        (Some(_), TerrainMode::UseCover | TerrainMode::UseCoverOnly) => {
            (cover.map_or([0.0; 4], |c| c.bounds), cover.is_some())
        }
        (Some(map), _) => (map.bounds, map.view.is_some()),
        (None, _) => ([0.0; 4], false),
    };
    // Without a map there is no ground to sit on, and without a cover nothing to show
    let mode = match mode {
        TerrainMode::UseHeightMap | TerrainMode::UseCover if !has_map => TerrainMode::None,
        TerrainMode::UseCoverOnly if !has_map => TerrainMode::Skip,
        other => other,
    };
    if let Some(map) = &terrain {
        let texture = match (mode, cover) {
            (TerrainMode::UseCover | TerrainMode::UseCoverOnly, Some(cover)) => &cover.view,
            _ => map.texture(),
        };
        model.set_texture("terrain_map", texture.clone())?;
        model.set_sampler("terrain_mapSampler", map.sampler.clone())?;
    }
    let block = model.uniforms("terrain")?;
    block.set_vec4("bounds", Vec4::from(bounds))?;
    block.set_f32("mode", mode.shader_value())?;
    Ok(())
}

/// The mode a layer's own models draw with, before its extensions have their say: terrain
/// layers write the height map in its pass and show their cover on screen.
pub fn layer_terrain_mode(ctx: &LayerContext, is_terrain_layer: bool) -> TerrainMode {
    match (ctx.terrain_pass, ctx.terrain_cover_pass, is_terrain_layer) {
        (true, _, true) => TerrainMode::WriteHeightMap,
        (false, false, true) => TerrainMode::UseCover,
        _ => TerrainMode::None,
    }
}

/// An empty map, for decks without terrain.
pub fn empty_map(device: &wgpu::Device, queue: &wgpu::Queue) -> Arc<TerrainMap> {
    Arc::new(TerrainMap::new(device, queue))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewport::WebMercatorViewportOptions;

    #[test]
    fn the_map_covers_the_view_with_a_margin() {
        let viewport = Viewport::web_mercator(&WebMercatorViewportOptions {
            width: 800.0,
            height: 600.0,
            longitude: -122.4,
            latitude: 37.8,
            zoom: 12.0,
            ..Default::default()
        });
        let [x, y, width, height] = height_map_bounds(&viewport);
        // The viewport centre falls inside the map
        let centre = viewport.center;
        assert!(
            (centre.x as f32) > x && (centre.x as f32) < x + width,
            "centre {} in {x}..{}",
            centre.x,
            x + width
        );
        assert!((centre.y as f32) > y && (centre.y as f32) < y + height);
        // And the map is wider than the view itself
        let bounds = viewport.get_bounds(0.0);
        let bl = viewport.project_position(glam::DVec3::new(bounds[0], bounds[1], 0.0));
        let tr = viewport.project_position(glam::DVec3::new(bounds[2], bounds[3], 0.0));
        assert!(width as f64 > (tr.x - bl.x).abs());
        assert!(height as f64 > (tr.y - bl.y).abs());
    }

    fn globe(zoom: f64) -> Viewport {
        Viewport::globe(&crate::viewport::GlobeViewportOptions {
            width: 800.0,
            height: 600.0,
            longitude: -122.4,
            latitude: 37.8,
            zoom,
            ..Default::default()
        })
    }

    #[test]
    fn the_visible_part_of_a_globe_is_found_in_flat_space() {
        let [x0, y0, x1, y1] = visible_flat_bounds(&globe(4.0));
        let [cx, cy] = lng_lat_to_world([-122.4, 37.8]);
        assert!(x0 < cx && cx < x1 && y0 < cy && cy < y1, "{x0} {y0} {x1} {y1}");
        // 800 pixels at zoom 4 are about 50 world units, curvature adding a little; not the world
        assert!(x1 - x0 < 128.0 && y1 - y0 < 128.0, "{x0} {y0} {x1} {y1}");
        // Zoomed out, pixels past the limb land on it, so the bounds stay finite
        let whole = visible_flat_bounds(&globe(0.0));
        assert!(whole.iter().all(|v| v.is_finite()), "{whole:?}");
    }

    #[test]
    fn a_cover_fits_its_bounds_at_the_zoom_asked_for() {
        let reference = globe(10.0);
        let bounds = [100.0, 200.0, 100.25, 200.125];
        let (viewport, covered) = cover_viewport(bounds, 11.0, &reference).unwrap();
        // 2^11 pixels per common unit
        assert_eq!((viewport.width, viewport.height), (512.0, 256.0));
        assert_eq!(viewport.projection_mode(), ProjectionMode::WebMercator);
        for i in 0..4 {
            assert!((covered[i] - bounds[i]).abs() < 1e-9, "{covered:?}");
        }
        // The viewport's centre is the bounds' centre
        let centre = viewport.project_position(DVec3::new(viewport.longitude, viewport.latitude, 0.0));
        assert!((centre.x - 100.125).abs() < 1e-6 && (centre.y - 200.0625).abs() < 1e-6);
        // Too large for one texture, it shrinks to the largest side and keeps its aspect
        let (viewport, covered) = cover_viewport(bounds, 14.0, &reference).unwrap();
        assert_eq!((viewport.width, viewport.height), (2048.0, 1024.0));
        assert!((viewport.zoom - 13.0).abs() < 1e-9);
        assert!((covered[2] - covered[0] - 0.25).abs() < 1e-9);
        assert!(cover_viewport([1.0, 1.0, 1.0, 2.0], 10.0, &reference).is_none());
    }

    #[test]
    fn a_cover_is_cut_to_twice_the_view_on_a_map_and_to_the_globe_in_sight() {
        let map = Viewport::web_mercator(&WebMercatorViewportOptions {
            width: 800.0,
            height: 600.0,
            longitude: -122.4,
            latitude: 37.8,
            zoom: 12.0,
            ..Default::default()
        });
        let view = visible_flat_bounds(&map);
        let world = Some([0.0, 0.0, 512.0, 512.0]);
        let cut = cover_bounds(world, &map);
        assert!(
            (cut[2] - cut[0] - 2.0 * (view[2] - view[0])).abs() < 1e-9,
            "{cut:?} {view:?}"
        );
        // A small layer is covered whole
        let small = [view[0] + 0.001, view[1] + 0.001, view[0] + 0.002, view[1] + 0.002];
        assert_eq!(cover_bounds(Some(small), &map), small);
        let globe = globe(4.0);
        assert_eq!(cover_bounds(world, &globe), visible_flat_bounds(&globe));
        assert_eq!(cover_bounds(None, &globe), visible_flat_bounds(&globe));
    }
}
