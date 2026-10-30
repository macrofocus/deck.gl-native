//! Port of `@deck.gl/extensions/src/terrain`: put a layer on the ground that the terrain
//! layers of the deck draw.
//!
//! A layer whose operation is [`Operation::TERRAIN`](deck_gl::layer::Operation::TERRAIN)
//! writes its elevation into a height map covering the view and is drawn with a cover
//! texture. A layer carrying this extension meets that ground in one of two ways, its
//! [`TerrainDrawMode`]: offset, where the anchor of each object is looked up in the map and
//! its geometry moves up by the ground's height there, or drape, where the layer is drawn
//! into the terrain layers' covers instead of onto the screen.

use std::any::Any;
use std::sync::Arc;

use deck_gl::luma_gl::Model;
use deck_gl::terrain::{write_terrain_uniforms, TerrainDrawMode, TerrainMode};
use deck_gl::{
    same_extension, ExtensionShaders, Extensions, LayerContext, LayerExtension, LayerProps, Result, Viewport,
};

/// deck.gl's `TerrainExtension`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TerrainExtension {
    /// deck.gl's `terrainDrawMode`. Without one it is deduced from the layer: offset for
    /// layers with an anchor per object (`instancePositions`) or extruded polygons, drape
    /// for the others.
    pub terrain_draw_mode: Option<TerrainDrawMode>,
}

impl TerrainExtension {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_draw_mode(terrain_draw_mode: TerrainDrawMode) -> Self {
        Self {
            terrain_draw_mode: Some(terrain_draw_mode),
        }
    }

    /// The draw mode of a model of the layer, deck.gl's `updateState`: `is3d || hasAnchor`
    /// is offset. Layers whose anchors do not show in their buffer names (extruded polygons,
    /// paths) settle the mode with [`resolve_terrain_draw_mode`].
    fn draw_mode(&self, model: &Model) -> TerrainDrawMode {
        self.terrain_draw_mode
            .unwrap_or(if model.has_vertex_slot("instancePositions") {
                TerrainDrawMode::Offset
            } else {
                TerrainDrawMode::Drape
            })
    }
}

/// Give the terrain extension among `extensions` the draw mode `mode` unless it has one,
/// for layers that know better than their buffer names: extruded polygons are 3D, and the
/// path layer's `instancePositions` are its vertices rather than anchors.
pub fn resolve_terrain_draw_mode(extensions: &Extensions, mode: TerrainDrawMode) -> Extensions {
    Extensions::new(
        extensions
            .iter()
            .map(
                |extension| match extension.as_any().downcast_ref::<TerrainExtension>() {
                    Some(terrain) if terrain.terrain_draw_mode.is_none() => {
                        Arc::new(TerrainExtension::with_draw_mode(mode)) as Arc<dyn LayerExtension>
                    }
                    _ => extension.clone(),
                },
            )
            .collect(),
    )
}

impl LayerExtension for TerrainExtension {
    fn name(&self) -> &'static str {
        "TerrainExtension"
    }

    fn needs_terrain(&self) -> bool {
        true
    }

    /// The terrain module is on every layer of a deck that has ground, so the extension adds
    /// no shader code of its own; it only picks the mode.
    fn shaders(&self) -> ExtensionShaders {
        ExtensionShaders::default()
    }

    fn update_uniforms(
        &self,
        model: &mut Model,
        ctx: &LayerContext,
        _viewport: &Viewport,
        _props: &LayerProps,
    ) -> Result<()> {
        let draw_mode = self.draw_mode(model);
        let mode = if ctx.terrain_pass {
            // While the height map itself is being drawn, this layer is not part of it
            TerrainMode::None
        } else if ctx.terrain_cover_pass {
            // Draped layers draw flat into the cover; offset layers are not in it
            match draw_mode {
                TerrainDrawMode::Drape => TerrainMode::None,
                TerrainDrawMode::Offset => TerrainMode::Skip,
            }
        } else {
            match draw_mode {
                TerrainDrawMode::Drape => TerrainMode::Skip,
                TerrainDrawMode::Offset => TerrainMode::UseHeightMap,
            }
        };
        write_terrain_uniforms(model, ctx, mode)
    }

    fn equals(&self, other: &dyn LayerExtension) -> bool {
        same_extension(self, other)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modes(extensions: &Extensions) -> Vec<Option<TerrainDrawMode>> {
        extensions
            .iter()
            .filter_map(|e| e.as_any().downcast_ref::<TerrainExtension>())
            .map(|e| e.terrain_draw_mode)
            .collect()
    }

    #[test]
    fn a_layer_settles_the_draw_mode_left_open() {
        let open = Extensions::from_one(TerrainExtension::new());
        let resolved = resolve_terrain_draw_mode(&open, TerrainDrawMode::Offset);
        assert_eq!(modes(&resolved), vec![Some(TerrainDrawMode::Offset)]);
        // An explicit mode wins
        let explicit = Extensions::from_one(TerrainExtension::with_draw_mode(TerrainDrawMode::Drape));
        let resolved = resolve_terrain_draw_mode(&explicit, TerrainDrawMode::Offset);
        assert_eq!(modes(&resolved), vec![Some(TerrainDrawMode::Drape)]);
        // Resolving again gives an equal set, so a layer's props compare equal frame to frame
        let once = resolve_terrain_draw_mode(&open, TerrainDrawMode::Drape);
        assert!(resolve_terrain_draw_mode(&once, TerrainDrawMode::Drape) == once);
    }
}
