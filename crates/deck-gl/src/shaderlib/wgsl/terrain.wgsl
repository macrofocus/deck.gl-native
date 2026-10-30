// deck.gl
// SPDX-License-Identifier: MIT
// Copyright (c) vis.gl contributors
// Port of @deck.gl/extensions/src/terrain/shader-module.ts. A layer whose operation is
// `terrain` draws its ground elevation into a height map covering the view, and is drawn
// with the terrain cover the draped layers rendered into; a layer with the terrain extension
// is either lifted onto the height map (offset) or drawn only into the covers (drape).
//
// The map holds metres rather than deck.gl's common space units, so the sixteen bit float
// texture keeps about a metre of precision at any elevation. Maps and covers are laid out in
// flat (Mercator) common space, also under a globe, where positions are flattened first.

const TERRAIN_MODE_NONE: f32 = 0.0;
const TERRAIN_MODE_WRITE_HEIGHT_MAP: f32 = 1.0;
const TERRAIN_MODE_USE_HEIGHT_MAP: f32 = 2.0;
const TERRAIN_MODE_USE_COVER: f32 = 3.0;
const TERRAIN_MODE_USE_COVER_ONLY: f32 = 4.0;
const TERRAIN_MODE_SKIP: f32 = 5.0;

struct terrainUniforms {
  // Origin and size of the map or cover in absolute flat common space: x, y, width, height
  bounds: vec4<f32>,
  mode: f32,
};

@group(0) @binding(auto) var<uniform> terrain: terrainUniforms;
@group(0) @binding(auto) var terrain_map: texture_2d<f32>;
@group(0) @binding(auto) var terrain_mapSampler: sampler;

/// The absolute flat common space position of a common space position. Positions are
/// relative to the layer's origin in auto offset mode, so the origin is added back; under a
/// globe the sphere position is flattened, and its origin must not be added.
fn terrain_flatten(commonPosition: vec3<f32>) -> vec2<f32> {
  if (project.projectionMode == PROJECTION_MODE_GLOBE) {
    return project_common_position_to_flat(commonPosition);
  }
  return commonPosition.xy + project.commonOrigin.xy;
}

/// Where an absolute flat position falls in the map or cover, in texture coordinates.
fn terrain_texCoords(flatPosition: vec2<f32>) -> vec2<f32> {
  return (flatPosition - terrain.bounds.xy) / terrain.bounds.zw;
}

/// Sample the map or cover. It was drawn with clip space y up; texture rows start at the top.
fn terrain_sample(uv: vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(terrain_map, terrain_mapSampler, vec2<f32>(uv.x, 1.0 - uv.y), 0.0);
}

/// The ground elevation the map holds under a flat position, in metres, and whether it has one.
fn terrain_heightAt(flatPosition: vec2<f32>) -> vec2<f32> {
  let uv = terrain_texCoords(flatPosition);
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {
    return vec2<f32>(0.0, 0.0);
  }
  return vec2<f32>(terrain_sample(uv).r, 1.0);
}

fn terrain_setVertexPosition(position: vec4<f32>) -> vec4<f32> {
  if (terrain.mode == TERRAIN_MODE_SKIP) {
    // Draped layers are drawn into the terrain covers, never on screen
    return vec4<f32>(0.0);
  }
  let commonPosition = geometry.position.xyz;
  terrain_flatPosition = terrain_flatten(commonPosition);
  // The elevation this vertex sits at, in metres
  let unitsPerMeter = max(project.commonUnitsPerMeter.z, 1e-12);
  if (project.projectionMode == PROJECTION_MODE_GLOBE) {
    terrain_height = (length(commonPosition) - GLOBE_RADIUS) / unitsPerMeter;
  } else {
    terrain_height = (commonPosition.z + project.commonOrigin.z) / unitsPerMeter;
  }

  if (terrain.mode == TERRAIN_MODE_WRITE_HEIGHT_MAP) {
    // Draw the ground into the map: the position is the place it covers, flattened
    let uv = terrain_texCoords(terrain_flatPosition);
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
  }
  if (terrain.mode == TERRAIN_MODE_USE_HEIGHT_MAP) {
    // Lift the geometry onto the ground under its anchor
    var anchor = geometry.worldPosition;
    anchor.z = 0.0;
    let anchorCommon = project_position_vec3_f32(anchor);
    let height = terrain_heightAt(terrain_flatten(anchorCommon));
    if (height.y > 0.5) {
      let lift = height.x * unitsPerMeter;
      if (project.projectionMode == PROJECTION_MODE_GLOBE) {
        geometry.position = vec4<f32>(
          geometry.position.xyz + normalize(geometry.position.xyz) * lift,
          geometry.position.w
        );
      } else {
        geometry.position.z += lift;
      }
      return project_common_position_to_clipspace(geometry.position);
    }
  }
  return position;
}

fn terrain_filterColor(color: vec4<f32>) -> vec4<f32> {
  if (terrain.mode == TERRAIN_MODE_WRITE_HEIGHT_MAP) {
    // The map holds the elevation of the ground, not its colour
    return vec4<f32>(terrain_height, 0.0, 0.0, 1.0);
  }
  if (terrain.mode == TERRAIN_MODE_USE_COVER || terrain.mode == TERRAIN_MODE_USE_COVER_ONLY) {
    let pixel = terrain_sample(terrain_texCoords(terrain_flatPosition));
    if (terrain.mode == TERRAIN_MODE_USE_COVER_ONLY) {
      return pixel;
    }
    // pixel is premultiplied
    return pixel + color * (1.0 - pixel.a);
  }
  return color;
}
