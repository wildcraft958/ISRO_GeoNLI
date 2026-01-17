# Remote Sensing System Prompt Documentation

## Overview

This document describes the modifications made to the SAM3 agent system prompts to optimize them for **remote sensing and satellite imagery analysis**, replacing the original prompts designed for general-purpose real-world object segmentation.

## Files Created/Modified

| File | Purpose |
|------|---------|
| `system_prompt_remote_sensing.txt` | Main system prompt for satellite/aerial image segmentation |
| `system_prompt_iterative_checking_remote_sensing.txt` | Mask verification prompt for iterative refinement |
| `REMOTE_SENSING_PROMPT_DOCUMENTATION.md` | This documentation file |

## Key Differences from Original Prompts

### 1. Perspective and Viewing Geometry

**Original (General Purpose):**
- Assumed standard camera perspective with potential side views
- Objects identified by shape, color, and typical appearance
- Examples: "person", "dog", "car", "laptop"

**Remote Sensing Version:**
- Explicitly addresses **top-down/nadir view** perspective
- All objects described as they appear from above:
  - Buildings → rooftops (rectangles with shadows)
  - Vehicles → small rectangles
  - Roads → linear features
  - Water → dark uniform surfaces
- Added guidance on using shadows to infer height and structure type

### 2. Feature Terminology

**Original Examples:**
```
rope, bird beak, speed monitor, brown handbag, person torso, 
giraffe, person holding a blender, two ladies walking a dog
```

**Remote Sensing Examples:**
```
building, road, water body, forest, agricultural field, solar panel,
vehicle, ship, runway, parking lot, bridge, dam, wind turbine,
vegetation, bare soil, urban area, cropland, wetland
```

### 3. Spectral and Textural Characteristics

**Added section on remote sensing image interpretation:**

| Feature | Spectral Signature | Texture |
|---------|-------------------|---------|
| Healthy vegetation | Green tones | Irregular canopy texture |
| Stressed vegetation | Yellow-brown | Mixed patterns |
| Water bodies | Dark blue/black | Smooth, uniform |
| Bare soil | Tan to dark brown | Varies with moisture |
| Urban areas | Gray tones | Regular patterns, high contrast |
| Agricultural fields | Seasonal variation | Regular rows, geometric shapes |

### 4. Spatial Reasoning

**Original:**
- Relative positions: "left", "right", "second from right"
- General spatial references

**Remote Sensing Version:**
- Cardinal directions mapped to image coordinates:
  - Top = North
  - Bottom = South
  - Left = West
  - Right = East
- Geographic context: "northern buildings", "eastern water body"
- Spatial relationships to geographic features: "near the river", "adjacent to highway"

### 5. Scale and Resolution Awareness

**New section addressing multi-scale analysis:**

| Resolution | Detectable Features |
|------------|---------------------|
| High (<1m GSD) | Individual vehicles, small structures, detailed rooftops |
| Medium (1-10m) | Buildings, roads, field boundaries, large vehicles |
| Low (>10m) | Major features, land cover patterns, large water bodies |

Guidance added for adapting text_prompts based on apparent resolution.

### 6. Tool Usage Guidelines

#### segment_phrase Tool

**Original rules focused on:**
- Avoiding action-based descriptions
- Using singular noun phrases
- Not grounding secondary objects

**Remote Sensing modifications:**
- Use standard remote sensing terminology
- Try general terms before specific (e.g., "paved surface" before "runway")
- Consider resolution limitations
- Distinguish linear vs. area features
- Use visual descriptors: "dark water", "green vegetation"

#### ~~filter_masks_by_spatial_position Tool~~ (DISABLED)

**Now done in LLM thinking process, not as a tool call.**
- LLM analyzes mask positions visually
- Determines qualifying masks based on query criteria
- Calls `select_masks_and_return` directly with filtered list

#### ~~filter_masks_by_relative_position Tool~~ (DISABLED)

**Now done in LLM thinking process, not as a tool call.**
- LLM identifies reference feature visually
- Analyzes each mask's position relative to reference
- Calls `select_masks_and_return` directly with qualifying masks

#### segment_phrase_in_region Tool

**Use cases for remote sensing:**
- Focus on specific geographic areas
- Example: Segment buildings only within an industrial zone
- Define regions around reference features (water body vicinity, road corridor)

#### filter_masks_by_attributes Tool

**Remote sensing color interpretation:**

| Query Term | Expected Color | Feature Type |
|------------|----------------|--------------|
| "green" | Green | Vegetation |
| "blue"/"dark" | Blue-black | Water |
| "brown" | Brown | Bare soil, dry vegetation |
| "gray" | Gray | Paved surfaces, urban |
| "white" | White | Clouds, snow, bright roofs |

### 7. Thinking Process Modifications

#### Scenario 1 (Initial Analysis)

**Original focus:**
- Describe objects, people, actions
- Identify primary vs. secondary targets

**Remote sensing focus:**
- Describe land cover types, structures, infrastructure
- Identify spectral and textural patterns
- Note image resolution and extent
- Consider top-down perspective interpretation

#### Scenario 2 (Mask Evaluation)

**Original focus:**
- Match masks to described objects
- Verify object identity

**Remote sensing focus:**
- Verify geographic feature type
- Check spatial position against query constraints
- Assess boundary accuracy for land cover transitions
- Consider shadow effects and spectral ambiguity

### 8. Error Handling and Edge Cases

**New guidance for remote sensing-specific issues:**

1. **Shadow Confusion**
   - Shadows can be mistaken for water
   - Use shadow shape and attachment to tall objects to differentiate

2. **Spectral Ambiguity**
   - Paved surfaces vs. water (both dark)
   - Bare soil vs. unpaved roads (both brown)
   - Use context and texture to disambiguate

3. **Scale Mismatch**
   - Query for "building" might get individual structure or building complex
   - Verify scale matches query intent

4. **Seasonal Variation**
   - Vegetation appearance changes with season
   - Water levels fluctuate
   - Agricultural fields change throughout growing season

## Iterative Checking Prompt Changes

### Original Purpose
Verify mask correctness for everyday objects with zoomed examination.

### Remote Sensing Adaptations

1. **Feature Type Verification**
   - Check mask covers correct land cover type
   - Verify spectral signature matches expected feature

2. **Boundary Assessment**
   - Different precision expectations:
     - Buildings: Sharp boundaries expected
     - Forest edges: Fuzzy boundaries acceptable
     - Water bodies: Clear shoreline delineation

3. **Accept/Reject Criteria**

| Accept If | Reject If |
|-----------|-----------|
| Feature type matches query | Wrong feature type |
| Boundaries reasonably accurate | Major over/under-segmentation |
| Location matches constraints | Wrong geographic position |
| Target feature (not reference) | Segmented reference instead of target |

4. **Common Error Patterns**
   - Shadow misclassification
   - Partial overlap with adjacent features
   - Missing instances in multi-object queries

## Available Tools Summary

| Tool | When to Use in Remote Sensing |
|------|------------------------------|
| `segment_phrase` | Initial segmentation of geographic features |
| `examine_each_mask` | Dense urban areas, clustered objects, overlapping fields |
| `select_masks_and_return` | Final selection after verification (includes spatially-filtered masks) |
| `report_no_mask` | Feature type doesn't exist in image |
| `segment_phrase_in_region` | Focus on specific geographic area |
| `filter_masks_by_attributes` | Filter by color (vegetation=green, water=dark) |

### Spatial Filtering - Done in Thinking (NOT as separate tools)

⚠️ **IMPORTANT**: `filter_masks_by_spatial_position` and `filter_masks_by_relative_position` are **DISABLED**.

Spatial and relative filtering is now done **in the LLM's thinking process**, not via separate tool calls.

#### How Spatial Filtering Works Now

1. LLM receives masks from `segment_phrase`
2. In `<think>` block, LLM analyzes each mask's position
3. LLM determines which masks qualify based on query criteria
4. LLM calls `select_masks_and_return` with ONLY qualifying masks

#### Absolute Position Reasoning (in thinking)

For queries like "northern buildings", "leftmost road", "top-right corner":

```
<think>
Query asks for "leftmost" objects.
- Mask 1: ~20% from left → QUALIFIES (leftmost)
- Mask 2: ~50% from left → DOES NOT QUALIFY
- Mask 3: ~80% from left → DOES NOT QUALIFY

Final selection: [1]
</think>
select_masks_and_return(final_answer_masks=[1])
```

#### Relative Position Reasoning (in thinking)

For queries like "buildings near the lake", "road west of factory":

```
<think>
Query asks for objects "near the lake".
Reference: Lake is visible at ~70% from left.
- Mask 1: ~20% from left → FAR from lake (~50% away) - DOES NOT QUALIFY
- Mask 2: ~65% from left → NEAR lake (~5% away) - QUALIFIES
- Mask 3: ~75% from left → NEAR lake (~5% away) - QUALIFIES

Final selection: [2, 3]
</think>
select_masks_and_return(final_answer_masks=[2, 3])
```

## Example Query Transformations

### Original Style → Remote Sensing Style

| Original Query | Remote Sensing Equivalent |
|----------------|---------------------------|
| "The person on the left" | "The building in the western part" |
| "Red car" | "Large industrial building" |
| "Dog lying down" | "Flooded agricultural field" |
| "Two ladies walking" | "Road network connecting settlements" |
| "Guy with white hat" | "Solar farm with white panels" |

### SAM3 Text Prompt Best Practices

SAM3 uses a vision-language model (similar to CLIP) that works best with **simple, direct noun phrases**. The model understands common objects and visual attributes.

#### Effective Prompt Patterns

| Pattern | Examples | When to Use |
|---------|----------|-------------|
| Single noun | "building", "road", "water", "tree" | First attempt, broad matching |
| Noun + color | "green field", "blue water", "gray roof" | Distinguish by color |
| Noun + size | "large building", "small pond" | Filter by relative size |
| Compound noun | "parking lot", "solar panel", "swimming pool" | Specific feature types |
| Material | "concrete surface", "asphalt road" | Material-based distinction |

#### Fallback Hierarchies

When a prompt fails, try these alternatives in order:

| Target | Primary → Fallback Options |
|--------|---------------------------|
| Buildings | "building" → "structure" → "roof" → "gray rectangle" |
| Water | "water" → "lake" → "pond" → "blue area" → "dark surface" |
| Roads | "road" → "street" → "pavement" → "gray line" → "path" |
| Vegetation | "vegetation" → "trees" → "forest" → "green area" |
| Vehicles | "vehicle" → "car" → "truck" → "small object" |

#### Text Prompt Examples

| Query | Recommended text_prompt | Why |
|-------|------------------------|-----|
| "All buildings in the image" | "building" | Simple noun, SAM3 handles well |
| "The main road" | "road" | Use spatial filtering afterward |
| "Water features" | "water" | Simple noun works better than compound |
| "Farmland" | "field" | Simpler than "agricultural field" |
| "Parking areas" | "parking lot" | Common compound noun |
| "The airport" | "runway" | Specific detectable feature |
| "Green vegetation" | "green field" or "trees" | Color adjective helps |
| "Storage facilities" | "tank" or "warehouse" | Simple specific nouns |

#### Prompts to Avoid

| Bad Prompt | Why | Better Alternative |
|------------|-----|-------------------|
| "building near the river" | Relational description | "building" + spatial filter |
| "flooded area" | State-based | "water" or "wet ground" |
| "photovoltaic installation" | Too technical | "solar panel" |
| "Boeing 747" | Too specific | "airplane" or "aircraft" |
| "road connecting cities" | Action-based | "road" |

## Confidence-Based Examination Strategy

The prompts implement an efficient examination strategy to avoid over-analyzing masks:

### Quick Scan First
Before detailed analysis, categorize each mask by confidence:

| Confidence | Criteria | Action |
|------------|----------|--------|
| **HIGH ✓** | Clearly correct type, location, boundaries | Accept directly |
| **MEDIUM ~** | Probably correct but uncertain | Brief examination |
| **LOW ?** | May be wrong or ambiguous | Detailed analysis |

### Decision Flow

```
For each mask:
├── Quick scan from overview
├── Assign confidence (HIGH/MEDIUM/LOW)
│
├── If ALL HIGH:
│   └── Skip examine_each_mask → select_masks_and_return
│
├── If SOME MEDIUM/LOW:
│   ├── Call examine_each_mask
│   ├── Focus on uncertain masks only
│   └── Quick confirm confident masks
│
└── Make final selection
```

### Iterative Checking Prompt Changes

The `system_prompt_iterative_checking_remote_sensing.txt` has been optimized for efficiency:

1. **Quick Assessment First**: 5-second scan before detailed analysis
2. **Confidence-Based Depth**: More analysis for uncertain cases, less for obvious ones
3. **Remote Sensing Quick Checks**:
   - Feature type verification (building, water, vegetation, etc.)
   - Spatial position checks (absolute and relative)
   - Boundary quality assessment
4. **Common Error Awareness**: Shadow confusion, spectral ambiguity, scale mismatch

### Hierarchical Segmentation Strategy

For small objects that fail direct segmentation:

```
Query: "segment cars in parking lot"

Step 1: Try "car" or "vehicle"
        ↓ (fails - too small)
Step 2: Segment container "parking lot"
        ↓ (get parking lot bbox)
Step 3: segment_phrase_in_region("vehicle", bbox=parking_lot_area)
        ↓
Step 4: examine_each_mask if needed
```

Examples:
- Cars → Parking lot → segment_phrase_in_region
- Ships → Harbor/Water → segment_phrase_in_region
- Trees → Park/Forest → segment_phrase_in_region
- Planes → Airport/Runway → segment_phrase_in_region

## Configuration Notes

To use the remote sensing prompts:

1. **Main prompt**: Load `system_prompt_remote_sensing.txt` instead of `system_prompt.txt`

2. **Iterative checking**: Load `system_prompt_iterative_checking_remote_sensing.txt` instead of `system_prompt_iterative_checking.txt`

3. **Agent configuration**: Ensure the agent loads the appropriate prompt files based on the image type (satellite/aerial vs. general photography).

## Performance Considerations

1. **Resolution Impact**: Low-resolution imagery may require more general text_prompts
2. **Large Images**: May benefit from `segment_phrase_in_region` to focus processing
3. **Complex Scenes**: Urban areas with many buildings may need `examine_each_mask` for verification
4. **Ambiguous Features**: May require multiple segment_phrase attempts with different terminology

## Future Improvements

1. **Multi-spectral Support**: Extend prompts for imagery with non-RGB bands (NIR, SWIR)
2. **Change Detection**: Add guidance for temporal comparison queries
3. **3D Analysis**: Incorporate DSM/DTM data interpretation
4. **Object Counting**: Enhance for inventory-type queries ("count all buildings")

