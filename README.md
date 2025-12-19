# Christmas Ball Ornament Lithophane Generator

Generate 3D-printable Christmas ball ornaments with lithophane effects from images. When illuminated from inside with an LED, the varying wall thicknesses reveal your image.

## Installation

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Create a config file (see `example_config.yaml`) and run:

```bash
python ornament_generator.py --config configs/my_ornament.yaml
```

## Configuration Parameters

### Basic Information

**`name`**  
A descriptive name for your ornament (e.g., "family_photo_2025")

**`description`**  
Optional notes about this ornament

### Input/Output

**`image_path`**  
Path to your source image. Supports JPEG, PNG, and other common formats.

**`output_path`**  
Where to save the generated STL file (e.g., "output/ornament.stl")

### Ornament Dimensions

**`diameter`** (mm)  
The outer diameter of the spherical ornament. Default: 80.0

**`min_thickness`** (mm)  
Minimum wall thickness for the brightest parts of your image (lets most light through). Default: 1.0

**`max_thickness`** (mm)  
Maximum wall thickness for the darkest parts of your image (blocks light). Default: 3.0

### Mesh Resolution

**`resolution_lat`**  
Latitude resolution (vertical). Higher = more detail but slower generation. Default: 100

**`resolution_lon`**  
Longitude resolution (horizontal). Higher = more detail but slower generation. Default: 100

### Image Processing

**`image_width`** (pixels)  
Width to resize the image before processing. Default: 200

**`image_height`** (pixels)  
Height to resize the image before processing. Default: 200

**`invert_heightmap`** (true/false)  
Inverts the light/dark mapping. Set to `true` if your image appears reversed. Default: false

### Image Mapping Controls

**`image_scale`**  
Base zoom level. Default: 1.0
- `1.0` = auto-fit (image sized to sphere with correct aspect ratio)
- `< 1.0` = zoom out (e.g., 0.5 shows image at half size)
- `> 1.0` = zoom in (e.g., 2.0 magnifies image 2x)

```
  ╭───────╮  scale=1.0   ╭───────────╮  scale=0.5   ╭─────╮
  │ Image │  ────────▶   │   Image   │  ────────▶   │ Img │
  ╰───────╯  (auto-fit)  ╰───────────╯  (zoom out)  ╰─────╯
```

**`image_scale_x`**  
Width multiplier applied on top of `image_scale`. Default: 1.0
- `1.0` = no change
- `2.0` = stretch horizontally 2x
- `0.5` = squeeze to half width

**`image_scale_y`**  
Height multiplier applied on top of `image_scale`. Default: 1.0
- `1.0` = no change
- `2.0` = stretch vertically 2x
- `0.5` = squeeze to half height

**`latitude_correction`**  
Compensates for spherical distortion where images stretch more at the equator than at poles. Default: 0.0

```
Without correction (0.0):     With correction (1.0):
     Top (pole)                    Top (pole)
  ╭──( o )──╮                   ╭───(o)───╮
 │  <-W->   │ narrow           │  <-w->   │ narrow
 │ <--W-->  │ wider            │  <-w->   │ uniform
═╪═<---W--->╪═ equator        ═╪═<--w-->═╪═ uniform
 │ <--W-->  │ wider            │  <-w->   │ uniform
 │  <-W->   │ narrow           │  <-w->   │ narrow
  ╰─────────╯                   ╰─────────╯
```

- `0.0` = no correction (equator stretched horizontally)
- `1.0` = full correction (uniform sampling across all latitudes)
- `0.5` = partial correction (good starting point)

**`image_offset_x`**  
Horizontal positioning. Range: -1.0 to 1.0. Default: 0.0
- `-1.0` = shift left
- `0.0` = centered
- `1.0` = shift right

**`image_offset_y`**  
Vertical positioning. Range: -1.0 to 1.0. Default: 0.0
- `-1.0` = shift up
- `0.0` = centered
- `1.0` = shift down

**`image_rotation`**  
Rotates the image around the sphere (controls where 3D printer slicing starts). Default: 0.0

```
rotation=0°         rotation=90°        rotation=180°
   Front               Front                Front
╭────[A]────╮      ╭────[D]────╮       ╭────[C]────╮
│ B       D │      │ A       C │       │ D       B │
│     C     │      │     B     │       │     A     │
╰───────────╯      ╰───────────╯       ╰───────────╯
```

- `0` = default orientation
- `180` = rotated to opposite side (useful for printer seam positioning)
- Any value 0-360 degrees

**`image_tiling`** (true/false)  
Controls behavior when zoomed out or image doesn't cover full sphere. Default: true
- `true` = image repeats/tiles to fill the sphere
- `false` = areas beyond image bounds become black (max thickness, blocks light)

### Top Hole (for hanging)

**`add_top_hole`** (true/false)  
Whether to add a hole at the top for hanging. Default: true

**`top_hole_diameter`** (mm)  
Diameter of the top hole opening. Default: 5.0

**`top_hole_height`** (mm)  
How deep the hole extends into the sphere. Default: 10.0

### Bottom Hole (for LED/wiring)

**`add_bottom_hole`** (true/false)  
Whether to add a flat opening at the bottom for LED access. Default: false

**`bottom_hole_diameter`** (mm)  
Diameter of the bottom hole opening. Default: 5.0

## How It Works

1. Your image is converted to grayscale
2. Dark pixels → thick walls (block light)
3. Light pixels → thin walls (allow light through)
4. When lit from inside, the image becomes visible

## 3D Printing Tips

- Use transparent or translucent filament (clear PLA, natural PETG)
- Print with thin layers (0.1-0.2mm) for best detail
- No supports needed for the sphere itself
- Experiment with `min_thickness` and `max_thickness` to get the right contrast
