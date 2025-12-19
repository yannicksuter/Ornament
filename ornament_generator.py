"""
Christmas Ball Ornament Lithophane Generator

This script generates an STL model of a Christmas ball ornament with a lithophane
effect. The heightmap from an input image creates varying wall thicknesses that
become visible when illuminated from inside with an LED.
"""

import numpy as np
from PIL import Image
from stl import mesh
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime


class OrnamentConfig:
    """Configuration for ornament generation."""
    
    def __init__(self, config_dict=None):
        """Initialize config from dictionary or defaults."""
        if config_dict is None:
            config_dict = {}
        
        # Basic parameters
        self.image_path = config_dict.get('image_path', None)
        self.output_path = config_dict.get('output_path', 'ornament.stl')
        
        # Ornament dimensions
        self.diameter = config_dict.get('diameter', 80.0)
        self.min_thickness = config_dict.get('min_thickness', 1.0)
        self.max_thickness = config_dict.get('max_thickness', 3.0)
        
        # Mesh resolution
        self.resolution_lat = config_dict.get('resolution_lat', 100)
        self.resolution_lon = config_dict.get('resolution_lon', 100)
        
        # Image processing
        self.image_width = config_dict.get('image_width', 200)
        self.image_height = config_dict.get('image_height', 200)
        self.invert_heightmap = config_dict.get('invert_heightmap', False)
        
        # Image mapping controls
        self.image_scale = config_dict.get('image_scale', 1.0)  # Base scale factor for zooming
        self.image_scale_x = config_dict.get('image_scale_x', 1.0)  # Horizontal multiplier (multiplied with image_scale)
        self.image_scale_y = config_dict.get('image_scale_y', 1.0)  # Vertical multiplier (multiplied with image_scale)
        self.latitude_correction = config_dict.get('latitude_correction', 0.0)  # Spherical distortion correction (0.0-1.0)
        self.image_offset_x = config_dict.get('image_offset_x', 0.0)  # Horizontal offset (0.0-1.0)
        self.image_offset_y = config_dict.get('image_offset_y', 0.0)  # Vertical offset (0.0-1.0)
        self.image_rotation = config_dict.get('image_rotation', 0.0)  # Rotation in degrees (0-360)
        self.image_tiling = config_dict.get('image_tiling', True)  # Whether to tile/wrap image
        
        # Top hole for hanging
        self.add_top_hole = config_dict.get('add_top_hole', True)
        self.top_hole_diameter = config_dict.get('top_hole_diameter', 5.0)
        self.top_hole_height = config_dict.get('top_hole_height', 10.0)
        
        # Bottom hole for LED/wiring
        self.add_bottom_hole = config_dict.get('add_bottom_hole', False)
        self.bottom_hole_diameter = config_dict.get('bottom_hole_diameter', 5.0)
        
        # Metadata
        self.name = config_dict.get('name', 'ornament')
        self.description = config_dict.get('description', '')
        self.created_at = config_dict.get('created_at', datetime.now().isoformat())
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'image_path': self.image_path,
            'output_path': self.output_path,
            'diameter': self.diameter,
            'min_thickness': self.min_thickness,
            'max_thickness': self.max_thickness,
            'resolution_lat': self.resolution_lat,
            'resolution_lon': self.resolution_lon,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'invert_heightmap': self.invert_heightmap,
            'image_scale': self.image_scale,
            'image_scale_x': self.image_scale_x,
            'image_scale_y': self.image_scale_y,
            'latitude_correction': self.latitude_correction,
            'image_offset_x': self.image_offset_x,
            'image_offset_y': self.image_offset_y,
            'image_rotation': self.image_rotation,
            'image_tiling': self.image_tiling,
            'add_top_hole': self.add_top_hole,
            'top_hole_diameter': self.top_hole_diameter,
            'top_hole_height': self.top_hole_height,
            'add_bottom_hole': self.add_bottom_hole,
            'bottom_hole_diameter': self.bottom_hole_diameter,
            'created_at': self.created_at,
        }
    
    @classmethod
    def from_file(cls, config_path):
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return cls(config_dict)
    
    def save(self, config_path):
        """Save configuration to YAML or JSON file."""
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            elif config_path.suffix == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        print(f"Configuration saved to: {config_path}")


class OrnamentGenerator:
    def __init__(self, config):
        """
        Initialize the ornament generator.
        
        Args:
            config: OrnamentConfig object with all parameters
        """
        self.config = config
        self.radius = config.diameter / 2
        
    def load_and_process_image(self):
        """Load image and convert to grayscale heightmap."""
        img = Image.open(self.config.image_path)
        
        # Store original aspect ratio for auto-fit calculation
        self.original_aspect_ratio = img.width / img.height
        
        # Convert to grayscale
        img_gray = img.convert('L')
        # Resize to configured resolution
        img_gray = img_gray.resize(
            (self.config.image_width, self.config.image_height), 
            Image.Resampling.LANCZOS
        )
        # Normalize to 0-1 range
        heightmap = np.array(img_gray) / 255.0
        
        # Optionally invert heightmap
        if self.config.invert_heightmap:
            heightmap = 1.0 - heightmap
        
        return heightmap
    
    def create_sphere_mesh(self):
        """
        Create a spherical mesh with lithophane effect.
        
        Returns:
            List of triangular faces with vertices
        """
        heightmap = self.load_and_process_image()
        
        # Use config resolution
        res_lat = self.config.resolution_lat
        res_lon = self.config.resolution_lon
        
        # Create spherical coordinates
        # theta: longitude (0 to 2*pi)
        # phi: latitude (0 to pi)
        theta = np.linspace(0, 2 * np.pi, res_lon, endpoint=False)
        phi = np.linspace(0, np.pi, res_lat)
        
        # Create vertex arrays
        outer_vertices = []
        inner_vertices = []
        
        # Calculate auto-fit scaling based on image aspect ratio
        # Sphere unwrapped is 2:1 aspect ratio (360° longitude : 180° latitude)
        sphere_aspect_ratio = 2.0
        
        # Determine base scale to fit image without distortion
        if self.original_aspect_ratio > sphere_aspect_ratio:
            # Image is wider - fit by width
            base_scale = self.original_aspect_ratio / sphere_aspect_ratio
        else:
            # Image is taller - fit by height
            base_scale = 1.0
        
        # Determine effective scaling for x and y
        # Combine: base_scale (auto-fit) * image_scale (user zoom) * scale_x/y (stretch/squash)
        effective_scale_x = base_scale * self.config.image_scale * self.config.image_scale_x
        effective_scale_y = self.config.image_scale * self.config.image_scale_y
        
        # Convert rotation from degrees to 0-1 range (normalized to longitude)
        rotation_offset = (self.config.image_rotation % 360) / 360.0
        
        # Generate vertices
        for j in range(res_lat):
            for i in range(res_lon):
                # Map to heightmap coordinates with scaling and offset
                # Normalize to 0-1 range
                u = i / res_lon
                v = j / res_lat
                
                # Apply rotation by offsetting u coordinate
                u = (u + rotation_offset) % 1.0
                
                # Calculate latitude-based correction factor
                # At equator (phi=π/2), sin(phi)=1 (no correction needed)
                # Near poles (phi→0 or π), sin(phi)→0 (maximum correction needed)
                phi_val = phi[j]
                sin_phi = np.sin(phi_val)
                # Avoid division by zero at poles and blend correction strength
                if sin_phi < 0.01:
                    sin_phi = 0.01
                latitude_factor = 1.0 + self.config.latitude_correction * (1.0 / sin_phi - 1.0)
                
                # Apply effective scale (auto-fit + user adjustment) - center the scaling
                # Divide by scale: <1.0 zooms out (samples more), >1.0 zooms in (samples less)
                u_scaled = (u - 0.5) / (effective_scale_x * latitude_factor) + 0.5
                v_scaled = (v - 0.5) / effective_scale_y + 0.5
                
                # Apply offset
                u_scaled += self.config.image_offset_x
                v_scaled += self.config.image_offset_y
                
                # Check if coordinates are within bounds
                if self.config.image_tiling:
                    # Tile/wrap the image
                    img_x = int(u_scaled * heightmap.shape[1]) % heightmap.shape[1]
                    img_y = int(v_scaled * heightmap.shape[0]) % heightmap.shape[0]
                    heightmap_value = heightmap[img_y, img_x]
                else:
                    # No tiling - use max_thickness (black) for out-of-bounds areas
                    if 0 <= u_scaled <= 1 and 0 <= v_scaled <= 1:
                        img_x = int(u_scaled * heightmap.shape[1])
                        img_y = int(v_scaled * heightmap.shape[0])
                        # Clamp to valid indices
                        img_x = min(img_x, heightmap.shape[1] - 1)
                        img_y = min(img_y, heightmap.shape[0] - 1)
                        heightmap_value = heightmap[img_y, img_x]
                    else:
                        # Out of bounds - use black (0.0 = max thickness)
                        heightmap_value = 0.0
                
                # Get thickness from heightmap (darker = thicker)
                thickness = self.config.min_thickness + (1 - heightmap_value) * (
                    self.config.max_thickness - self.config.min_thickness
                )
                
                # Calculate spherical coordinates
                theta_val = theta[i]
                phi_val = phi[j]
                
                # Outer surface vertex
                x_outer = self.radius * np.sin(phi_val) * np.cos(theta_val)
                y_outer = self.radius * np.sin(phi_val) * np.sin(theta_val)
                z_outer = self.radius * np.cos(phi_val)
                outer_vertices.append([x_outer, y_outer, z_outer])
                
                # Inner surface vertex
                inner_radius = self.radius - thickness
                x_inner = inner_radius * np.sin(phi_val) * np.cos(theta_val)
                y_inner = inner_radius * np.sin(phi_val) * np.sin(theta_val)
                z_inner = inner_radius * np.cos(phi_val)
                inner_vertices.append([x_inner, y_inner, z_inner])
        
        outer_vertices = np.array(outer_vertices)
        inner_vertices = np.array(inner_vertices)
        
        # Generate faces
        faces = []
        
        def get_index(lat, lon):
            """Get vertex index for given latitude and longitude indices."""
            lon = lon % res_lon  # Wrap around longitude
            return lat * res_lon + lon
        
        # Create faces for outer surface
        for j in range(res_lat - 1):
            for i in range(res_lon):
                # Two triangles per quad
                v0 = get_index(j, i)
                v1 = get_index(j, i + 1)
                v2 = get_index(j + 1, i)
                v3 = get_index(j + 1, i + 1)
                
                # Triangle 1
                faces.append([outer_vertices[v0], outer_vertices[v1], outer_vertices[v2]])
                # Triangle 2
                faces.append([outer_vertices[v1], outer_vertices[v3], outer_vertices[v2]])
        
        # Create faces for inner surface (reversed winding for correct normals)
        for j in range(res_lat - 1):
            for i in range(res_lon):
                v0 = get_index(j, i)
                v1 = get_index(j, i + 1)
                v2 = get_index(j + 1, i)
                v3 = get_index(j + 1, i + 1)
                
                # Triangle 1 (reversed winding)
                faces.append([inner_vertices[v0], inner_vertices[v2], inner_vertices[v1]])
                # Triangle 2 (reversed winding)
                faces.append([inner_vertices[v1], inner_vertices[v2], inner_vertices[v3]])
        
        return np.array(faces)
    
    def add_top_hole_to_mesh(self, faces):
        """
        Add a cylindrical hole at the top of the sphere for hanging.
        
        Args:
            faces: Existing mesh faces
            
        Returns:
            Modified faces array with top hole
        """
        if not self.config.add_top_hole:
            return faces
        
        hole_faces = []
        hole_radius = self.config.top_hole_diameter / 2
        hole_height = self.config.top_hole_height
        resolution = 16  # Number of segments around the hole
        
        # Top of sphere is at z = radius
        top_z = self.radius
        bottom_z = top_z - hole_height
        
        # Create cylinder vertices
        angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
        
        # Outer cylinder (at top hole diameter)
        top_outer = [[hole_radius * np.cos(a), hole_radius * np.sin(a), top_z] for a in angles]
        bottom_outer = [[hole_radius * np.cos(a), hole_radius * np.sin(a), bottom_z] for a in angles]
        
        # Create cylinder side faces
        for i in range(resolution):
            next_i = (i + 1) % resolution
            
            # Two triangles per quad segment
            hole_faces.append([top_outer[i], bottom_outer[i], top_outer[next_i]])
            hole_faces.append([top_outer[next_i], bottom_outer[i], bottom_outer[next_i]])
        
        # Combine with existing faces
        if len(hole_faces) > 0:
            return np.vstack([faces, np.array(hole_faces)])
        return faces
    
    def add_bottom_hole_to_mesh(self, faces):
        """
        Cut a circular opening at the bottom of the sphere and close it with a flat cap.
        Creates a watertight mesh with no open volumes.
        
        Args:
            faces: Existing mesh faces
            
        Returns:
            Modified faces array with bottom hole and closing cap
        """
        if not self.config.add_bottom_hole:
            return faces
        
        hole_radius = self.config.bottom_hole_diameter / 2
        
        # Calculate the z-height where the hole should be cut
        # Using the sphere equation: x^2 + y^2 + z^2 = r^2
        # At the hole radius: hole_radius^2 + z^2 = radius^2
        z_cut = -np.sqrt(max(0, self.radius**2 - hole_radius**2))
        
        # Filter out faces that are below the cut plane
        filtered_faces = []
        edge_vertices = []  # Vertices near the cut edge
        
        for face in faces:
            # Check if all vertices of the triangle are above the cut
            z_values = [v[2] for v in face]
            if all(z > z_cut for z in z_values):
                filtered_faces.append(face)
            elif any(z > z_cut for z in z_values):
                # Triangle crosses the cut plane - clip it
                # For simplicity, keep triangles if at least 2 vertices are above
                if sum(z > z_cut for z in z_values) >= 2:
                    filtered_faces.append(face)
        
        # Create a flat circular cap at the bottom
        resolution = 32  # Number of segments for the circular cap
        angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
        
        # Create vertices around the circle at the cut height
        circle_vertices = []
        for angle in angles:
            x = hole_radius * np.cos(angle)
            y = hole_radius * np.sin(angle)
            circle_vertices.append([x, y, z_cut])
        
        # Center point of the cap
        center = [0, 0, z_cut]
        
        # Create triangular faces for the cap (fan triangulation from center)
        cap_faces = []
        for i in range(resolution):
            next_i = (i + 1) % resolution
            # Create triangle from center to edge
            cap_faces.append([center, circle_vertices[i], circle_vertices[next_i]])
        
        # Combine filtered faces with cap faces
        result_faces = filtered_faces + cap_faces
        return np.array(result_faces)
    
    def generate_stl(self):
        """Generate and save the STL file."""
        print(f"Configuration: {self.config.name}")
        print(f"Loading image: {self.config.image_path}")
        print(f"Generating ornament mesh (resolution: {self.config.resolution_lat}x{self.config.resolution_lon})...")
        
        # Generate mesh faces
        faces = self.create_sphere_mesh()
        
        # Add holes if enabled
        holes_added = []
        if self.config.add_top_hole:
            faces = self.add_top_hole_to_mesh(faces)
            holes_added.append("top hole")
        if self.config.add_bottom_hole:
            faces = self.add_bottom_hole_to_mesh(faces)
            holes_added.append("bottom hole")
        
        if holes_added:
            print(f"Generated {len(faces)} triangular faces (with {', '.join(holes_added)})")
        else:
            print(f"Generated {len(faces)} triangular faces")
        
        # Create the mesh object
        ornament_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        
        # Assign vertices to the mesh
        for i, face in enumerate(faces):
            for j in range(3):
                ornament_mesh.vectors[i][j] = face[j]
        
        # Create output directory if it doesn't exist
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the STL file
        print(f"Saving STL to: {self.config.output_path}")
        ornament_mesh.save(str(output_path))
        
        print(f"STL generation complete!")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Christmas ball ornament lithophane STL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from config file
  python ornament_generator.py --config my_ornament.yaml
  
  # Generate with command line arguments
  python ornament_generator.py --image photo.jpg --diameter 80 --output ornament.stl
  
  # Save current config for later reuse
  python ornament_generator.py --image photo.jpg --save-config my_ornament.yaml
        """
    )
    
    # Config file options
    parser.add_argument('--config', '-c', help='Path to config file (YAML or JSON)')
    parser.add_argument('--save-config', help='Save current configuration to file')
    
    # Basic parameters
    parser.add_argument('--image', '-i', help='Path to input image')
    parser.add_argument('--output', '-o', help='Output STL file path (default: ornament.stl)')
    parser.add_argument('--name', help='Name for this ornament configuration')
    
    # Dimensions
    parser.add_argument('--diameter', type=float, help='Ornament diameter in mm (default: 80)')
    parser.add_argument('--min-thickness', type=float, help='Minimum wall thickness in mm (default: 1.0)')
    parser.add_argument('--max-thickness', type=float, help='Maximum wall thickness in mm (default: 3.0)')
    
    # Mesh resolution
    parser.add_argument('--resolution', type=int, help='Mesh resolution for lat/lon (default: 100)')
    
    # Image processing
    parser.add_argument('--invert', action='store_true', help='Invert the heightmap')
    
    args = parser.parse_args()
    
    # Load config from file or create from arguments
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = OrnamentConfig.from_file(args.config)
        # Command line args can override config file
        if args.image:
            config.image_path = args.image
        if args.output:
            config.output_path = args.output
    else:
        # Create config from command line arguments
        if not args.image:
            parser.error("--image is required when not using --config")
        
        config_dict = {
            'image_path': args.image,
            'output_path': args.output or 'ornament.stl',
            'name': args.name or Path(args.image).stem,
        }
        
        if args.diameter:
            config_dict['diameter'] = args.diameter
        if args.min_thickness:
            config_dict['min_thickness'] = args.min_thickness
        if args.max_thickness:
            config_dict['max_thickness'] = args.max_thickness
        if args.resolution:
            config_dict['resolution_lat'] = args.resolution
            config_dict['resolution_lon'] = args.resolution
        if args.invert:
            config_dict['invert_heightmap'] = True
        
        config = OrnamentConfig(config_dict)
    
    # Save config if requested
    if args.save_config:
        config.save(args.save_config)
        print(f"Configuration saved. You can regenerate this ornament with:")
        print(f"  python ornament_generator.py --config {args.save_config}")
        return
    
    # Validate required parameters
    if not config.image_path:
        parser.error("Image path must be specified in config or via --image")
    
    # Generate the ornament
    generator = OrnamentGenerator(config)
    generator.generate_stl()


if __name__ == '__main__':
    main()
