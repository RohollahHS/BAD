import bpy

# Define the new color (RGBA) for the material
#new_color = (0.230, 0.834, 1.0, 1.0)  # Example: cyan color
new_color = (1, 0.905, 0.05, 1.0)  # Alternative: yellow color
#new_color = (1, 0.1, 0.1, 1.0)

# Create a new material for colorization
material_name = "NewMaterial"
new_material = bpy.data.materials.get(material_name)
if not new_material:
    new_material = bpy.data.materials.new(name=material_name)
new_material.diffuse_color = new_color  # Set the material's color
new_material.use_nodes = False  # Disable nodes to make diffuse_color work

# Get all mesh objects with "frame" in their name
objects = [obj for obj in bpy.data.objects if obj.type == 'MESH' and "frame" in obj.name]

# Iterate through each object to set visibility keyframes and apply the material
for i, obj in enumerate(objects):
    # Assign the new material to the object
    if len(obj.data.materials) > 0:
        obj.data.materials[0] = new_material  # Replace the first material slot
    else:
        obj.data.materials.append(new_material)  # Add material if no slots exist

    # Set initial keyframes to hide the object by default
    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_viewport", frame=1)
    obj.keyframe_insert(data_path="hide_render", frame=1)

    # Make the object visible on its assigned frame (i+1)
    frame = i + 1  # Assign each object a frame starting from 1
    obj.hide_viewport = False
    obj.hide_render = False
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    obj.keyframe_insert(data_path="hide_render", frame=frame)

    # Immediately hide the object again on the next frame
    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_viewport", frame=frame + 1)
    obj.keyframe_insert(data_path="hide_render", frame=frame + 1)

print(f"Processed {len(objects)} objects with 'frame' in their names.")